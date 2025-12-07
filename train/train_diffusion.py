"""
Training script for Conditional Diffusion Model
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import json
from torchvision.utils import make_grid, save_image

try:
    from torch.amp import GradScaler as _GradScaler
except ImportError:  # Older PyTorch fallback
    from torch.cuda.amp import GradScaler as _GradScaler


def _create_grad_scaler(device, enabled):
    if device.type != 'cuda' or not enabled:
        return _GradScaler(enabled=False)
    try:
        return _GradScaler(device_type='cuda', enabled=True)
    except TypeError:
        return _GradScaler(enabled=True)

# Set CUDA device before any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.conditional_diffusion import ConditionalDiffusionModel
from data.data_loader import create_data_loaders
from data.dataset_utils import ensure_dataset
from device_utils import get_device


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


@torch.no_grad()
def generate_comparison_diffusion(model, disease_classes, device, real_images_dict, num_inference_steps=50, out_path="output/comparison.png"):
    """Generate comparison grid: real vs fake images for each class."""
    model.eval()
    num_classes = len(disease_classes)
    imgs = []
    
    for c in range(num_classes):
        # Real image
        real_img = real_images_dict[c]
        imgs.append(real_img.cpu())
        
        # Fake image
        class_labels = torch.full((1,), c, dtype=torch.long, device=device)
        generated = model.sample(class_labels, batch_size=1, num_inference_steps=num_inference_steps)
        imgs.append(generated[0].cpu())
    
    grid = make_grid(torch.cat(imgs, 0), nrow=2, normalize=True, value_range=(-1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs, scaler=None, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None and scaler.is_enabled()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", 
                unit='batch', ncols=80, position=1, leave=False, mininterval=0.5)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            predicted_noise, target_noise, t = model(images, labels)
            loss = criterion(predicted_noise, target_noise)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Increased from 1.0
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Increased from 1.0
            optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with detailed info
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device, amp_enabled=False):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation", unit='batch', ncols=80, position=2, leave=False, mininterval=0.5)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        if device.type == 'cuda':
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=torch.float16)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            predicted_noise, target_noise, t = model(images, labels)
            loss = criterion(predicted_noise, target_noise)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'val_loss': f'{loss.item():.4f}',
            'avg_val_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision('high')

    # Ensure dataset exists (download if needed)
    dataset_dir = Path(args.ham_csv).parent if Path(args.ham_csv).is_absolute() else Path("dataset")
    if not ensure_dataset(dataset_dir=str(dataset_dir)):
        print("Error: Could not ensure dataset is available. Please check the download.")
        sys.exit(1)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, _, disease_classes = create_data_loaders(
        ham_metadata_path=args.ham_csv,
        ham_img_dir=args.ham_img_dir,
        bcn_metadata_path=args.bcn_csv,
        bcn_img_dir=args.bcn_img_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        balance_classes=not args.no_balance,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        max_samples_per_class=args.max_samples_per_class,
    )

    num_classes = len(disease_classes)
    print(f"Detected {num_classes} classes: {list(disease_classes.keys())}")

    # Create model
    print(f"\nCreating model...")
    print(f"  LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    try:
        channel_mult = tuple(int(x.strip()) for x in args.channel_mult.split(','))
        if not channel_mult:
            raise ValueError
    except ValueError:
        print(f"Invalid channel_mult value '{args.channel_mult}'. Expected comma-separated integers like '1,2,4,8'.")
        sys.exit(1)
    print(f"  Base channels: {args.model_channels}")
    print(f"  Channel mult: {channel_mult}")
    print(f"  Residual blocks: {args.num_res_blocks}")
    model = ConditionalDiffusionModel(
        image_size=args.img_size,
        num_classes=num_classes,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=channel_mult,
        num_timesteps=args.num_timesteps,
        beta_schedule='linear',
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = _create_grad_scaler(device, args.amp)
    
    # Initialize EMA
    ema = EMA(model, decay=0.999)  # Reduced from 0.9999 for stability
    print("  EMA enabled (decay=0.9999)")

    if device.type == 'cuda' and not args.amp:
        print("\n  [TIP] For faster training, use --amp to enable Automatic Mixed Precision!\n")
    
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect one real image per class for comparison
    real_images_dict = {}
    for images, labels, _ in train_loader:
        for img, label in zip(images, labels):
            if label.item() not in real_images_dict:
                real_images_dict[label.item()] = img
        if len(real_images_dict) == num_classes:
            break

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    # Initialize these so that zero-epoch runs don't crash when saving final summary/checkpoint
    train_loss = None
    val_loss = None
    best_checkpoint_path = None
    
    epochs_pbar = tqdm(total=args.epochs, desc="Training", unit="epoch", ncols=80, position=0, leave=False)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs, scaler=scaler, ema=ema)
        
        # Validate
        if epoch % 2 == 0 or epoch == 1:
            # Use EMA for validation
            ema.apply_shadow()
            val_loss = validate(model, val_loader, criterion, device, amp_enabled=scaler.is_enabled())
            ema.restore()
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} (EMA)")
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best model! (val_loss: {val_loss:.4f})")
                # Save best model
                best_checkpoint_path = os.path.join(args.checkpoint_dir, "diffusion_best.pt")
                
                # Save with EMA weights
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'disease_classes': disease_classes,
                    'num_classes': num_classes,
                    'ema_shadow': ema.shadow,  # Save EMA shadow
                    'model_config': {
                        'image_size': args.img_size,
                        'num_timesteps': args.num_timesteps,
                        'use_lora': args.use_lora,
                        'lora_r': args.lora_r,
                        'lora_alpha': args.lora_alpha,
                        'lora_dropout': args.lora_dropout,
                        'model_channels': args.model_channels,
                        'channel_mult': args.channel_mult,
                        'num_res_blocks': args.num_res_blocks,
                    }
                }, best_checkpoint_path)
                ema.restore()
                print(f"  Best model saved: {best_checkpoint_path}")
        else:
            print(f"\nEpoch {epoch}/{args.epochs}: Train Loss: {train_loss:.4f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"diffusion_epoch_{epoch}.pt")
            
            # Save with EMA weights
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'train_loss': train_loss,
                'val_loss': val_loss if epoch % 2 == 0 else None,
                'disease_classes': disease_classes,
                'num_classes': num_classes,
                'ema_shadow': ema.shadow,  # Save EMA shadow
                'model_config': {
                    'image_size': args.img_size,
                    'num_timesteps': args.num_timesteps,
                    'use_lora': args.use_lora,
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'lora_dropout': args.lora_dropout,
                    'model_channels': args.model_channels,
                    'channel_mult': args.channel_mult,
                    'num_res_blocks': args.num_res_blocks,
                }
            }, checkpoint_path)
            ema.restore()
            print(f"  Checkpoint saved: {checkpoint_path}")

            # Generate comparison images
            try:
                comparison_path = os.path.join(args.output_dir, f"comparison_epoch_{epoch}.png")
                # Use EMA for generation
                ema.apply_shadow()
                generate_comparison_diffusion(model, disease_classes, device, real_images_dict, 
                                            num_inference_steps=20, out_path=comparison_path)  # Use fewer steps for speed
                ema.restore()
            except Exception as e:
                print(f"  Warning: Failed to generate comparison images: {e}")
                print("  Continuing training without comparison generation...")

        # Update overall progress
        postfix = {'train_loss': f'{train_loss:.4f}'}
        if epoch % 2 == 0 or epoch == 1:
            postfix['val_loss'] = f'{val_loss:.4f}'
        epochs_pbar.set_postfix(postfix)
        epochs_pbar.update(1)

    epochs_pbar.close()

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "diffusion_final.pt")
    
    # Save with EMA weights
    ema.apply_shadow()
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'disease_classes': disease_classes,
        'num_classes': num_classes,
        'ema_shadow': ema.shadow,  # Save EMA shadow
        'model_config': {
            'image_size': args.img_size,
            'num_timesteps': args.num_timesteps,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'num_res_blocks': args.num_res_blocks,
        }
    }, final_checkpoint_path)
    ema.restore()
    print(f"  Final model saved: {final_checkpoint_path}")

    # Save training summary as JSON
    summary = {
        "epochs": args.epochs,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": best_checkpoint_path,
        "num_classes": num_classes,
        "disease_classes": disease_classes,
        "model_config": {
            'image_size': args.img_size,
            'num_timesteps': args.num_timesteps,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'amp': args.amp,
            'num_workers': args.num_workers,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'num_res_blocks': args.num_res_blocks,
        }
    }
    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  Training summary saved: {summary_path}")

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model")
    parser.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv")
    parser.add_argument("--ham_img_dir", default="dataset/HAM10000_images")
    parser.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv")
    parser.add_argument("--bcn_img_dir", default="dataset/ISIC_images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")  # Reduced from 1e-4
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=500, help="Diffusion timesteps (lower = faster training, higher = better quality)")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/conditional_diffusion", help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="output/conditional_diffusion", help="Output directory for generated samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers (default: 4)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    parser.add_argument("--no_balance", action="store_true", help="Disable class balancing (WeightedRandomSampler)")
    parser.add_argument("--model_channels", type=int, default=128, help="Base channel width for the UNet")
    parser.add_argument("--channel_mult", type=str, default="1,2,4,8", help="Comma separated channel multipliers per downsampling level")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Residual blocks per resolution level")
    parser.add_argument("--max_samples_per_class", type=int, default=2000, help="Maximum samples per class to prevent dominance")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

