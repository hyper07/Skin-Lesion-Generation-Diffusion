"""Training script for Probability Diffusion Model (Stable Diffusion based)."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path
import json

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import argparse

# Add project root to path and import project utilities
sys.path.append(str(Path(__file__).resolve().parents[1]))

from device_utils import get_device
from data.dataset_utils import ensure_dataset
from data.data_loader import create_data_loaders
from models.prebuilt_diffusion import DiffusionModel

try:
    from torch.amp import GradScaler as _GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler


def _create_grad_scaler(device, enabled):
    if device.type != 'cuda' or not enabled:
        return _GradScaler(enabled=False)
    try:
        return _GradScaler(device_type='cuda', enabled=True)
    except TypeError:
        return _GradScaler(enabled=True)

@torch.no_grad()
def generate_comparison_probability(model, disease_classes, device, real_images_dict, num_inference_steps=50, img_size=128, out_path="output/comparison.png"):
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
        latent_size = img_size // 8
        latents = torch.randn((1, 4, latent_size, latent_size), device=device)
        model.scheduler.set_timesteps(num_inference_steps)
        for t in model.scheduler.timesteps:
            timestep = t.expand(1).to(device)
            noise_pred = model.unet(latents, timestep, class_labels)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents / 0.18215
        images = model.vae.decode(latents).sample
        images = (images.clamp(-1, 1) + 1) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        imgs.append(images[0].cpu())
    
    grid = make_grid(torch.cat(imgs, 0), nrow=2, normalize=True, value_range=(0, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None and scaler.is_enabled()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", 
                unit='batch', ncols=100)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            # Forward pass - Probability Diffusion returns (predicted_noise, true_noise)
            predicted_noise, true_noise = model(images, labels)
            loss = criterion(predicted_noise, true_noise)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
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
    
    pbar = tqdm(val_loader, desc="Validation", unit='batch', ncols=100)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        if device.type == 'cuda':
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=torch.float16)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            predicted_noise, true_noise = model(images, labels)
            loss = criterion(predicted_noise, true_noise)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'val_loss': f'{loss.item():.4f}',
            'avg_val_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


@torch.no_grad()
def generate_sample_images(model, disease_classes, device, epoch, num_samples=4, img_size=128, output_dir="output/prebuilt_diffusion"):
    """Generate sample images during training"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    num_classes = len(disease_classes)
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    # Generate samples for each class
    for class_idx in range(min(num_classes, 3)):  # Limit to first 3 classes for speed
        class_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        
        # Latent size (Stable Diffusion uses 8x downsampling)
        latent_size = img_size // 8
        
        # Start from random latents
        latents = torch.randn((num_samples, 4, latent_size, latent_size), device=device)
        
        # Set scheduler timesteps
        model.scheduler.set_timesteps(50)  # Use 50 inference steps for quick samples
        
        # Denoising loop
        for t in model.scheduler.timesteps:
            timestep = t.expand(num_samples).to(device)
            noise_pred = model.unet(latents, timestep, class_labels)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to images
        latents = latents / 0.18215  # Scale back
        images = model.vae.decode(latents).sample
        
        # Normalize to [0, 1]
        images = (images.clamp(-1, 1) + 1) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # Save samples
        class_name_safe = class_names[class_idx].replace(" ", "_").replace("(", "").replace(")", "")
        sample_path = os.path.join(output_dir, f"sample_epoch_{epoch}_class_{class_name_safe}.png")
        save_image(images, sample_path, nrow=num_samples, normalize=False)
        print(f"  Generated samples for class '{class_names[class_idx]}' -> {sample_path}")


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
    )

    num_classes = len(disease_classes)
    print(f"Detected {num_classes} classes: {list(disease_classes.keys())}")

    # Create model
    print(f"\nCreating Probability Diffusion Model...")
    print(f"  Using Stable Diffusion v1.4 base model")
    print(f"  LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    print("  (This can take a while on CPU/MPS...)")
    
    model = DiffusionModel(
        num_classes=num_classes,
        use_lora=args.use_lora
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    
    # Only optimize UNet parameters (VAE is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = _create_grad_scaler(device, args.amp)
    
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect one real image per class for comparison (optional)
    real_images_dict = {}
    try:
        print("\nCollecting one real image per class for comparisons...")
        for images, labels, _ in train_loader:
            for img, label in zip(images, labels):
                if label.item() not in real_images_dict:
                    real_images_dict[label.item()] = img
            if len(real_images_dict) == num_classes:
                break
        if len(real_images_dict) < num_classes:
            print(f"  Warning: only collected {len(real_images_dict)}/{num_classes} classes for comparison images.")
    except Exception as e:
        print(f"  Warning: failed to collect real images for comparison: {e}")
        real_images_dict = None

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs, scaler=scaler)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, amp_enabled=scaler.is_enabled())
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"diffusion_epoch_{epoch}.pt")
            torch.save({
                'unet_state_dict': model.unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'disease_classes': disease_classes,
                'model_config': {
                    'num_classes': num_classes,
                    'use_lora': args.use_lora,
                    'img_size': args.img_size,
                }
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "diffusion_best.pt")
            torch.save({
                'unet_state_dict': model.unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'disease_classes': disease_classes,
                'model_config': {
                    'num_classes': num_classes,
                    'use_lora': args.use_lora,
                    'img_size': args.img_size,
                }
            }, best_checkpoint_path)
            print(f"  New best model! (val_loss: {val_loss:.4f})")
        
        # Generate sample images periodically (can be heavy on CPU/MPS)
        if args.enable_sampling and epoch % args.sample_interval == 0:
            try:
                print(f"\n  Generating sample images (this may be slow)...")
                generate_sample_images(model, disease_classes, device, epoch,
                                     num_samples=4, img_size=args.img_size, 
                                     output_dir=args.output_dir)
            except Exception as e:
                print(f"  Warning: failed to generate sample images: {e}")

        # Generate comparison images every epoch (optional)
        if args.enable_sampling and real_images_dict is not None:
            try:
                comparison_path = os.path.join(args.output_dir, f"comparison_epoch_{epoch}.png")
                generate_comparison_probability(model, disease_classes, device, real_images_dict,
                                              num_inference_steps=20, img_size=args.img_size, 
                                              out_path=comparison_path)
            except Exception as e:
                print(f"  Warning: failed to generate comparison images: {e}")

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "diffusion_final.pt")
    torch.save({
        'unet_state_dict': model.unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'epoch': args.epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'num_classes': num_classes,
        'disease_classes': disease_classes,
        'model_config': {
            'num_classes': num_classes,
            'use_lora': args.use_lora,
            'img_size': args.img_size,
            'amp': args.amp,
            'num_workers': args.num_workers,
        }
    }, final_checkpoint_path)
    print(f"  Final model saved: {final_checkpoint_path}")

    # Save training summary as JSON
    summary = {
        "epochs": args.epochs,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "num_classes": num_classes,
        "disease_classes": disease_classes,
        "model_config": {
            'num_classes': num_classes,
            'use_lora': args.use_lora,
            'img_size': args.img_size,
            'amp': args.amp,
            'num_workers': args.num_workers,
        }
    }
    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  Training summary saved: {summary_path}")

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Probability Diffusion Model")
    parser.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv")
    parser.add_argument("--ham_img_dir", default="dataset/HAM10000_images")
    parser.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv")
    parser.add_argument("--bcn_img_dir", default="dataset/ISIC_images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (smaller for Stable Diffusion)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--img_size", type=int, default=64, help="Image size (must be multiple of 8 for Stable Diffusion)")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=20, help="Generate sample images every N epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="output/prebuilt_diffusion", help="Output directory for generated samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA for efficient fine-tuning (LoRA config is hardcoded in model)")
    parser.add_argument("--enable_sampling", action="store_true", help="Enable Stable Diffusion image sampling during training (can be slow)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers (default: auto)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    
    args = parser.parse_args()
    
    # Validate image size
    if args.img_size % 8 != 0:
        print(f"Warning: Image size {args.img_size} is not a multiple of 8.")
        print("Stable Diffusion requires image size to be a multiple of 8.")
        print(f"Adjusting to {(args.img_size // 8) * 8}")
        args.img_size = (args.img_size // 8) * 8
    
    train(args)


if __name__ == "__main__":
    main()

