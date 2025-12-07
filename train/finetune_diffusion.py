"""
Fine-tuning script for Conditional Diffusion Model
Resumes training from an existing checkpoint.
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
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.conditional_diffusion import ConditionalDiffusionModel
from data.data_loader import create_data_loaders
from data.dataset_utils import ensure_dataset
from device_utils import get_device
from train.train_diffusion import EMA, train_epoch, validate, generate_comparison_diffusion, _create_grad_scaler


def finetune(args):
    device = get_device()
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Load Data
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

    # 2. Create Model Structure (Must match checkpoint)
    print(f"\nInitializing model structure...")
    # Note: We assume the config matches the checkpoint. 
    # Ideally, we should load config from checkpoint first, but for now we use args.
    try:
        channel_mult = tuple(int(x.strip()) for x in args.channel_mult.split(','))
    except ValueError:
        print(f"Invalid channel_mult value. Using default 1,2,4,8")
        channel_mult = (1, 2, 4, 8)

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

    # 3. Load Checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}...")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)
        
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  Model weights loaded.")

    # 4. Setup Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = _create_grad_scaler(device, args.amp)
    criterion = nn.MSELoss()

    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  Optimizer state loaded.")
    
    if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("  Scaler state loaded.")

    # 5. Setup EMA
    ema = EMA(model, decay=0.999)  # Reduced from 0.9999 for stability
    if 'ema_shadow' in checkpoint:
        # Load EMA shadow weights
        ema.shadow = checkpoint['ema_shadow']
        print("  EMA shadow loaded from checkpoint.")
    else:
        print("  No EMA shadow in checkpoint. Starting EMA from current weights.")

    # 6. Training Loop
    start_epoch = checkpoint.get('epoch', 0) + 1
    total_epochs = start_epoch + args.epochs - 1
    
    print(f"\nResuming training from epoch {start_epoch} to {total_epochs}...")
    
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    if best_val_loss == float('inf') and 'val_loss' in checkpoint:
         best_val_loss = checkpoint['val_loss']

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    real_images_dict = {}
    # Collect real images for comparison
    for images, labels, _ in train_loader:
        for img, label in zip(images, labels):
            if label.item() not in real_images_dict:
                real_images_dict[label.item()] = img
        if len(real_images_dict) == num_classes:
            break

    for epoch in range(start_epoch, total_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs, scaler=scaler, ema=ema)
        
        # Validate
        if epoch % 2 == 0 or epoch == 1:
            ema.apply_shadow()
            val_loss = validate(model, val_loader, criterion, device, amp_enabled=scaler.is_enabled())
            ema.restore()
            
            print(f"\nEpoch {epoch}/{total_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} (EMA)")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best model! (val_loss: {val_loss:.4f})")
                best_path = os.path.join(args.checkpoint_dir, "diffusion_finetuned_best.pt")
                
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_loss,
                    'disease_classes': disease_classes,
                    'num_classes': num_classes,
                    'ema_shadow': ema.shadow,  # Save EMA shadow
                    'model_config': checkpoint.get('model_config', {})
                }, best_path)
                ema.restore()
                print(f"  Saved: {best_path}")
        else:
            print(f"\nEpoch {epoch}/{total_epochs}: Train Loss: {train_loss:.4f}")

        # Save Checkpoint
        if epoch % args.save_interval == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"diffusion_finetuned_epoch_{epoch}.pt")
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'disease_classes': disease_classes,
                'num_classes': num_classes,
                'ema_shadow': ema.shadow,  # Save EMA shadow
                'model_config': checkpoint.get('model_config', {})
            }, ckpt_path)
            ema.restore()
            print(f"  Checkpoint saved: {ckpt_path}")
            
            # Generate samples
            try:
                comp_path = os.path.join(args.output_dir, f"finetune_epoch_{epoch}.png")
                ema.apply_shadow()
                generate_comparison_diffusion(model, disease_classes, device, real_images_dict, 
                                            num_inference_steps=20, out_path=comp_path)
                ema.restore()
            except Exception as e:
                print(f"Generation failed: {e}")

    print("\nFine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Conditional Diffusion Model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to existing checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50, help="Number of ADDITIONAL epochs to train")
    
    # Dataset args
    parser.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv")
    parser.add_argument("--ham_img_dir", default="dataset/HAM10000_images")
    parser.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv")
    parser.add_argument("--bcn_img_dir", default="dataset/ISIC_images")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower learning rate for fine-tuning")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/conditional_diffusion_finetune")
    parser.add_argument("--output_dir", type=str, default="output/conditional_diffusion_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--no_balance", action="store_true")
    
    # Model args (Must match original training)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--channel_mult", type=str, default="1,2,4,8")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--max_samples_per_class", type=int, default=2000, help="Maximum samples per class to prevent dominance")

    args = parser.parse_args()
    finetune(args)

if __name__ == "__main__":
    main()
