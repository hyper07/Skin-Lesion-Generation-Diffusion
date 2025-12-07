"""
Training script for Conditional Diffusion Model
Generates synthetic skin lesion images conditioned on disease class
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
import warnings
from copy import deepcopy

from model import ConditionalDiffusionModel
from training_config import TrainingConfig
from dataset import SkinLesionDataset
from data_loader import load_data_loaders_from_notebook

# Load environment variables
load_dotenv()

# Set PyTorch CUDA memory management for better allocation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


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


def train_epoch(model, train_loader, optimizer, criterion, config, epoch, scaler=None, ema=None):
    """Train for one epoch with mixed precision support and EMA"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Enable autocast for mixed precision if available (CUDA only, MPS handles it automatically)
    use_amp = config.mixed_precision and scaler is not None and config.device.type == 'cuda'
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels, _) in enumerate(pbar):
        # Move to device
        images = config.move_to_device(images)
        labels = config.move_to_device(labels)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with torch.amp.autocast('cuda'):
                predicted_noise, target_noise, t = model(images, labels)
                loss = criterion(predicted_noise, target_noise)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training (MPS handles mixed precision automatically)
            predicted_noise, target_noise, t = model(images, labels)
            loss = criterion(predicted_noise, target_noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, config):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels, _ in tqdm(val_loader, desc="Validation"):
        images = config.move_to_device(images)
        labels = config.move_to_device(labels)
        
        predicted_noise, target_noise, t = model(images, labels)
        loss = criterion(predicted_noise, target_noise)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def generate_samples(model, disease_classes, config, num_samples=16, save_dir="samples"):
    """Generate sample images for each class"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Get class names
    class_names = {idx: name for name, idx in disease_classes.items()}
    num_classes = len(disease_classes)
    
    # Generate samples for each class
    samples_per_class = max(1, num_samples // num_classes)  # Ensure at least 1 sample per class
    actual_samples = samples_per_class * num_classes
    
    if actual_samples != num_samples:
        print(f"Adjusted samples: {actual_samples} (was {num_samples}) to fit grid layout")
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(15, 5 * num_classes))
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx in range(num_classes):
        class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)
        class_labels = config.move_to_device(class_labels)
        
        # Generate images
        generated = model.sample(class_labels, batch_size=samples_per_class, num_inference_steps=25)
        
        # Denormalize for visualization
        generated = (generated + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        generated = torch.clamp(generated, 0.0, 1.0)
        
        for i in range(samples_per_class):
            if samples_per_class == 1:
                ax = axes[class_idx]
            else:
                ax = axes[class_idx, i] if num_classes > 1 else axes[i]
            img = generated[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(class_names[class_idx], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"generated_samples.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved generated samples to {save_dir}/generated_samples.png")


def train_diffusion_model(
    epochs=100,
    batch_size=32,  # Increased default for faster training
    learning_rate=1e-4,
    image_size=256,
    num_timesteps=1000,
    save_interval=10,
    checkpoint_dir="checkpoints",
    samples_dir="samples",
    device=None,
    no_compile=False,
):
    """Main training function"""
    
    global torch
    
    print("=" * 60)
    print("Conditional Diffusion Model Training")
    print("=" * 60)
    
    # Check device
    if device is None:
        # Auto-detect
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    print(f"✅ Training configured for device: {device}")
    if device.type == 'mps':
        print("🚀 Metal Performance Shaders (MPS) acceleration enabled!")
    
    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset, test_dataset, disease_classes = load_data_loaders_from_notebook(image_size)
    num_classes = len(disease_classes)
    
    # Create training config
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        image_size=image_size,
        num_classes=num_classes
    )
    
    # Override device if specified
    config.device = device
    config.use_mps = device.type == 'mps'
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.dataloader_num_workers > 0 else False,  # Keep workers alive
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=True if config.dataloader_num_workers > 0 else False,
    )
    
    # Create model
    print(f"\nCreating model with {num_classes} classes...")
    model = ConditionalDiffusionModel(
        image_size=image_size,
        num_classes=num_classes,
        model_channels=64,  # Reduced from 128 for memory efficiency
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),  # Smoother channel progression
        num_timesteps=num_timesteps,
        beta_schedule='cosine',  # Cosine schedule is better for images
        time_emb_dim=512,  # Larger time embedding
        class_emb_dim=256,  # Larger class embedding
    )
    
    model = config.move_to_device(model)
    
    # Compile model for faster training if available and not disabled
    if not no_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)
    
    # Create EMA model
    print("Creating EMA model for better sample quality...")
    ema = EMA(model, decay=0.9999)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer)
    scaler = config.get_scaler()  # Get scaler for mixed precision (CUDA only)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Device: {config.device}")
    if config.use_mps:
        print("🚀 Metal Performance Shaders (MPS) acceleration enabled!")
    if scaler is not None:
        print("⚡ Mixed precision training enabled (CUDA)")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {image_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(disease_classes.keys())}")
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_train_loss = float('inf')
    last_val_loss = None  # Track last validation loss for final checkpoint
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch, scaler, ema)
        
        # Validate less frequently to save time (every 2 epochs or first epoch)
        if epoch % 2 == 0 or epoch == 1:
            val_loss = validate(model, val_loader, criterion, config)
            last_val_loss = val_loss  # Update last validation loss
            # Update scheduler only when we validate
            scheduler.step(val_loss)
        else:
            val_loss = None
        
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            print(f"  New best model! (val_loss: {val_loss:.4f})")
        
        # Generate samples less frequently to save time (every 20 epochs instead of 10)
        if epoch % (save_interval * 2) == 0:
            print("  Generating samples with EMA model...")
            # Use EMA model for better sample quality
            ema.apply_shadow()
            generate_samples(model, disease_classes, config, num_samples=9, save_dir=samples_dir)
            ema.restore()
            
            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final checkpoint with metadata
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    # Apply EMA for final model
    ema.apply_shadow()
    
    # Save final checkpoint with all metadata
    final_checkpoint = {
        'epoch': epochs,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': last_val_loss if last_val_loss is not None else best_val_loss,
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'disease_classes': disease_classes,
        'num_classes': num_classes,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'image_size': image_size,
            'num_timesteps': num_timesteps,
            'device': str(config.device),
            'use_mps': config.use_mps,
        },
        'model_config': {
            'image_size': image_size,
            'num_classes': num_classes,
            'model_channels': 64,
            'num_res_blocks': 2,
            'channel_mult': (1, 2, 3, 4),
            'num_timesteps': num_timesteps,
            'beta_schedule': 'cosine',
            'time_emb_dim': 512,
            'class_emb_dim': 256,
        }
    }
    
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print(f"Checkpoint includes:")
    print(f"  - Model weights (with EMA)")
    print(f"  - EMA shadow parameters")
    print(f"  - Optimizer state")
    print(f"  - Scheduler state")
    print(f"  - Training metrics (best val loss: {best_val_loss:.4f})")
    print(f"  - Disease classes mapping: {disease_classes}")
    print(f"  - Training configuration")
    print(f"  - Model configuration")
    print(f"Generated samples saved in: {samples_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4 for memory efficiency)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--samples-dir", type=str, default="samples", help="Samples directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' (default: auto-detect)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (if available)")
    
    args = parser.parse_args()
    
    train_diffusion_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_timesteps=args.num_timesteps,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        samples_dir=args.samples_dir,
        device=args.device,
        no_compile=args.no_compile,
    )

