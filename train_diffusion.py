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

from models.conditional_diffusion import ConditionalDiffusionModel
from training_config import TrainingConfig
from dataset import SkinLesionDataset

# Load environment variables
load_dotenv()


def load_data_loaders_from_notebook():
    """
    Load data loaders using the same logic as the notebook
    This replicates the data loading from 'Train Test Val (Top 3).ipynb'
    """
    import pandas as pd
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    
    DATA_DIR = os.getenv('DATA_DIR')
    if DATA_DIR is None:
        raise ValueError("DATA_DIR not found in .env file. Please create .env with DATA_DIR=<path>")
    
    # HAM10000 diagnosis code mapping
    ham_dx_map = {
        'nv': 'Nevus',
        'mel': 'Melanoma (HAM)',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratosis',
        'vasc': 'Vascular lesion',
        'df': 'Dermatofibroma',
        'bkl': 'Benign keratosis (HAM)'
    }
    
    # ISIC diagnosis mapping
    isic_diagnosis_map = {
        'Nevus': 'Nevus',
        'Melanoma, NOS': 'Melanoma (BCN)',
        'Melanoma metastasis': 'Melanoma metastasis',
        'Basal cell carcinoma': 'Basal cell carcinoma',
        'Seborrheic keratosis': 'Seborrheic keratosis',
        'Solar or actinic keratosis': 'Actinic keratosis',
        'Squamous cell carcinoma, NOS': 'Squamous cell carcinoma',
        'Scar': 'Scar',
        'Solar lentigo': 'Solar lentigo',
        'Dermatofibroma': 'Dermatofibroma'
    }
    
    # Load HAM10000
    ham_metadata_path = os.path.join(DATA_DIR, "dataverse_files/HAM10000_metadata.csv")
    ham_images_dir = os.path.join(DATA_DIR, "dataverse_files/HAM10000_images")
    
    ham_df = pd.read_csv(ham_metadata_path)
    ham_paths = []
    ham_labels = []
    
    for _, row in ham_df.iterrows():
        image_id = row.get('image_id', row.get('lesion_id', row.iloc[0]))
        dx_code = row.get('dx', row.get('diagnosis', row.iloc[1]))
        diagnosis = ham_dx_map.get(dx_code, dx_code)
        
        img_path = os.path.join(ham_images_dir, f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            ham_paths.append(img_path)
            ham_labels.append(diagnosis)
    
    # Load ISIC
    isic_metadata_path = os.path.join(DATA_DIR, "dataverse_files/ISIC_metadata.csv")
    isic_images = os.path.join(DATA_DIR, "dataverse_files/ISIC_images")
    
    isic_df = pd.read_csv(isic_metadata_path)
    isic_paths = []
    isic_labels = []
    
    for _, row in isic_df.iterrows():
        image_id = row['isic_id']
        raw_diagnosis = row['diagnosis_3']
        
        if pd.isna(raw_diagnosis):
            continue
        
        diagnosis = isic_diagnosis_map.get(raw_diagnosis, raw_diagnosis)
        img_path = os.path.join(isic_images, f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            isic_paths.append(img_path)
            isic_labels.append(diagnosis)
    
    # Combine
    all_paths = ham_paths + isic_paths
    all_labels = ham_labels + isic_labels
    
    # Stratified sampling: 400 samples max for Nevus, all samples for other classes
    nevus_limit = 400
    sampled_paths = []
    sampled_labels = []
    
    # Group by class
    from collections import defaultdict
    class_data = defaultdict(list)
    for path, label in zip(all_paths, all_labels):
        class_data[label].append(path)
    
    # Sample from each class
    for class_name, paths in class_data.items():
        if class_name == 'Nevus':
            # Limit Nevus to 400 samples
            import random
            random.seed(42)
            sampled_class_paths = random.sample(paths, min(nevus_limit, len(paths)))
        else:
            # Use all samples for other classes
            sampled_class_paths = paths
        
        sampled_paths.extend(sampled_class_paths)
        sampled_labels.extend([class_name] * len(sampled_class_paths))
    
    total_samples = len(sampled_paths)
    print(f"Stratified sampling completed:")
    print(f"  Nevus: limited to {nevus_limit} samples")
    print(f"  Other classes: all available samples")
    print(f"  Total samples: {total_samples}")
    
    # Create class mapping from all unique diseases in the sample
    unique_diseases = sorted(list(set(sampled_labels)))
    disease_classes = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    print(f"Classes in sample: {unique_diseases}")
    print(f"Class distribution in sample:")
    label_counts_sample = pd.Series(sampled_labels).value_counts()
    for class_name, count in label_counts_sample.items():
        print(f"  {class_name}: {count}")
    
    # Use sampled data
    filtered_paths = sampled_paths
    filtered_labels = sampled_labels
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filtered_paths, filtered_labels,
        test_size=0.3,
        stratify=filtered_labels,
        random_state=42
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Transforms (256x256 images, normalized to [-1, 1])
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_paths, train_labels, disease_classes, train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, disease_classes, val_test_transform)
    test_dataset = SkinLesionDataset(test_paths, test_labels, disease_classes, val_test_transform)
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Classes: {disease_classes}")
    
    return train_dataset, val_dataset, test_dataset, disease_classes


def train_epoch(model, train_loader, optimizer, criterion, config, epoch, scaler=None):
    """Train for one epoch with mixed precision support"""
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
            with torch.cuda.amp.autocast():
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
        generated = model.sample(class_labels, batch_size=samples_per_class, num_inference_steps=50)
        
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
    compile_model=True,  # Enable model compilation by default
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
        # MPS does not support torch.compile with inductor backend, so disable compilation
        compile_model = False
        print("ℹ️  Model compilation disabled for MPS device")
    
    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset, test_dataset, disease_classes = load_data_loaders_from_notebook()
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
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        num_timesteps=num_timesteps,
        beta_schedule='linear',
    )
    
    model = config.move_to_device(model)
    
    # Compile model for faster training (PyTorch 2.0+)
    # Use 'default' mode for better compatibility across different GPUs
    # 'reduce-overhead' requires more GPU compute units and may not work on all GPUs
    if compile_model:
        try:
            # Suppress the SM warning by using default mode
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*max_autotune_gemm.*")
                warnings.filterwarnings("ignore", message=".*Not enough SMs.*")
                # Suppress dynamo errors for MPS device since inductor doesn't support MPS
                if device.type == 'mps':
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                model = torch.compile(model, mode='default')
            print("✅ Model compiled with torch.compile for faster training")
        except Exception as e:
            print(f"⚠️  Could not compile model: {e}. Continuing without compilation.")
    else:
        print("ℹ️  Model compilation disabled")
    
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
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(disease_classes.keys())}")
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_train_loss = float('inf')
    last_val_loss = None  # Track last validation loss for final checkpoint
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch, scaler)
        
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
            print("  Generating samples...")
            generate_samples(model, disease_classes, config, num_samples=9, save_dir=samples_dir)
    
    # Save final checkpoint with metadata
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    # Save final checkpoint with all metadata
    final_checkpoint = {
        'epoch': epochs,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
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
            'model_channels': 128,
            'num_res_blocks': 2,
            'channel_mult': (1, 2, 4, 8),
            'num_timesteps': num_timesteps,
            'beta_schedule': 'linear',
        }
    }
    
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print(f"Checkpoint includes:")
    print(f"  - Model weights")
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32 for faster training)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--samples-dir", type=str, default="samples", help="Samples directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' (default: auto-detect)")
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation (use if you get SM warnings)")
    
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
        compile_model=not args.no_compile,
    )

