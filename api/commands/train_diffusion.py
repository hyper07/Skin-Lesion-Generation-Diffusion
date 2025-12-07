"""
Training script for Diffusion Model on CIFAR-10.
Can be run manually from command line.
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from help_lib.checkpoints import save_checkpoint, load_latest_checkpoint
from models.energy_diffusion_models import DiffusionModel, DiffusionTrainer, train_diffusion_model


def save_generated_images(generated_images, model_name='diffusion', epoch=None, timestamp=None):
    """
    Save generated images to results directory.

    Args:
        generated_images: Tensor of generated images (batch_size, 3, 32, 32)
        model_name: Name of the model (used for directory)
        epoch: Current epoch number (optional)
        timestamp: Timestamp string (optional, auto-generated if None)
    """
    # Images should already be in [0, 1] range from denormalize
    if generated_images.max() > 1.0:
        images = (generated_images * 0.5 + 0.5).clamp(0, 1)
    else:
        images = generated_images.clamp(0, 1)
    
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (B, H, W, C)

    # Create results directory
    results_dir = f'results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    num_images = min(len(images), 16)  # Limit to 16 images for grid

    # Plot images in a grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i]
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()

    # Save grid image
    epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
    filename = f'{results_dir}/generated_samples.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {num_images} generated images to '{filename}'")
    return filename


def save_original_cifar10_images(num_images=16, model_name='diffusion_cifar', save_upscaled=True, upscale_size=128):
    """
    Save original CIFAR-10 images (no normalization) for comparison.

    Args:
        num_images: Number of images to save (grid limited to 16)
        model_name: Results subfolder name
    """
    # Dataset with only ToTensor (keeps [0,1] range)
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=False, num_workers=2)

    images, _ = next(iter(dataloader))  # (B, 3, 32, 32) in [0,1]

    # Convert to (B, H, W, C)
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()

    # Prepare dirs
    results_dir = f'results/{model_name}'
    original_dir = f'{results_dir}/original'
    os.makedirs(original_dir, exist_ok=True)

    # Save grid (limit to 16)
    grid_count = min(len(images_np), 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(grid_count):
        axes[i].imshow(images_np[i])
        axes[i].axis('off')
    plt.tight_layout()
    grid_path = f'{results_dir}/original_samples.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save individual images with exact native resolution (32x32)
    for i in range(grid_count):
        out_path = f'{original_dir}/sample_{i:02d}.png'
        plt.imsave(out_path, images_np[i])

    # Optionally also save upscaled copies for easier visual comparison
    if save_upscaled:
        try:
            from PIL import Image
            up_dir = f'{results_dir}/original_{upscale_size}'
            os.makedirs(up_dir, exist_ok=True)
            for i in range(grid_count):
                img = (images_np[i] * 255).astype('uint8')
                Image.fromarray(img).resize((upscale_size, upscale_size), Image.NEAREST).save(
                    f'{up_dir}/sample_{i:02d}.png'
                )
        except Exception as e:
            print(f"Warning: could not save upscaled originals: {e}")

    print(f"Saved {grid_count} original CIFAR-10 images to '{grid_path}' and '{original_dir}/'.")
    return grid_path


def generate_sample_images(num_images=16, checkpoint_dir='checkpoints', model_name='diffusion_cifar', 
                           diffusion_steps=200):
    """Generate and display sample images from trained Diffusion Model."""

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Initialize model and trainer
    model = DiffusionModel(input_channels=3, img_size=32)
    trainer = DiffusionTrainer(model, device=device)

    # Load trained model from latest_checkpoint.pth or final model file
    from help_lib.checkpoints import get_checkpoint_base_dir, load_latest_checkpoint, get_checkpoint_metadata
    base_dir = get_checkpoint_base_dir()
    
    # Try loading from latest_checkpoint.pth first (preferred)
    checkpoint_subdir = os.path.join(base_dir, 'diffusion_cifar')
    latest_checkpoint = os.path.join(checkpoint_subdir, 'latest_checkpoint.pth')
    final_model_path = os.path.join(checkpoint_subdir, 'diffusion_cifar.pth')
    
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        trainer.model.network.load_state_dict(checkpoint['model_state_dicts']['diffusion_model'])
        # Try to load normalizer from final model file
        normalizer_loaded = False
        if os.path.exists(final_model_path):
            final_checkpoint = torch.load(final_model_path, map_location=device)
            if 'normalizer_mean' in final_checkpoint and 'normalizer_std' in final_checkpoint:
                trainer.model.set_normalizer(final_checkpoint['normalizer_mean'], final_checkpoint['normalizer_std'])
                normalizer_loaded = True
                print(f"Loaded normalizer from: {final_model_path}")
        
        # If normalizer not found, compute it from dataset (this should match training)
        if not normalizer_loaded:
            print("Normalizer not found in checkpoint. Computing from dataset...")
            from models.energy_diffusion_models import get_cifar10_dataloader
            dataloader = get_cifar10_dataloader(batch_size=128, download=False)
            mean = torch.zeros(3)
            std = torch.zeros(3)
            total_samples = 0
            
            for imgs, _ in dataloader:
                batch_size = imgs.size(0)
                imgs_flat = imgs.view(batch_size, 3, -1)
                batch_mean = imgs_flat.mean(dim=(0, 2))
                batch_std = imgs_flat.std(dim=(0, 2))
                mean += batch_mean * batch_size
                std += batch_std * batch_size
                total_samples += batch_size
                if total_samples >= 10000:  # Use first 10k samples for speed
                    break
            
            mean /= total_samples
            std /= total_samples
            mean = mean.reshape(1, 3, 1, 1).to(device)
            std = std.reshape(1, 3, 1, 1).to(device)
            trainer.model.set_normalizer(mean, std)
            print(f"Computed normalizer - Mean: {mean.squeeze()}, Std: {std.squeeze()}")
        
        print(f"Loaded model from checkpoint: {latest_checkpoint}")
    elif os.path.exists(final_model_path):
        checkpoint = torch.load(final_model_path, map_location=device)
        trainer.model.network.load_state_dict(checkpoint['model_state_dict'])
        if 'normalizer_mean' in checkpoint and 'normalizer_std' in checkpoint:
            trainer.model.set_normalizer(checkpoint['normalizer_mean'], checkpoint['normalizer_std'])
            print(f"Loaded normalizer from: {final_model_path}")
        print(f"Loaded model from checkpoint: {final_model_path}")
    else:
        print("No checkpoint found. Using untrained model.")

    # Save a set of original CIFAR-10 images for comparison
    save_original_cifar10_images(num_images=num_images, model_name=model_name)

    # Generate images
    print(f"Generating {num_images} images with {diffusion_steps} diffusion steps...")
    generated_images = trainer.model.generate(num_images, diffusion_steps=diffusion_steps, image_size=32)

    # Save images
    save_generated_images(generated_images, model_name=model_name)

    # Also save individual images
    individual_dir = f'results/{model_name}/individual'
    os.makedirs(individual_dir, exist_ok=True)

    # Images should already be in [0, 1] range
    if generated_images.max() > 1.0:
        images = (generated_images * 0.5 + 0.5).clamp(0, 1)
    else:
        images = generated_images.clamp(0, 1)
    
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    for i in range(num_images):
        img = images[i]
        individual_filename = f'{individual_dir}/sample_{i:02d}.png'
        plt.imsave(individual_filename, img)

    print(f"Individual images saved to '{individual_dir}/'")


def train_diffusion(epochs=50, batch_size=128, learning_rate=0.001, checkpoint_dir='checkpoints', 
                   device=None, diffusion_steps=20):
    """
    Train Diffusion Model on CIFAR-10.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save checkpoints
        device: Device to use (auto-detect if None)
        diffusion_steps: Number of steps for reverse diffusion during generation
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Training Diffusion Model on CIFAR-10")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    print(f"Generation diffusion steps: {diffusion_steps}")

    # Use the train_diffusion_model function
    # Get the base checkpoint directory and construct the save path properly
    from help_lib.checkpoints import get_checkpoint_base_dir
    base_dir = get_checkpoint_base_dir()
    final_model_path = os.path.join(base_dir, 'diffusion_cifar', 'diffusion_cifar.pth')
    
    trainer, losses_history = train_diffusion_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_path=final_model_path,
        checkpoint_dir='diffusion_cifar'  # Relative to sps_genai/checkpoints
    )

    # Generate and save sample images after training
    print("\nGenerating sample images...")
    generate_sample_images(num_images=16, checkpoint_dir=checkpoint_dir, 
                          model_name='diffusion_cifar', diffusion_steps=diffusion_steps)

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Diffusion Model on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default=None, help='Device to use (auto-detect if not specified: mps > cuda > cpu)')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps for generation')
    args = parser.parse_args()

    train_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        diffusion_steps=args.diffusion_steps
    )

