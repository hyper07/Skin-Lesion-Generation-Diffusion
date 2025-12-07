"""
Training script for Energy Model on CIFAR-10.
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
from models.energy_diffusion_models import EnergyModel, EnergyModelTrainer, train_energy_model


def save_generated_images(generated_images, model_name='energy', epoch=None, timestamp=None):
    """
    Save generated images to results directory.

    Args:
        generated_images: Tensor of generated images (batch_size, 3, 32, 32)
        model_name: Name of the model (used for directory)
        epoch: Current epoch number (optional)
        timestamp: Timestamp string (optional, auto-generated if None)
    """
    # Denormalize from [-1, 1] to [0, 1]
    images = (generated_images * 0.5 + 0.5).clamp(0, 1)
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


def save_original_cifar10_images(num_images=16, model_name='energy_cifar', save_upscaled=True, upscale_size=128):
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


def generate_sample_images(num_images=16, checkpoint_dir='checkpoints', model_name='energy_cifar'):
    """Generate and display sample images from trained Energy Model."""

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Initialize model and trainer
    model = EnergyModel(input_channels=3, img_size=32)
    trainer = EnergyModelTrainer(model, device=device)

    # Load trained model from latest_checkpoint.pth or final model file
    from help_lib.checkpoints import get_checkpoint_base_dir, load_latest_checkpoint
    base_dir = get_checkpoint_base_dir()
    
    # Try loading from latest_checkpoint.pth first (preferred)
    checkpoint_subdir = os.path.join(base_dir, 'energy_cifar')
    latest_checkpoint = os.path.join(checkpoint_subdir, 'latest_checkpoint.pth')
    final_model_path = os.path.join(checkpoint_subdir, 'energy_cifar.pth')
    
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dicts']['energy_model'])
        print(f"Loaded model from checkpoint: {latest_checkpoint}")
    elif os.path.exists(final_model_path):
        checkpoint = torch.load(final_model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {final_model_path}")
    else:
        print("No checkpoint found. Using untrained model.")

    # Save a set of original CIFAR-10 images for comparison
    save_original_cifar10_images(num_images=num_images, model_name=model_name)

    # Generate images (more conservative defaults for stability)
    generated_images = trainer.generate_samples(num_images, num_steps=512, step_size=1.0, noise_std=0.005)

    # Save images
    save_generated_images(generated_images, model_name=model_name)

    # Also save individual images
    individual_dir = f'results/{model_name}/individual'
    os.makedirs(individual_dir, exist_ok=True)

    # Denormalize images
    images = (generated_images * 0.5 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    for i in range(num_images):
        fig, ax = plt.subplots(figsize=(2, 2))
        img = images[i]
        ax.imshow(img)
        ax.axis('off')
        individual_filename = f'{individual_dir}/sample_{i:02d}.png'
        plt.savefig(individual_filename, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Individual images saved to '{individual_dir}/'")


def train_energy(epochs=50, batch_size=128, learning_rate=0.0001, checkpoint_dir='checkpoints', 
                device=None, alpha=0.1, steps=60, step_size=10.0, noise=0.005):
    """
    Train Energy Model on CIFAR-10.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save checkpoints
        device: Device to use (auto-detect if None)
        alpha: Regularization parameter for contrastive divergence
        steps: Number of Langevin dynamics steps
        step_size: Step size for Langevin dynamics
        noise: Noise level for Langevin dynamics
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Training Energy Model on CIFAR-10")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    print(f"Training parameters: alpha={alpha}, steps={steps}, step_size={step_size}, noise={noise}")

    # Use the train_energy_model function
    # Get the base checkpoint directory and construct the save path properly
    from help_lib.checkpoints import get_checkpoint_base_dir
    base_dir = get_checkpoint_base_dir()
    final_model_path = os.path.join(base_dir, 'energy_cifar', 'energy_cifar.pth')
    
    trainer, losses_history = train_energy_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_path=final_model_path,
        checkpoint_dir='energy_cifar',  # Relative to sps_genai/checkpoints
        alpha=alpha,
        steps=steps,
        step_size=step_size,
        noise=noise
    )

    # Generate and save sample images after training
    print("\nGenerating sample images...")
    generate_sample_images(num_images=16, checkpoint_dir=checkpoint_dir, model_name='energy_cifar')

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Energy Model on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default=None, help='Device to use (auto-detect if not specified: mps > cuda > cpu)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Regularization parameter for contrastive divergence')
    parser.add_argument('--steps', type=int, default=60, help='Number of Langevin dynamics steps')
    parser.add_argument('--step_size', type=float, default=10.0, help='Step size for Langevin dynamics')
    parser.add_argument('--noise', type=float, default=0.005, help='Noise level for Langevin dynamics')
    args = parser.parse_args()

    train_energy(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        alpha=args.alpha,
        steps=args.steps,
        step_size=args.step_size,
        noise=args.noise
    )

