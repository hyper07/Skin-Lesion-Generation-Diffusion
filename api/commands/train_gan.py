import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
# from models.gan_models import GAN
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from help_lib.checkpoints import save_checkpoint, load_latest_checkpoint
from models.gan_models import GAN, train_gan as train_gan_func


def train_gan(dataset='mnist', epochs=10, batch_size=64, learning_rate=0.0002, latent_dim=100, checkpoint_dir='checkpoints', device=None, save_images_every=10):
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    device = torch.device(device)

    # Use the train_gan function from gan_models
    gan = train_gan_func(dataset=dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim, device=device, save_path=f"{checkpoint_dir}/gan_{dataset}/gan_{dataset}.pth", checkpoint_dir=checkpoint_dir)

    # Generate and save sample images after training
    print("Generating sample images...")
    generate_sample_images(num_images=16, checkpoint_dir=f"{checkpoint_dir}/gan_{dataset}", model_name=f"gan_{dataset}", dataset=dataset)

    # Additional logic for saving images during training can be added here if needed
    # For now, just return the trained GAN
    return gan

def save_generated_images(generated_images, model_name='gan', epoch=None, timestamp=None):
    """
    Save generated images to results directory.

    Args:
        generated_images: Tensor of generated images (batch_size, 1, 28, 28)
        model_name: Name of the model (used for directory)
        epoch: Current epoch number (optional)
        timestamp: Timestamp string (optional, auto-generated if None)
    """

    # Denormalize from [-1, 1] to [0, 1]
    images = (generated_images + 1) / 2
    images = torch.clamp(images, 0, 1)

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
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()

    # Save grid image
    epoch_str = f'_{epoch}' if epoch is not None else ''
    filename = f'{results_dir}/generated_samples{epoch_str}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {num_images} generated images to '{filename}'")
    return filename

def generate_sample_images(num_images=16, checkpoint_dir='checkpoints', model_name='gan', dataset='mnist'):
    """Generate and display sample images from trained GAN."""

    # Initialize GAN
    gan = GAN(dataset=dataset)

    # Load trained model
    models = {'generator': gan.generator, 'discriminator': gan.discriminator}
    optimizers = {'generator': gan.g_optimizer, 'discriminator': gan.d_optimizer}
    epoch, losses, accuracies = load_latest_checkpoint(models, optimizers, checkpoint_dir, gan.device)

    if epoch > 0:
        print(f"Loaded model from epoch {epoch}")
    else:
        print("No checkpoint found. Using untrained model.")

    # Generate images
    gan.generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, gan.latent_dim, device=gan.device)
        generated_images = gan.generator(noise).cpu()

    # Use the utility function to save images
    save_generated_images(generated_images, model_name=model_name, epoch=epoch)

    # Also save individual images
    individual_dir = f'results/{model_name}/individual'
    os.makedirs(individual_dir, exist_ok=True)

    # Denormalize images
    images = (generated_images + 1) / 2
    images = torch.clamp(images, 0, 1)

    for i in range(num_images):
        fig, ax = plt.subplots(figsize=(2, 2))
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        individual_filename = f'{individual_dir}/sample_{i:02d}.png'
        plt.savefig(individual_filename, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Individual images saved to '{individual_dir}/'")

def save_generated_images(generated_images, model_name='gan', epoch=None, timestamp=None):
    """
    Save generated images to results directory.

    Args:
        generated_images: Tensor of generated images (batch_size, 1, 28, 28)
        model_name: Name of the model (used for directory)
        epoch: Current epoch number (optional)
        timestamp: Timestamp string (optional, auto-generated if None)
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Denormalize from [-1, 1] to [0, 1]
    images = (generated_images + 1) / 2
    images = torch.clamp(images, 0, 1)

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
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()

    # Save grid image
    epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
    filename = f'{results_dir}/generated_samples.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {num_images} generated images to '{filename}'")
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on specified dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'], help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default=None, help='Device to use (auto-detect if not specified: mps > cuda > cpu)')
    parser.add_argument('--save_images_every', type=int, default=10, help='Save generated images every N epochs')
    args = parser.parse_args()

    train_gan(args.dataset, args.epochs, args.batch_size, args.lr, args.latent_dim, args.checkpoint_dir, args.device, args.save_images_every)