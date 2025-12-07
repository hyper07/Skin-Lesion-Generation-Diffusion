"""
GAN Model Definitions
Generative Adversarial Network architectures for image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import os
import base64
from io import BytesIO
from PIL import Image
from typing import List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import checkpoint utilities
from help_lib.checkpoints import save_checkpoint


class BaseGAN(nn.Module):
    """Base GAN class with common functionality."""

    def __init__(self, latent_dim: int = 100):
        super(BaseGAN, self).__init__()
        self.latent_dim = latent_dim

    def get_device(self):
        """Get device of the model."""
        return next(self.parameters()).device


class Generator(BaseGAN):
    """
    Generator network for GAN.
    Supports different image sizes and channels.
    """

    def __init__(self, latent_dim: int = 100, output_channels: int = 1, img_size: int = 28):
        super(Generator, self).__init__(latent_dim)
        self.output_channels = output_channels
        self.img_size = img_size
        self.spatial = img_size // 4  # 7 for 28, 8 for 32

        # Fully connected layer: latent_dim -> spatial*spatial*128
        self.fc = nn.Linear(latent_dim, self.spatial * self.spatial * 128)

        # First ConvTranspose2D: 128 -> 64, spatial -> spatial*2
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Second ConvTranspose2D: 64 -> output_channels, spatial*2 -> spatial*4 = img_size
        self.conv_trans2 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, latent_dim)
        x = self.fc(x)  # (batch_size, spatial*spatial*128)
        x = x.view(x.size(0), 128, self.spatial, self.spatial)  # Reshape to (batch_size, 128, spatial, spatial)

        # First conv transpose block
        x = self.conv_trans1(x)  # (batch_size, 64, spatial*2, spatial*2)
        x = self.bn1(x)
        x = self.relu(x)

        # Second conv transpose block
        x = self.conv_trans2(x)  # (batch_size, output_channels, img_size, img_size)
        x = self.tanh(x)

        return x


class Discriminator(BaseGAN):
    """
    Discriminator network for GAN.
    Supports different image sizes and channels.
    """

    def __init__(self, input_channels: int = 1, img_size: int = 28):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.spatial = img_size // 4

        # First Conv2D: input_channels -> 64, img_size -> img_size//2
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Second Conv2D: 64 -> 128, img_size//2 -> img_size//4
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(128)

        # Flatten and Linear layer to single output
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * self.spatial * self.spatial, 1)

    def forward(self, x):
        # x: (batch_size, input_channels, img_size, img_size)

        # First conv block
        x = self.conv1(x)  # (batch_size, 64, img_size//2, img_size//2)
        x = self.leaky_relu(x)

        # Second conv block
        x = self.conv2(x)  # (batch_size, 128, img_size//4, img_size//4)
        x = self.bn(x)
        x = self.leaky_relu(x)

        # Flatten and linear
        x = self.flatten(x)  # (batch_size, 128*spatial*spatial)
        x = self.fc(x)  # (batch_size, 1)

        return x


class GAN(BaseGAN):
    """
    Complete GAN implementation for image generation.
    Supports different datasets like MNIST and CIFAR-10.
    """

    def __init__(self, latent_dim: int = 100, dataset: str = 'mnist', device: Optional[str] = None, learning_rate: float = 0.0002):
        super(GAN, self).__init__(latent_dim)
        self.dataset = dataset.lower()
        if self.dataset == 'mnist':
            self.channels = 1
            self.img_size = 28
        elif self.dataset == 'cifar':
            self.channels = 3
            self.img_size = 32
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Auto-detect best available device: MPS (Mac) > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        # Initialize networks
        self.generator = Generator(latent_dim, self.channels, self.img_size).to(self.device)
        self.discriminator = Discriminator(self.channels, self.img_size).to(self.device)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def _get_real_labels(self, batch_size: int) -> torch.Tensor:
        """Generate real labels (1) with slight noise for stability."""
        return torch.ones(batch_size, 1, device=self.device) * 0.9

    def _get_fake_labels(self, batch_size: int) -> torch.Tensor:
        """Generate fake labels (0) with slight noise for stability."""
        return torch.zeros(batch_size, 1, device=self.device) + 0.1

    def train_discriminator(self, real_images: torch.Tensor) -> float:
        """Train discriminator on one batch."""
        batch_size = real_images.size(0)
        self.d_optimizer.zero_grad()

        # Train on real images
        real_output = self.discriminator(real_images)
        real_loss = self.criterion(real_output, self._get_real_labels(batch_size))

        # Train on fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(fake_output, self._get_fake_labels(batch_size))

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, batch_size: int) -> float:
        """Train generator on one batch."""
        self.g_optimizer.zero_grad()

        # Generate fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)

        # Get discriminator output for fake images
        fake_output = self.discriminator(fake_images)

        # Generator wants discriminator to think fake images are real
        g_loss = self.criterion(fake_output, self._get_real_labels(batch_size))
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def train_epoch(self, dataloader, epoch=None, total_epochs=None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        d_losses = []
        g_losses = []

        # Use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{total_epochs}" if epoch is not None else "Training",
                ncols=0,
                dynamic_ncols=True,
                leave=False,
                position=0
            )
        except ImportError:
            iterator = dataloader

        for batch_idx, (real_images, _) in enumerate(iterator):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            # Train discriminator
            d_loss = self.train_discriminator(real_images)
            d_losses.append(d_loss)

            # Train generator
            g_loss = self.train_generator(batch_size)
            g_losses.append(g_loss)

            # Update progress bar
            if hasattr(iterator, 'set_postfix'):
                avg_d_loss = sum(d_losses) / len(d_losses)
                avg_g_loss = sum(g_losses) / len(g_losses)
                iterator.set_postfix({
                    "D Loss": f"{avg_d_loss:.4f}",
                    "G Loss": f"{avg_g_loss:.4f}",
                })

        return float(torch.tensor(d_losses).mean()), float(torch.tensor(g_losses).mean())

    def generate_images(self, num_images: int = 16) -> torch.Tensor:
        """Generate images from random noise."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
        return fake_images.cpu()

    def save_models(self, path: str):
        """Save model checkpoints."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'latent_dim': self.latent_dim
        }, path)

    def load_models(self, path: str):
        """Load model checkpoints."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])


class GANFactory:
    """Factory class to create GAN models based on requirements."""

    @staticmethod
    def create_generator(latent_dim: int = 100, output_channels: int = 1) -> Generator:
        """Create a Generator model."""
        return Generator(latent_dim=latent_dim, output_channels=output_channels)

    @staticmethod
    def create_discriminator(input_channels: int = 1) -> Discriminator:
        """Create a Discriminator model."""
        return Discriminator(input_channels=input_channels)

    @staticmethod
    def create_gan(dataset: str, **kwargs) -> BaseGAN:
        """
        Create complete GAN model based on dataset.

        Args:
            dataset: 'mnist', 'cifar'
            **kwargs: Model-specific parameters
        """
        return GAN(dataset=dataset, **kwargs)

def get_dataloader(dataset: str, batch_size: int = 128, download: bool = True) -> DataLoader:
    """Get dataloader for the specified dataset with proper transforms."""
    dataset = dataset.lower()
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    elif dataset == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader


def train_gan(dataset: str, epochs: int = 50, batch_size: int = 128, save_path: str = "checkpoints/gan_{dataset}/gan_{dataset}.pth", checkpoint_dir: str = "checkpoints", **kwargs) -> Tuple[GAN, List[float], List[float]]:
    """Train GAN on the specified dataset."""
    gan = GAN(dataset=dataset, **kwargs)
    dataloader = get_dataloader(dataset, batch_size)

    print(f"Training GAN on {dataset.upper()} for {epochs} epochs...")
    print(f"Using device: {gan.device}")

    # Track losses across epochs
    d_losses_history = []
    g_losses_history = []

    for epoch in range(epochs):
        d_loss, g_loss = gan.train_epoch(dataloader, epoch, epochs)

        # Store losses for return
        d_losses_history.append(d_loss)
        g_losses_history.append(g_loss)

        # Save checkpoint every epoch
        models = {'generator': gan.generator, 'discriminator': gan.discriminator}
        optimizers = {'generator': gan.g_optimizer, 'discriminator': gan.d_optimizer}
        losses = {'d_loss': d_loss, 'g_loss': g_loss}
        checkpoint_path = f"{checkpoint_dir}/gan_{dataset}"
        save_checkpoint(models, optimizers, epoch + 1, losses, {}, checkpoint_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    # Save trained model
    save_path = save_path.format(dataset=dataset)
    gan.save_models(save_path)
    print(f"Training completed. Model saved to {save_path}")

    return gan, d_losses_history, g_losses_history


# Helper functions for image processing
def save_generated_images(images: torch.Tensor, dataset: str, timestamp: str, prefix: str = "") -> List[str]:
    """Save generated images to disk and return file paths."""
    # Create directory for generated images
    output_dir = f"results/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    for i, img in enumerate(images):
        # Denormalize from [-1, 1] to [0, 1]
        img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
        
        # Convert to PIL Image
        if img.shape[0] == 1:  # Grayscale (MNIST)
            img_pil = Image.fromarray((img_denorm.squeeze().numpy() * 255).astype('uint8'), mode='L')
        else:  # RGB (CIFAR)
            img_pil = Image.fromarray((img_denorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        # Save image
        filename = f"{prefix}{dataset}_generated_{i+1}.png"

        filepath = os.path.join(output_dir, filename)
        img_pil.save(filepath)
        image_paths.append(filepath)
    
    return image_paths

def images_to_base64(images: torch.Tensor) -> List[str]:
    """Convert tensor images to base64 encoded strings."""
    base64_images = []
    for img in images:
        # Denormalize from [-1, 1] to [0, 1]
        img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
        
        # Convert to PIL Image
        if img.shape[0] == 1:  # Grayscale (MNIST)
            img_pil = Image.fromarray((img_denorm.squeeze().numpy() * 255).astype('uint8'), mode='L')
        else:  # RGB (CIFAR)
            img_pil = Image.fromarray((img_denorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        # Convert to base64
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_images.append(f"data:image/png;base64,{img_base64}")
    
    return base64_images



# Example usage and configuration
def get_gan(dataset: str, latent_dim: int = 100, device: Optional[str] = None) -> GAN:
    """Get a pre-configured GAN for the specified dataset."""
    return GAN(latent_dim=latent_dim, dataset=dataset, device=device)

def get_generator_only(latent_dim: int = 100, output_channels: int = 1, img_size: int = 28) -> Generator:
    """Get only the Generator network for inference."""
    return Generator(latent_dim=latent_dim, output_channels=output_channels, img_size=img_size)

def get_discriminator_only(input_channels: int = 1, img_size: int = 28) -> Discriminator:
    """Get only the Discriminator network for evaluation."""
    return Discriminator(input_channels=input_channels, img_size=img_size)

def load_trained_gan(model_path: str, dataset: str, device: Optional[str] = None) -> GAN:
    """Load a trained GAN model from checkpoint."""
    gan = GAN(dataset=dataset, device=device)
    gan.load_models(model_path)
    return gan


if __name__ == "__main__":
    # Example: Train GAN on MNIST
    gan = train_gan(dataset='mnist', epochs=50)

    # Generate sample images
    sample_images = gan.generate_images(16)
    print(f"Generated {sample_images.shape[0]} sample images")