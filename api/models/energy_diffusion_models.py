"""
Energy Model and Diffusion Model Definitions
Generative models for image generation on CIFAR-10.
Based on Module 8 Practical implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import os
import math
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import checkpoint utilities
from help_lib.checkpoints import save_checkpoint, get_checkpoint_base_dir


# ============= Energy Model =============

def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """
    Energy-Based Model for CIFAR-10.
    Based on Module 8 Practical 1 implementation.
    """
    
    def __init__(self, input_channels: int = 3, img_size: int = 32):
        super(EnergyModel, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        
        # CNN-based energy function (similar to reference)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        # Calculate size after 4 conv layers with stride 2: img_size // 16
        spatial_size = img_size // 16
        self.fc1 = nn.Linear(64 * spatial_size * spatial_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Output is a single energy value
        
    def forward(self, x):
        """Compute energy E(x) for input x."""
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x).squeeze()
    
    def energy(self, x):
        """Alias for forward."""
        return self.forward(x)


def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Generate samples using Langevin dynamics.
    Based on Module 8 Practical 1.
    """
    nn_energy_model.eval()
    
    for _ in range(steps):
        # Add noise
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
        
        inp_imgs.requires_grad_(True)
        
        # Compute energy and gradients
        energy = nn_energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))
        
        # Apply gradient clipping for stability
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    
    return inp_imgs.detach()


class Buffer:
    """
    Buffer for maintaining samples in Energy Model training.
    Based on Module 8 Practical 1.
    """
    def __init__(self, model, device, buffer_size: int = 8192, initial_size: int = 128):
        self.model = model
        self.device = device
        # Start with random images in the buffer
        self.examples = [torch.rand((1, 3, 32, 32), device=self.device) * 2 - 1 
                         for _ in range(initial_size)]
        self.buffer_size = buffer_size
    
    def sample_new_exmps(self, steps, step_size, noise):
        """Generate new examples using buffer."""
        n_new = np.random.binomial(128, 0.05)
        
        # Generate new random images for around 5% of the inputs
        new_rand_imgs = torch.rand((n_new, 3, 32, 32), device=self.device) * 2 - 1
        
        # Sample old images from the buffer for the rest
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
        
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        
        # Run Langevin dynamics
        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)
        
        # Update buffer
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:self.buffer_size]
        
        return new_imgs


class EnergyModelTrainer:
    """
    Trainer for Energy-Based Model using Contrastive Divergence.
    Based on Module 8 Practical 1.
    """
    
    def __init__(self, model: EnergyModel, device: Optional[str] = None, 
                 learning_rate: float = 0.0001, alpha: float = 0.1, 
                 steps: int = 60, step_size: float = 10.0, noise: float = 0.005):
        self.model = model
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.0, 0.999))
        self.buffer = Buffer(self.model, self.device)
    
    def train_step(self, real_images: torch.Tensor) -> float:
        """Train on one batch using contrastive divergence."""
        batch_size = real_images.size(0)
        self.model.train()
        
        # Add noise to real images
        real_imgs = real_images + torch.randn_like(real_images) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)
        
        # Generate fake images using buffer
        fake_imgs = self.buffer.sample_new_exmps(
            steps=self.steps, step_size=self.step_size, noise=self.noise)
        
        # Concatenate and compute energies
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)
        
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)
        
        # Contrastive divergence loss
        cdiv_loss = real_out.mean() - fake_out.mean()
        # Regularization loss
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch=None, total_epochs=None) -> float:
        """Train for one epoch."""
        losses = []
        
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
            loss = self.train_step(real_images)
            losses.append(loss)
            
            if hasattr(iterator, 'set_postfix'):
                avg_loss = sum(losses) / len(losses)
                iterator.set_postfix({"Loss": f"{avg_loss:.4f}"})
        
        return float(torch.tensor(losses).mean())
    
    def generate_samples(self, num_samples: int = 16, num_steps: int = 256, 
                        step_size: float = 10.0, noise_std: float = 0.01) -> torch.Tensor:
        """Generate samples using Langevin dynamics."""
        self.model.eval()
        # Initialize from random noise
        x = torch.rand((num_samples, self.model.input_channels, self.model.img_size, self.model.img_size), 
                      device=self.device) * 2 - 1
        return generate_samples(self.model, x, num_steps, step_size, noise_std).cpu()


# ============= Diffusion Model =============

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timesteps.
    Based on Module 8 Practical 2.
    """
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """
    Offset cosine diffusion schedule.
    Based on Module 8 Practical 2.
    """
    # Flatten diffusion_times to handle any shape
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Compute start and end angles from signal rate bounds
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Linearly interpolate angles
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    # Compute signal and noise rates
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    """Residual block for UNet."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block for UNet."""
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for UNet."""
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for Diffusion Model.
    Based on Module 8 Practical 2.
    """
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2) 

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2) 
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips) 
        x = self.down3(x, skips)    

        x = self.mid1(x)     
        x = self.mid2(x)   

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    """
    Diffusion Model (DDPM) for CIFAR-10.
    Based on Module 8 Practical 2.
    """
    
    def __init__(self, input_channels: int = 3, img_size: int = 32, embedding_dim: int = 32):
        super(DiffusionModel, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.network = UNet(img_size, input_channels, embedding_dim)
        self.schedule_fn = offset_cosine_diffusion_schedule
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def set_normalizer(self, mean, std):
        """Set normalization parameters."""
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        """Denormalize images."""
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Denoise images."""
        if training:
            network = self.network
            network.train()
        else:
            network = self.network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Reverse diffusion process for sampling."""
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=32, initial_noise=None):
        """Generate images."""
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.input_channels, image_size, image_size), 
                                      device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))


class DiffusionTrainer:
    """
    Trainer for Diffusion Model.
    Based on Module 8 Practical 2.
    """
    
    def __init__(self, model: DiffusionModel, device: Optional[str] = None, 
                 learning_rate: float = 0.001):
        self.model = model
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.L1Loss()
    
    def train_step(self, real_images: torch.Tensor) -> float:
        """Train on one batch."""
        batch_size = real_images.size(0)
        self.model.train()
        self.optimizer.zero_grad()
        
        # Normalize images
        images = (real_images - self.model.normalizer_mean) / self.model.normalizer_std
        noises = torch.randn_like(images)
        
        # Sample random timesteps
        diffusion_times = torch.rand((batch_size, 1, 1, 1), device=self.device)
        noise_rates, signal_rates = self.model.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        
        pred_noises, _ = self.model.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = self.loss_fn(pred_noises, noises)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch=None, total_epochs=None) -> float:
        """Train for one epoch."""
        losses = []
        
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
            loss = self.train_step(real_images)
            losses.append(loss)
            
            if hasattr(iterator, 'set_postfix'):
                avg_loss = sum(losses) / len(losses)
                iterator.set_postfix({"Loss": f"{avg_loss:.4f}"})
        
        return float(torch.tensor(losses).mean())
    
    def sample(self, num_samples: int = 16, img_size: int = 32, channels: int = 3, 
               diffusion_steps: int = 20) -> torch.Tensor:
        """Generate samples using reverse diffusion."""
        return self.model.generate(num_samples, diffusion_steps, img_size).cpu()


# ============= Factory and Helper Functions =============

def get_energy_model(input_channels: int = 3, img_size: int = 32, device: Optional[str] = None) -> EnergyModel:
    """Create an Energy Model for CIFAR-10."""
    model = EnergyModel(input_channels=input_channels, img_size=img_size)
    if device:
        model = model.to(device)
    return model


def get_diffusion_model(input_channels: int = 3, img_size: int = 32, 
                       device: Optional[str] = None) -> DiffusionModel:
    """Create a Diffusion Model for CIFAR-10."""
    model = DiffusionModel(input_channels=input_channels, img_size=img_size)
    if device:
        model = model.to(device)
    return model


def get_cifar10_dataloader(batch_size: int = 128, download: bool = True) -> DataLoader:
    """Get CIFAR-10 dataloader with proper transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader


def train_energy_model(epochs: int = 50, batch_size: int = 128, learning_rate: float = 0.0001,
                      save_path: str = None,
                      checkpoint_dir: str = None, **kwargs) -> Tuple[EnergyModelTrainer, List[float]]:
    """Train Energy Model on CIFAR-10."""
    # Set default paths relative to sps_genai/checkpoints
    base_dir = get_checkpoint_base_dir()
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(base_dir, "energy_cifar")
    if save_path is None:
        save_path = os.path.join(checkpoint_dir, "energy_cifar.pth")
    
    model = EnergyModel(input_channels=3, img_size=32)
    trainer = EnergyModelTrainer(model, learning_rate=learning_rate, **kwargs)
    dataloader = get_cifar10_dataloader(batch_size)
    
    print(f"Training Energy Model on CIFAR-10 for {epochs} epochs...")
    print(f"Using device: {trainer.device}")
    
    losses_history = []
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader, epoch, epochs)
        losses_history.append(loss)
        
        # Save checkpoint
        models = {'energy_model': trainer.model}
        optimizers = {'energy_optimizer': trainer.optimizer}
        losses = {'loss': loss}
        save_checkpoint(models, optimizers, epoch + 1, losses, {}, checkpoint_dir)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f}")
    
    # Final model is already saved in latest_checkpoint.pth, just update metadata
    # Also save a separate final model file for backward compatibility
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, save_path)
    print(f"Training completed. Latest checkpoint: {os.path.join(checkpoint_dir, 'latest_checkpoint.pth')}")
    print(f"Final model path: {save_path}")
    
    return trainer, losses_history


def train_diffusion_model(epochs: int = 50, batch_size: int = 128, learning_rate: float = 0.001,
                         save_path: str = None,
                         checkpoint_dir: str = None, **kwargs) -> Tuple[DiffusionTrainer, List[float]]:
    """Train Diffusion Model on CIFAR-10."""
    # Set default paths relative to sps_genai/checkpoints
    base_dir = get_checkpoint_base_dir()
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(base_dir, "diffusion_cifar")
    if save_path is None:
        save_path = os.path.join(checkpoint_dir, "diffusion_cifar.pth")
    
    model = DiffusionModel(input_channels=3, img_size=32)
    trainer = DiffusionTrainer(model, learning_rate=learning_rate, **kwargs)
    dataloader = get_cifar10_dataloader(batch_size)
    
    # Calculate normalization statistics
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
    
    mean /= total_samples
    std /= total_samples
    mean = mean.reshape(1, 3, 1, 1).to(trainer.device)
    std = std.reshape(1, 3, 1, 1).to(trainer.device)
    model.set_normalizer(mean, std)
    
    print(f"Training Diffusion Model on CIFAR-10 for {epochs} epochs...")
    print(f"Using device: {trainer.device}")
    print(f"Normalization stats - Mean: {mean.squeeze()}, Std: {std.squeeze()}")
    
    losses_history = []
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader, epoch, epochs)
        losses_history.append(loss)
        
        # Save checkpoint - save the network state dict, not the whole model
        models = {'diffusion_model': trainer.model.network}
        optimizers = {'diffusion_optimizer': trainer.optimizer}
        losses = {'loss': loss}
        save_checkpoint(models, optimizers, epoch + 1, losses, {}, checkpoint_dir)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f}")
    
    # Final model is already saved in latest_checkpoint.pth, just update metadata
    # Also save a separate final model file for backward compatibility
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.network.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'normalizer_mean': model.normalizer_mean,
        'normalizer_std': model.normalizer_std,
    }, save_path)
    print(f"Training completed. Latest checkpoint: {os.path.join(checkpoint_dir, 'latest_checkpoint.pth')}")
    print(f"Final model path: {save_path}")
    
    return trainer, losses_history
