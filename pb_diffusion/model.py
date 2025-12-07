# model.py

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from peft import get_peft_model, LoraConfig, TaskType


class DiseaseConditionedUNet(nn.Module):
    """
    Wraps a pre-trained UNet2DConditionModel and injects class conditioning
    via learnable embeddings in place of text encoder outputs.
    Supports optional LoRA injection for efficient fine-tuning.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 768, use_lora: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.class_embed = nn.Embedding(num_classes, embedding_dim)

        # Load pre-trained UNet from Stable Diffusion
        base_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        )

        if use_lora:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # adjust based on UNet internals
                lora_dropout=0.1,
                bias="none"
            )
            self.unet = get_peft_model(base_unet, lora_config)
        else:
            self.unet = base_unet

    def forward(self, noisy_latents: torch.Tensor, timestep: torch.Tensor, class_id: torch.Tensor):
        """
        Args:
            noisy_latents: [B, 4, H, W] - latent image with noise
            timestep: [B] - diffusion timestep
            class_id: [B] - integer class labels

        Returns:
            Predicted noise residual
        """
        # Embed class labels
        class_embedding = self.class_embed(class_id)  # [B, D]
        class_embedding = class_embedding.unsqueeze(1)  # [B, 1, D]

        # Forward through UNet with class conditioning
        output = self.unet(
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=class_embedding
        )
        return output.sample


class DiffusionModel(nn.Module):
    """
    Combines the VAE, UNet, and scheduler into a trainable diffusion model.
    """

    def __init__(self, num_classes: int, use_lora: bool = False):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.unet = DiseaseConditionedUNet(num_classes, use_lora=use_lora)
        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

    def forward(self, image: torch.Tensor, class_id: torch.Tensor):
        """
        Args:
            image: [B, 3, H, W] - clean RGB image
            class_id: [B] - integer class labels

        Returns:
            predicted_noise: [B, 4, H, W]
            true_noise: [B, 4, H, W]
        """
        device = image.device

        # Encode image to latent space
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample() * 0.18215  # scaling factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (image.shape[0],), device=device
        ).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        predicted_noise = self.unet(noisy_latents, timesteps, class_id)

        return predicted_noise, noise
