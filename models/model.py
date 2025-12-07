"""
Conditional Diffusion Model for Skin Lesion Generation

Architecture: U-Net with time and class conditioning
Input: Disease class + noise
Output: Synthetic lesion image of that class
"""
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings for diffusion process"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ClassEmbedding(nn.Module):
    """Embedding layer for class labels"""

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(class_labels)


class ResidualBlock(nn.Module):
    """Residual block with time and class conditioning"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, class_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, class_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)

        # Add time conditioning
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # Add class conditioning
        class_emb = self.class_mlp(class_emb)
        h = h + class_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block with improved stability"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, H * W)  # [B, heads, head_dim, HW]
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)  # [B, heads, HW, HW]
        h = (attn @ v).transpose(2, 3).contiguous().view(B, C, H, W)  # [B, C, H, W]

        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    """
    U-Net architecture for conditional diffusion model
    Input: Noisy image + time step + class label
    Output: Predicted noise
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_classes: int = 3,
        time_emb_dim: int = 256,
        class_emb_dim: int = 128,
        dropout: float = 0.1,
        attention_resolutions: Tuple[int, ...] = (16, 8),
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels // 4),
            nn.Linear(model_channels // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Class embedding
        self.class_embed = ClassEmbedding(num_classes, class_emb_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        input_ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(input_ch, out_ch, time_emb_dim, class_emb_dim, dropout))
                input_ch = out_ch

            # Attention at specific resolutions
            if image_size // (2 ** (i + 1)) in attention_resolutions:
                self.down_blocks.append(AttentionBlock(out_ch))

            # Downsample (except for last level)
            if i < len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

        # Middle block
        self.mid_block1 = ResidualBlock(input_ch, input_ch, time_emb_dim, class_emb_dim, dropout)
        self.mid_attn = AttentionBlock(input_ch)
        self.mid_block2 = ResidualBlock(input_ch, input_ch, time_emb_dim, class_emb_dim, dropout)

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):  # +1 for the attention/skip connection
                self.up_blocks.append(ResidualBlock(input_ch + out_ch, out_ch, time_emb_dim, class_emb_dim, dropout))
                input_ch = out_ch

            # Attention at specific resolutions
            if image_size // (2 ** (i + 1)) in attention_resolutions:
                self.up_blocks.append(AttentionBlock(out_ch))

            # Upsample (except for last level)
            if i > 0:
                self.up_blocks.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timestep)

        # Class embedding
        class_emb = self.class_embed(class_labels)

        # Encoder
        h = self.input_conv(x)
        hs = [h]  # Skip connections

        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, class_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # Downsampling conv
                hs.append(h)
                h = block(h)

        # Middle
        h = self.mid_block1(h, time_emb, class_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, class_emb)

        # Decoder
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                # Concatenate skip connection
                if hs:
                    h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb, class_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # Upsampling conv
                h = block(h)

        return self.output_conv(h)


class ConditionalDiffusionModel(nn.Module):
    """
    Complete conditional diffusion model

    Input: Disease class + noise
    Output: Synthetic lesion image of that class
    """

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256,
        class_emb_dim: int = 128,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
    ):
        super().__init__()
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes

        # Create U-Net
        self.unet = ConditionalUNet(
            image_size=image_size,
            in_channels=3,
            out_channels=3,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            num_classes=num_classes,
            time_emb_dim=time_emb_dim,
            class_emb_dim=class_emb_dim,
            dropout=dropout,
        )

        # Register diffusion parameters
        self.register_buffer('betas', self._get_beta_schedule(num_timesteps, beta_schedule))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Posterior variance for q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def _get_beta_schedule(self, num_timesteps: int, schedule: str = 'linear') -> torch.Tensor:
        if schedule == 'linear':
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to images
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step
        p(x_{t-1} | x_t, c)
        """
        # Predict noise
        predicted_noise = self.unet(x, t, class_labels)

        # Get alpha and beta values
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        # Predict x_0
        pred_x_start = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        if t[0] == 0:
            pred_mean = pred_x_start
        else:
            alpha_cumprod_prev = self.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
            pred_mean = (torch.sqrt(alpha_cumprod_prev) * beta_t / (1.0 - alpha_cumprod_t)) * pred_x_start + \
                       (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)) * x

        # Sample
        if t[0] == 0:
            return pred_mean
        else:
            posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return pred_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, class_labels: Union[int, torch.Tensor], batch_size: int = 1, num_inference_steps: int = 50, use_ddim: bool = True) -> torch.Tensor:
        """
        Generate synthetic images from class labels

        Args:
            class_labels: Class labels [B] or int
            batch_size: Batch size (if class_labels is int)
            num_inference_steps: Number of denoising steps
            use_ddim: Use DDIM sampling (more stable) or DDPM

        Returns:
            Generated images [B, C, H, W]
        """
        self.eval()

        # Handle single class label
        if isinstance(class_labels, int):
            class_labels = torch.full((batch_size,), class_labels, dtype=torch.long, device=self.betas.device)
        else:
            class_labels = class_labels.detach().clone().to(dtype=torch.long, device=self.betas.device)
            batch_size = len(class_labels)

        # Start from pure noise
        shape = (batch_size, 3, self.image_size, self.image_size)
        x = torch.randn(shape, device=self.betas.device)

        if use_ddim:
            # DDIM sampling - more deterministic and stable
            # Create evenly spaced timesteps
            step_ratio = self.num_timesteps / num_inference_steps
            timesteps = [int(self.num_timesteps - 1 - i * step_ratio) for i in range(num_inference_steps)]
            timesteps = [max(0, t) for t in timesteps]  # Ensure non-negative
            timesteps.append(0)  # Ensure we end at 0

            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_next = timesteps[i + 1]

                # Expand for batch
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.betas.device)

                # Predict noise
                predicted_noise = self.unet(x, t_batch, class_labels)

                # Get alpha values
                alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                alpha_cumprod_t_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1) if t_next >= 0 else torch.ones_like(alpha_cumprod_t)

                # Predict x_0
                pred_x_start = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)

                # DDIM update: x_{t-1} = sqrt(alpha_{t-1}) * pred_x_0 + sqrt(1 - alpha_{t-1}) * predicted_noise
                if t_next == 0:
                    x = pred_x_start
                else:
                    x = torch.sqrt(alpha_cumprod_t_next) * pred_x_start + torch.sqrt(1.0 - alpha_cumprod_t_next) * predicted_noise
        else:
            # Original DDPM sampling
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps + 1, dtype=torch.long, device=self.betas.device)

            for i in range(num_inference_steps):
                t = timesteps[i:i+1].expand(batch_size)
                x = self.p_sample(x, t, class_labels)

        # Denormalize (assuming images are normalized to [-1, 1])
        x = torch.clamp(x, -1.0, 1.0)
        return x

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: predict noise from noisy image and class

        Args:
            x: Clean images [B, C, H, W]
            class_labels: Class labels [B]

        Returns:
            predicted_noise, noise, timesteps
        """
        batch_size = x.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

        # Sample noise
        noise = torch.randn_like(x)

        # Add noise to images
        x_noisy = self.q_sample(x, t, noise)

        # Predict noise
        predicted_noise = self.unet(x_noisy, t, class_labels)

        return predicted_noise, noise, t
