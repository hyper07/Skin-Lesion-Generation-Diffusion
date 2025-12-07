"""
Conditional Diffusion Model for Skin Lesion Generat        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )isease class + noise
Output: Synthetic lesion image of that class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings for diffusion process"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ClassEmbedding(nn.Module):
    """Embedding layer for class labels"""
    
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, class_labels):
        return self.embedding(class_labels)


class ResidualBlock(nn.Module):
    """Residual block with time and class conditioning"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, dropout=0.1):
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
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb, class_emb):
        h = self.block1(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add class conditioning
        class_emb = self.class_mlp(class_emb)
        h = h + class_emb[:, :, None, None]
        
        h = self.block2(h)
        return h + self.res_conv(x)


class LoRALinear(nn.Module):
    """LoRA adapter for Linear/Conv2d layers (1x1 conv)"""
    def __init__(self, base_layer, r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        
        # Get input and output dimensions
        if isinstance(base_layer, nn.Conv2d):
            # For 1x1 Conv2d (used in attention), treat as linear layer
            assert base_layer.kernel_size == (1, 1), "LoRA only supports 1x1 Conv2d"
            in_dim = base_layer.in_channels
            out_dim = base_layer.out_channels
        elif isinstance(base_layer, nn.Linear):
            in_dim = base_layer.in_features
            out_dim = base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA adapters - applied to channel dimension
        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
    def forward(self, x):
        # Base layer output
        base_out = self.base_layer(x)
        
        # LoRA adaptation
        if isinstance(self.base_layer, nn.Conv2d):
            # For 1x1 Conv2d: apply LoRA per spatial location
            # x: [B, C_in, H, W]
            B, C_in, H, W = x.shape
            x_reshaped = x.view(B, C_in, H * W).transpose(1, 2)  # [B, HW, C_in]
            x_dropped = self.lora_dropout(x_reshaped)
            lora_out = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling  # [B, HW, C_out]
            lora_out = lora_out.transpose(1, 2).view(B, -1, H, W)  # [B, C_out, H, W]
        else:
            # For Linear
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_out + lora_out


class AttentionBlock(nn.Module):
    """Self-attention block with optional LoRA support"""
    
    def __init__(self, channels, use_lora=False, lora_r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.channels = channels
        self.norm = nn.BatchNorm2d(channels)
        
        # Base layers
        base_qkv = nn.Conv2d(channels, channels * 3, 1)
        base_proj = nn.Conv2d(channels, channels, 1)
        
        # Apply LoRA if requested
        if use_lora:
            self.qkv = LoRALinear(base_qkv, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.proj = LoRALinear(base_proj, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            self.qkv = base_qkv
            self.proj = base_proj
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        k = k.view(B, C, H * W)  # [B, C, HW]
        v = v.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        
        # Attention
        attn = torch.softmax(q @ k / math.sqrt(C), dim=-1)  # [B, HW, HW]
        h = (attn @ v).transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    """
    U-Net architecture for conditional diffusion model
    Input: Noisy image + time step + class label
    Output: Predicted noise
    """
    
    def __init__(
        self,
        image_size=256,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        num_classes=3,
        time_emb_dim=256,
        class_emb_dim=128,
        dropout=0.1,
        attention_resolutions=(16, 8),
        use_lora=False,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
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
            output_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(input_ch, output_ch, time_emb_dim, class_emb_dim, dropout)
                )
                input_ch = output_ch
                
                # Add attention at specific resolutions
                if image_size // (2 ** i) in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(output_ch, use_lora=use_lora, 
                                                           lora_r=lora_r, lora_alpha=lora_alpha, 
                                                           lora_dropout=lora_dropout))
            
            if i != len(channel_mult) - 1:  # Don't downsample after last block
                self.down_blocks.append(nn.Conv2d(output_ch, output_ch, 3, stride=2, padding=1))
        
        # Middle block
        self.mid_block1 = ResidualBlock(input_ch, input_ch, time_emb_dim, class_emb_dim, dropout)
        self.mid_attn = AttentionBlock(input_ch, use_lora=use_lora, 
                                       lora_r=lora_r, lora_alpha=lora_alpha, 
                                       lora_dropout=lora_dropout)
        self.mid_block2 = ResidualBlock(input_ch, input_ch, time_emb_dim, class_emb_dim, dropout)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        current_ch = model_channels * channel_mult[-1]  # Start from the deepest level
        
        # First level (no upsampling, no concatenation)
        for _ in range(num_res_blocks):
            self.up_blocks.append(ResidualBlock(current_ch, current_ch, time_emb_dim, class_emb_dim, dropout))
            if image_size // (2 ** (len(channel_mult) - 1)) in attention_resolutions:
                self.up_blocks.append(AttentionBlock(current_ch, use_lora=use_lora, 
                                                    lora_r=lora_r, lora_alpha=lora_alpha, 
                                                    lora_dropout=lora_dropout))
        
        # Subsequent levels with upsampling and concatenation
        for i, mult in enumerate(reversed(channel_mult[:-1])):
            output_ch = model_channels * mult
            # Upsample
            self.up_blocks.append(nn.ConvTranspose2d(current_ch, output_ch, 4, stride=2, padding=1))
            
            # Residual blocks with concatenation for first block
            for j in range(num_res_blocks):
                if j == 0:
                    self.up_blocks.append(ResidualBlock(output_ch * 2, output_ch, time_emb_dim, class_emb_dim, dropout))
                else:
                    self.up_blocks.append(ResidualBlock(output_ch, output_ch, time_emb_dim, class_emb_dim, dropout))
                
                # Add attention
                if image_size // (2 ** (len(channel_mult) - 2 - i)) in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(output_ch, use_lora=use_lora, 
                                                        lora_r=lora_r, lora_alpha=lora_alpha, 
                                                        lora_dropout=lora_dropout))
            
            current_ch = output_ch
        
        # Output projection
        self.output_norm = nn.BatchNorm2d(model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def forward(self, x, timestep, class_labels):
        """
        Args:
            x: Noisy image [B, C, H, W]
            timestep: Time step [B]
            class_labels: Class labels [B]
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time and class embeddings
        time_emb = self.time_embed(timestep)
        class_emb = self.class_embed(class_labels)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder - save skip connections at the end of each level
        skip_connections = []
        last_h = None
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb, class_emb)
                last_h = h
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Downsampling
                skip_connections.append(last_h)
                h = module(h)
        
        # Append the last level's skip
        skip_connections.append(last_h)
        
        # Middle
        h = self.mid_block1(h, time_emb, class_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, class_emb)
        
        # Decoder - use skip connections
        skip_idx = len(skip_connections) - 2  # Start from the second last skip
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                # Check if this is the first block in a level (after upsampling)
                # We need to concatenate skip connection if available
                if skip_idx >= 0 and skip_connections:
                    # Check if we should concatenate (matching channel logic)
                    expected_channels = module.block1[2].in_channels
                    if h.shape[1] * 2 == expected_channels and skip_idx >= 0:
                        skip = skip_connections[skip_idx]
                        # Ensure spatial dimensions match
                        if h.shape[2:] != skip.shape[2:]:
                            skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
                        h = torch.cat([h, skip], dim=1)
                        skip_idx -= 1
                h = module(h, time_emb, class_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Upsampling
                h = module(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)


class ConditionalDiffusionModel(nn.Module):
    """
    Complete conditional diffusion model
    
    Input: Disease class + noise
    Output: Synthetic lesion image of that class
    """
    
    def __init__(
        self,
        image_size=256,
        num_classes=3,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=256,
        class_emb_dim=128,
        dropout=0.1,
        num_timesteps=1000,
        beta_schedule='linear',
        use_lora=False,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        
        # UNet backbone
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
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Noise schedule
        self.register_buffer('betas', self._get_beta_schedule(num_timesteps, beta_schedule))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    
    def _get_beta_schedule(self, num_timesteps, schedule='linear'):
        """Generate noise schedule"""
        if schedule == 'linear':
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            # Cosine schedule
            s = 0.008
            timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alphas_cumprod = torch.cos(((timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to images
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t, class_labels, clip_denoised=True):
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
            alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
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
    def sample(self, class_labels, batch_size=1, num_inference_steps=50, use_ddim=True):
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
    
    def forward(self, x, class_labels):
        """
        Training forward pass
        
        Args:
            x: Clean images [B, C, H, W]
            class_labels: Class labels [B]
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to images
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.unet(x_noisy, t, class_labels)
        
        return predicted_noise, noise, t