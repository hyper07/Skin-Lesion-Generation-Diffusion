"""
Unified Image Generation API for Skin Disease Models
FastAPI service for generating synthetic skin lesion images
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from models.cgan import Generator as CGANGenerator
from models.conditional_diffusion import ConditionalDiffusionModel
from models.prebuilt_gan import ConditionalGenerator
from models.prebuilt_diffusion import DiffusionModel
from device_utils import get_device

# ============================
# Configuration
# ============================

CHECKPOINT_PATHS = {
    "cgan": "checkpoints/cgan/cgan128_final.pt",
    "conditional_diffusion": "checkpoints/conditional_diffusion/diffusion_final.pt",
    "prebuilt_gan": "checkpoints/prebuilt_gan/G_final.pt",
    "prebuilt_diffusion": "checkpoints/prebuilt_diffusion/prebuilt_diffusion_epoch_88_final.pt"
}

# 13 classes in order from data_loader.py
DISEASE_CLASSES = {
    "Actinic keratosis": 0,
    "Basal cell carcinoma": 1,
    "Benign keratosis (HAM)": 2,
    "Dermatofibroma": 3,
    "Melanoma (BCN)": 4,
    "Melanoma (HAM)": 5,
    "Melanoma metastasis": 6,
    "Nevus": 7,
    "Scar": 8,
    "Seborrheic keratosis": 9,
    "Solar lentigo": 10,
    "Squamous cell carcinoma": 11,
    "Vascular lesion": 12
}

# Reverse mapping: index -> name
IDX_TO_CLASS = {v: k for k, v in DISEASE_CLASSES.items()}

OUTPUT_BASE_DIR = "output"

# ============================
# Pydantic Models
# ============================

from enum import Enum

class ModelType(str, Enum):
    cgan = "cgan"
    conditional_diffusion = "conditional_diffusion"
    prebuilt_gan = "prebuilt_gan"
    prebuilt_diffusion = "prebuilt_diffusion"

    

class GenerateRequest(BaseModel):
    model: ModelType  # Now restricted to valid values
    class_ids: Optional[List[int]] = None
    num_samples: int = 50

class ModelStatusResponse(BaseModel):
    cgan: bool
    conditional_diffusion: bool
    prebuilt_gan: bool
    prebuilt_diffusion: bool

# ============================
# FastAPI App
# ============================

app = FastAPI(title="Skin Disease Image Generator")
device = get_device()

# ============================
# Helper Functions
# ============================

def check_checkpoint_exists(path: str) -> bool:
    """Check if checkpoint file exists"""
    return os.path.exists(path)

def load_cgan(checkpoint_path: str):
    """Load CGAN model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_from_checkpoint = checkpoint.get("args", {})
    
    img_size = args_from_checkpoint.get("img_size", 128)
    g_ch = args_from_checkpoint.get("g_ch", 64)
    z_dim = args_from_checkpoint.get("z_dim", 128)
    num_classes = len(DISEASE_CLASSES)
    
    G = CGANGenerator(z_dim, num_classes, base_ch=g_ch, img_size=img_size).to(device)
    G.load_state_dict(checkpoint["G"])
    G.eval()
    
    return G, z_dim


def load_conditional_diffusion(checkpoint_path: str):
    """Load Conditional Diffusion model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint.get("model_config", {})
    image_size = model_config.get("image_size", 256)
    num_timesteps = model_config.get("num_timesteps", 1000)
    use_lora = model_config.get("use_lora", False)
    lora_r = model_config.get("lora_r", 4)
    lora_alpha = model_config.get("lora_alpha", 16)
    lora_dropout = model_config.get("lora_dropout", 0.1)
    model_channels = model_config.get("model_channels", 32)
    channel_mult_str = model_config.get("channel_mult", "1")
    
    try:
        channel_mult = tuple(int(x.strip()) for x in channel_mult_str.split(','))
    except:
        channel_mult = (1,)
    
    num_res_blocks = model_config.get("num_res_blocks", 1)
    num_classes = len(DISEASE_CLASSES)
    
    model = ConditionalDiffusionModel(
        image_size=image_size,
        num_classes=num_classes,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_timesteps=num_timesteps,
        beta_schedule='linear',
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    model.eval()
    return model


def load_prebuilt_gan(checkpoint_path: str):
    """Load Prebuilt GAN model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_from_ckpt = checkpoint.get("args", {})
    
    z_dim = args_from_ckpt.get("z_dim", 128)
    img_size = args_from_ckpt.get("img_size", 128)
    base_channels = args_from_ckpt.get("base_channels", 256)
    num_classes = len(DISEASE_CLASSES)
    
    G = ConditionalGenerator(
        z_dim=z_dim,
        num_classes=num_classes,
        img_size=img_size,
        base_channels=base_channels
    ).to(device)
    
    G.load_state_dict(checkpoint)
    G.eval()
    
    return G, z_dim

def load_prebuilt_diffusion(checkpoint_path: str):
    """Load Prebuilt Diffusion model"""
    num_classes = len(DISEASE_CLASSES)
    model = DiffusionModel(num_classes=num_classes, use_lora=True).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "unet_state_dict" in checkpoint:
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    return model

@torch.no_grad()
def generate_cgan_images(G, z_dim, class_id, class_name, num_samples, output_dir):
    """Generate images using CGAN"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc=f"CGAN - {class_name}", leave=False):
        z = torch.randn(1, z_dim, device=device)
        y = torch.tensor([class_id], dtype=torch.long, device=device)
        fake = G(z, y)
        
        image = (fake.clamp(-1, 1) + 1) / 2
        output_path = output_dir / f"img_{i+1:03d}.png"
        save_image(image, output_path)

@torch.no_grad()
def generate_conditional_diffusion_images(model, class_id, class_name, num_samples, output_dir):
    """Generate images using Conditional Diffusion"""
    os.makedirs(output_dir, exist_ok=True)
    num_inference_steps = 10
    
    for sample_idx in tqdm(range(num_samples), desc=f"Cond. Diffusion - {class_name}", leave=False):
        class_labels = torch.full((1,), class_id, dtype=torch.long, device=device)
        
        x = torch.randn((1, 3, model.image_size, model.image_size), device=device)
        
        step_ratio = model.num_timesteps / num_inference_steps
        timesteps = [int(model.num_timesteps - 1 - i * step_ratio) for i in range(num_inference_steps)]
        timesteps = [max(0, t) for t in timesteps]
        timesteps.append(0)
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_batch = torch.full((1,), t, dtype=torch.long, device=device)
            predicted_noise = model.unet(x, t_batch, class_labels)
            
            alpha_cumprod_t = model.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_cumprod_t_next = model.alphas_cumprod[t_next].view(-1, 1, 1, 1) if t_next >= 0 else torch.ones_like(alpha_cumprod_t)
            
            pred_x_start = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
            
            if t_next == 0:
                x = pred_x_start
            else:
                x = torch.sqrt(alpha_cumprod_t_next) * pred_x_start + torch.sqrt(1.0 - alpha_cumprod_t_next) * predicted_noise
        
        generated = x
        generated = (generated + 1.0) / 2.0
        generated = torch.clamp(generated, 0.0, 1.0)
        
        img_path = output_dir / f"img_{sample_idx+1:03d}.png"
        save_image(generated, img_path, normalize=False)


@torch.no_grad()
def generate_prebuilt_gan_images(G, z_dim, class_id, class_name, num_samples, output_dir):
    """Generate images using Prebuilt GAN"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc=f"Prebuilt GAN - {class_name}", leave=False):
        z = torch.randn(1, z_dim, device=device)
        y = torch.tensor([class_id], dtype=torch.long, device=device)
        fake = G(z, y)
        
        image = (fake.clamp(-1, 1) + 1) / 2
        output_path = output_dir / f"img_{i+1:03d}.png"
        save_image(image, output_path)

@torch.no_grad()
def generate_prebuilt_diffusion_images(model, class_id, class_name, num_samples, output_dir):
    """Generate images using Prebuilt Diffusion"""
    os.makedirs(output_dir, exist_ok=True)
    num_inference_steps = 100
    img_size = 64
    
    for sample_idx in tqdm(range(num_samples), desc=f"Prebuilt Diffusion - {class_name}", leave=False):
        latents = torch.randn((1, 4, img_size // 8, img_size // 8)).to(device)
        
        scheduler = model.scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        for t in scheduler.timesteps:
            class_tensor = torch.tensor([class_id], device=device)
            noise_pred = model.unet(latents, t, class_tensor)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        image = model.vae.decode(latents / 0.18215).sample
        image = (image.clamp(-1, 1) + 1) / 2
        
        img_path = output_dir / f"img_{sample_idx+1:03d}.png"
        save_image(image, img_path)

# ============================
# API Endpoints
# ============================

@app.get("/model_status", response_model=ModelStatusResponse)
async def get_model_status():
    """Check availability of all model checkpoints"""
    return ModelStatusResponse(
        cgan=check_checkpoint_exists(CHECKPOINT_PATHS["cgan"]),
        conditional_diffusion=check_checkpoint_exists(CHECKPOINT_PATHS["conditional_diffusion"]),
        prebuilt_gan=check_checkpoint_exists(CHECKPOINT_PATHS["prebuilt_gan"]),
        prebuilt_diffusion=check_checkpoint_exists(CHECKPOINT_PATHS["prebuilt_diffusion"])
    )

@app.post("/generate")
async def generate_images(request: GenerateRequest):
    """Generate synthetic skin lesion images (cgan, conditional_diffusion, prebuilt_gan, prebuilt_diffusion)"""
    
    # Validate model name
    if request.model not in CHECKPOINT_PATHS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {list(CHECKPOINT_PATHS.keys())}")
    

    # Convert enum to string
    model_name = request.model.value

    # Check checkpoint exists
    checkpoint_path = CHECKPOINT_PATHS[request.model]
    if not check_checkpoint_exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    
    # Validate class_ids
    if request.class_ids is not None:
        for cid in request.class_ids:
            if cid not in IDX_TO_CLASS:
                raise HTTPException(status_code=400, detail=f"Invalid class_id: {cid}. Must be 0-12")
        class_ids = request.class_ids
    else:
        class_ids = list(range(len(DISEASE_CLASSES)))
    

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(OUTPUT_BASE_DIR) / model_name / timestamp
    
    try:
        # Load model
        if model_name == "cgan":
            model, z_dim = load_cgan(checkpoint_path)
            
            for class_id in class_ids:
                class_name = IDX_TO_CLASS[class_id]
                class_dir = output_base / f"class_{class_id}_{class_name}"
                generate_cgan_images(model, z_dim, class_id, class_name, request.num_samples, class_dir)
        
        elif model_name == "conditional_diffusion":
            model = load_conditional_diffusion(checkpoint_path)
            
            for class_id in class_ids:
                class_name = IDX_TO_CLASS[class_id]
                class_dir = output_base / f"class_{class_id}_{class_name}"
                generate_conditional_diffusion_images(model, class_id, class_name, request.num_samples, class_dir)
        
        elif model_name == "prebuilt_gan":
            model, z_dim = load_prebuilt_gan(checkpoint_path)
            
            for class_id in class_ids:
                class_name = IDX_TO_CLASS[class_id]
                class_dir = output_base / f"class_{class_id}_{class_name}"
                generate_prebuilt_gan_images(model, z_dim, class_id, class_name, request.num_samples, class_dir)
        
        elif model_name == "prebuilt_diffusion":
            model = load_prebuilt_diffusion(checkpoint_path)
            
            for class_id in class_ids:
                class_name = IDX_TO_CLASS[class_id]
                class_dir = output_base / f"class_{class_id}_{class_name}"
                generate_prebuilt_diffusion_images(model, class_id, class_name, request.num_samples, class_dir)
        
        return {
            "status": "success",
            "model": model_name,
            "classes_generated": [IDX_TO_CLASS[cid] for cid in class_ids],
            "num_samples_per_class": request.num_samples,
            "total_images": len(class_ids) * request.num_samples,
            "output_directory": str(output_base)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ============================
# Run Server
# ============================

if __name__ == "__main__":
    import uvicorn
    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)