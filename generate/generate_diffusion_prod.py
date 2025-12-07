"""
Generation script for Conditional Diffusion Model
"""

import sys
from pathlib import Path
import torch
import argparse
from torchvision.utils import save_image
import os
import requests
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.conditional_diffusion import ConditionalDiffusionModel
from device_utils import get_device
from data.dataset_utils import download_from_google_drive


@torch.no_grad()
def generate_samples(model, disease_classes, device, class_id=None, num_samples=4, 
                     num_inference_steps=50, out_dir="output/conditional_diffusion"):
    """Generate samples for a specific class or all classes"""
    model.eval()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(out_dir) / timestamp
    
    # Get class names
    class_names = {idx: name for name, idx in disease_classes.items()}
    num_classes = len(disease_classes)
    
    if class_id is not None:
        # Generate for specific class only
        class_dirs = [class_id]
    else:
        # Generate for all classes
        class_dirs = range(num_classes)
    
    for cls_idx in class_dirs:
        # Create class directory
        class_dir = output_base / f"class_{cls_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_samples} samples for class {cls_idx} ({class_names[cls_idx]})...")
        
        # Generate samples one by one
        for sample_idx in range(num_samples):
            class_labels = torch.full((1,), cls_idx, dtype=torch.long, device=device)
            
            # Start from pure noise
            x = torch.randn((1, 3, model.image_size, model.image_size), device=device)
            
            # DDIM sampling
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
            generated = (generated + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            generated = torch.clamp(generated, 0.0, 1.0)
            
            # Save individual image
            img_path = class_dir / f"image_{sample_idx:03d}.png"
            save_image(generated, img_path, normalize=False)
        
        print(f"  Saved to {class_dir}")
    
    print(f"\nAll samples saved to {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Conditional Diffusion Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/conditional_diffusion/diffusion_final.pt", help="Path to checkpoint file")
    parser.add_argument("--class_id", type=int, default=None, help="Class ID to generate (None for all classes)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate per class")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--output", type=str, default="output/conditional_diffusion", help="Output base directory")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists, download if not
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found, downloading...")
        file_id = "1jov76Vx3M0N4yWInejbMaJMm6Gm3pxio"
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        if not download_from_google_drive(file_id, args.checkpoint):
            print("Failed to download checkpoint.")
            return
        print("Downloaded.")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    disease_classes = checkpoint.get("disease_classes", checkpoint.get("classes", {}))
    num_classes = checkpoint.get("num_classes", len(disease_classes))
    
    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    print(f"Model config from checkpoint: {model_config}")
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
    
    print(f"Loaded checkpoint with {num_classes} classes")
    print(f"Classes: {list(disease_classes.keys())}")
    print(f"Image size: {image_size}, Timesteps: {num_timesteps}")
    print(f"Model config: channels={model_channels}, channel_mult={channel_mult}, res_blocks={num_res_blocks}")
    if use_lora:
        print(f"Using LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Create model
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
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No model_state_dict found in checkpoint")
    
    # Generate samples
    generate_samples(model, disease_classes, device, args.class_id, args.num_samples, 
                     args.num_inference_steps, args.output)


if __name__ == "__main__":
    main()