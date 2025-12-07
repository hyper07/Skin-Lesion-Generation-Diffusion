import os
import torch
from torchvision.utils import save_image
from model import DiffusionModel
from diffusers import DDPMScheduler
from datetime import datetime
import argparse
from pathlib import Path

# ============================
# 🔧 Configuration
# ============================

IMG_SIZE = 64
NUM_INFERENCE_STEPS = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"

# ============================
# 📁 Utility Functions
# ============================

def find_latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError("No checkpoint files found.")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    print(f"✓ Found latest checkpoint: {files[0]}")
    return os.path.join(checkpoint_dir, files[0])

# ============================
# 🧠 Load Model
# ============================

def load_model(checkpoint_path, num_classes):
    model = DiffusionModel(num_classes=num_classes, use_lora=True).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if "unet_state_dict" in checkpoint:
        print("✓ Loading UNet state dict")
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
    else:
        print("✓ Loading full model state dict")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model

# ============================
# 🎨 Generate Images
# ============================

@torch.no_grad()
def generate_images(model, class_id, num_samples, output_dir):
    """Generate multiple images for a single class"""
    class_dir = Path(output_dir) / f"class_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples for class {class_id}...")
    
    for sample_idx in range(num_samples):
        latents = torch.randn((1, 4, IMG_SIZE // 8, IMG_SIZE // 8)).to(DEVICE)

        scheduler = model.scheduler
        scheduler.set_timesteps(NUM_INFERENCE_STEPS)

        for t in scheduler.timesteps:
            class_tensor = torch.tensor([class_id], device=DEVICE)
            noise_pred = model.unet(latents, t, class_tensor)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = model.vae.decode(latents / 0.18215).sample
        image = (image.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        
        img_path = class_dir / f"image_{sample_idx:03d}.png"
        save_image(image, img_path)
    
    print(f"  Saved to {class_dir}")

# ============================
# 🚀 CLI Entry Point
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--class_id", type=int, default=None, help="Disease class ID (None = all classes)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per class")
    parser.add_argument("--output", type=str, default="./generated", help="Output base folder")
    parser.add_argument("--num_classes", type=int, default=13, help="Total number of classes")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(CHECKPOINT_DIR)
    model = load_model(checkpoint_path, args.num_classes)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output) / timestamp
    
    if args.class_id is not None:
        # Generate for single class
        generate_images(model, args.class_id, args.num_samples, output_base)
    else:
        # Generate for all classes
        print(f"Generating for all {args.num_classes} classes...")
        for class_id in range(args.num_classes):
            generate_images(model, class_id, args.num_samples, output_base)
    
    print(f"\n✓ All samples saved to {output_base}")