import os
import sys

# Ensure project root (where `models/` lives) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fix for macOS mutex error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
torch.set_num_threads(1)

from torchvision.utils import save_image
from models.prebuilt_diffusion import DiffusionModel
from diffusers import DDPMScheduler
from datetime import datetime
import argparse
from pathlib import Path

IMG_SIZE = 64
NUM_INFERENCE_STEPS = 100

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f" Using device: {DEVICE}")

CHECKPOINT_DIR = "./checkpoints/prebuilt_diffusion"


def find_latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError("No checkpoint files found.")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    print(f" Found latest checkpoint: {files[0]}")
    return os.path.join(checkpoint_dir, files[0])


def load_model(checkpoint_path, num_classes):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    checkpoint_classes = None
    if "unet_state_dict" in checkpoint:
        if "class_embed.weight" in checkpoint["unet_state_dict"]:
            checkpoint_classes = checkpoint["unet_state_dict"]["class_embed.weight"].shape[0]
    elif "model_state_dict" in checkpoint:
        if "unet.class_embed.weight" in checkpoint["model_state_dict"]:
            checkpoint_classes = checkpoint["model_state_dict"]["unet.class_embed.weight"].shape[0]

    if checkpoint_classes is not None and checkpoint_classes != num_classes:
        print(f" Warning: Checkpoint has {checkpoint_classes} classes, but requested {num_classes}.")
        print(f" Adjusting model to match checkpoint: {checkpoint_classes} classes.")
        num_classes = checkpoint_classes

    model = DiffusionModel(num_classes=num_classes, use_lora=True).to(DEVICE)

    if "unet_state_dict" in checkpoint:
        print(" Loading UNet state dict")
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
    else:
        print(" Loading full model state dict")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


@torch.no_grad()
def generate_images(model, class_id, num_samples, output_dir):
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
        image = (image.clamp(-1, 1) + 1) / 2

        img_path = class_dir / f"image_{sample_idx:03d}.png"
        save_image(image, img_path)

    print(f"  Saved to {class_dir}")


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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output) / timestamp

    if args.class_id is not None:
        generate_images(model, args.class_id, args.num_samples, output_base)
    else:
        print(f"Generating for all {args.num_classes} classes...")
        for class_id in range(args.num_classes):
            generate_images(model, class_id, args.num_samples, output_base)

    print(f"\n All samples saved to {output_base}")
