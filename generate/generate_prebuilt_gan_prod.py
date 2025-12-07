"""
Prod Prebuilt GAN Image Generator
Generates individual synthetic images per class from a trained Prebuilt Conditional GAN
Organized by model name ("prebuilt_gan") and timestamp (yyyymmdd_HHMMSS)
"""

import sys
from pathlib import Path
import os
import argparse
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.prebuilt_gan import ConditionalGenerator
from device_utils import get_device


@torch.no_grad()
def generate_images(G, class_id, class_name, num_images, z_dim, output_dir, device):
    """Generate multiple images for a specific class"""
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Class {class_id}", leave=False):
        z = torch.randn(1, z_dim, device=device)
        y = torch.tensor([class_id], dtype=torch.long, device=device)
        fake = G(z, y)

        image = (fake.clamp(-1, 1) + 1) / 2
        output_path = output_dir / f"img_{i+1:03d}.png"
        save_image(image, output_path)


def main():
    parser = argparse.ArgumentParser(description="Prod Prebuilt GAN Image Generator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to generator checkpoint file")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="Latent dimension (overridden if checkpoint has it)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image size (overridden if checkpoint has it)")
    parser.add_argument("--base_channels", type=int, default=256,
                        help="Base channel size (overridden if checkpoint has it)")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Images per class")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/mps/cuda/cpu")
    parser.add_argument("--classes", type=str, default=None,
                        help="Optional path to class mapping file (JSON or TXT)")

    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint and extract model config
    checkpoint = torch.load(args.checkpoint, map_location=device)
    args_from_ckpt = checkpoint.get("args", {})

    z_dim = args_from_ckpt.get("z_dim", args.z_dim)
    img_size = args_from_ckpt.get("img_size", args.img_size)
    base_channels = args_from_ckpt.get("base_channels", args.base_channels)

    print(f"Loaded checkpoint config: z_dim={z_dim}, img_size={img_size}, base_channels={base_channels}")

    # Instantiate model
    G = ConditionalGenerator(
        z_dim=z_dim,
        num_classes=args.num_classes,
        img_size=img_size,
        base_channels=base_channels
    ).to(device)

    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()
    print(f"✅ Loaded generator from {args.checkpoint}")

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / "prebuilt_gan" / timestamp
    print(f"\nGenerating {args.num_images} images per class")
    print(f"Output directory: {output_base}\n")

    # Load class names if provided
    if args.classes:
        import json
        class_path = Path(args.classes)
        if class_path.suffix == ".json":
            with open(class_path) as f:
                class_map = json.load(f)
            class_map = {int(k): v for k, v in class_map.items()}
        else:
            with open(class_path) as f:
                lines = f.readlines()
                class_map = {i: line.strip() for i, line in enumerate(lines)}
    else:
        class_map = {i: f"class_{i}" for i in range(args.num_classes)}

    # Generate for each class
    for class_id in range(args.num_classes):
        class_name = class_map.get(class_id, f"class_{class_id}")
        class_dir = output_base / class_name
        generate_images(G, class_id, class_name, args.num_images, z_dim, class_dir, device)

    print(f"\n🎉 Done! Generated {args.num_images * args.num_classes} images")
    print(f"📁 Saved to: {output_base}")


if __name__ == "__main__":
    main()
