"""
Prod CGAN Image Generator
Generates individual synthetic images per class from a trained CGAN
Organized by model name ("cgan") and timestamp (yyyymmdd_hhmmss)
"""

import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
from datetime import datetime
import sys
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.cgan import Generator
from device_utils import get_device


def load_model(checkpoint_path, device):
    """Load CGAN generator from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Classes dictionary may be {id: name} or just numeric IDs
    disease_classes = checkpoint.get("classes", {})
    args_from_checkpoint = checkpoint.get("args", {})

    img_size = args_from_checkpoint.get("img_size", 64)
    g_ch = args_from_checkpoint.get("g_ch", 64)
    z_dim = args_from_checkpoint.get("z_dim", 128)

    num_classes = len(disease_classes) if disease_classes else checkpoint.get("num_classes", 0)

    G = Generator(z_dim, num_classes, base_ch=g_ch, img_size=img_size).to(device)
    G.load_state_dict(checkpoint["G"])
    G.eval()

    print(f"Loaded CGAN with {num_classes} classes")
    return G, disease_classes, z_dim, num_classes


@torch.no_grad()
def generate_images(G, class_id, class_name, num_images, z_dim, output_dir, device):
    """Generate multiple images for a specific class"""
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Class {class_id}", leave=False):
        z = torch.randn(1, z_dim, device=device)
        y = torch.tensor([class_id], dtype=torch.long, device=device)
        fake = G(z, y)

        # Normalize to [0,1]
        image = (fake.clamp(-1, 1) + 1) / 2
        output_path = output_dir / f"img_{i+1:03d}.png"
        save_image(image, output_path)


def main():
    parser = argparse.ArgumentParser(description="CGAN Image Generator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Images per class")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/mps/cuda/cpu")

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

    # Load model
    G, disease_classes, z_dim, num_classes = load_model(Path(args.checkpoint), device)

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / "cgan" / timestamp
    print(f"\nGenerating {args.num_images} images per class")
    print(f"Output directory: {output_base}\n")

    # Generate for each class
    if disease_classes:
        # Reverse the dictionary: {name → id} → {id → name}
        idx_to_class = {v: k for k, v in disease_classes.items()}
        for class_id, class_name in idx_to_class.items():
            class_dir = output_base / str(class_name)
            generate_images(G, class_id, class_name, args.num_images, z_dim, class_dir, device)
    else:
        # Fallback to class_0, class_1, ...
        for class_id in range(num_classes):
            class_name = f"class_{class_id}"
            class_dir = output_base / class_name
            generate_images(G, class_id, class_name, args.num_images, z_dim, class_dir, device)


    print(f"\n✅ Generated {args.num_images * num_classes} images")
    print(f"📁 Saved to: {output_base}")


if __name__ == "__main__":
    main()
