"""
Generation script for Pre-built Conditional GAN
"""

import sys
from pathlib import Path
import torch
import argparse
from torchvision.utils import make_grid, save_image
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.prebuilt_gan import ConditionalGenerator
from device_utils import get_device


@torch.no_grad()
def generate_samples_for_all_classes(
    G,
    num_classes,
    device,
    z_dim=128,
    samples_per_class=4,
    out_path="output/prebuilt_gan/prebuilt_gan_generated.png",
    class_id=None
):
    G.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if class_id is not None:
        # Generate individual images for specific class
        print(f"Generating {samples_per_class} individual images for class {class_id}")
        
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            z = torch.randn(1, z_dim, device=device)
            labels = torch.full((1,), class_id, dtype=torch.long, device=device)
            fake = G(z, labels)[0]  # Remove batch dimension
            
            # Denormalize from [-1, 1] to [0, 1]
            fake = (fake + 1) / 2
            fake = torch.clamp(fake, 0, 1)
            
            # Convert to PIL Image
            from PIL import Image as PILImage
            fake_np = fake.cpu().permute(1, 2, 0).numpy()
            fake_pil = PILImage.fromarray((fake_np * 255).astype('uint8'))
            
            # Save individual image
            img_path = Path(out_path).parent / f"{i:03d}.png"
            fake_pil.save(img_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{samples_per_class} images")
        
        print(f"[SAMPLES] Saved {samples_per_class} individual samples for class {class_id} to {Path(out_path).parent}")
        return

    # Original behavior: generate for all classes
    imgs = []
    for c in range(num_classes):
        z = torch.randn(samples_per_class, z_dim, device=device)
        labels = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
        fake = G(z, labels)
        imgs.append(fake.cpu())

    imgs = torch.cat(imgs, dim=0)
    grid = make_grid(
        imgs,
        nrow=samples_per_class,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, out_path)
    print(f"[SAMPLES] Saved generated samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Prebuilt Conditional GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint file (default: checkpoints/prebuilt_gan/)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--samples_per_class", type=int, default=4, help="Number of samples per class")
    parser.add_argument("--class_id", type=int, default=None, help="Class ID to generate (None for all classes)")
    parser.add_argument("--output", type=str, default="output/prebuilt_gan/prebuilt_gan_generated.png", help="Output path")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    G = ConditionalGenerator(
        z_dim=args.z_dim, 
        num_classes=args.num_classes, 
        img_size=args.img_size
    ).to(device)
    
    # Load checkpoint
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded generator from {args.checkpoint}")
    
    # Generate samples
    generate_samples_for_all_classes(
        G,
        num_classes=args.num_classes,
        device=device,
        z_dim=args.z_dim,
        samples_per_class=args.samples_per_class,
        out_path=args.output,
        class_id=args.class_id
    )


if __name__ == "__main__":
    main()

