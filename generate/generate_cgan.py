"""
Generation script for Conditional GAN (CGAN)
"""

import sys
from pathlib import Path
import torch
import argparse
from torchvision.utils import make_grid, save_image
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.cgan import Generator
from device_utils import get_device


@torch.no_grad()
def generate_samples(G, class_to_idx, z_dim, device, num_samples_per_class=8, out_path="output/cgan/cgan_generated.png", 
                    real_images_dict=None, class_id=None):
    """Generate class-conditioned image grid with optional real images and labels."""
    G.eval()
    
    if class_id is not None:
        # Generate individual images for specific class
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        if class_id not in idx_to_class:
            raise ValueError(f"Class ID {class_id} not found in class_to_idx")
        
        print(f"Generating {num_samples_per_class} individual images for class '{idx_to_class[class_id]}'")
        
        # Create output directory
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples_per_class):
            z = torch.randn(1, z_dim, device=device)
            y = torch.full((1,), class_id, dtype=torch.long, device=device)
            fake = G(z, y)[0]  # Remove batch dimension
            
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
                print(f"  Generated {i + 1}/{num_samples_per_class} images")
        
        print(f"Generated {num_samples_per_class} samples for class '{idx_to_class[class_id]}'")
        print(f"Saved to {Path(out_path).parent}")
        return
    
    # Original behavior: generate for all classes
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    imgs = []
    
    for c in range(len(idx_to_class)):
        # Add real image if available (first column)
        if real_images_dict and c in real_images_dict:
            imgs.append(real_images_dict[c].cpu())
        
        # Add generated images
        z = torch.randn(num_samples_per_class, z_dim, device=device)
        y = torch.full((num_samples_per_class,), c, dtype=torch.long, device=device)
        fake = G(z, y)
        imgs.extend([img.cpu() for img in fake])
    
    # Determine number of columns per row
    cols_per_row = num_samples_per_class + (1 if real_images_dict and len(real_images_dict) > 0 else 0)
    
    grid = make_grid(torch.cat(imgs, 0), nrow=cols_per_row, normalize=True, value_range=(-1, 1))
    
    # Save the grid
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    
    # Add labels using PIL if available
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(out_path)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add class labels for each row
        img_height = img.height
        row_height = img_height // len(idx_to_class)
        for i, (idx, name) in enumerate(idx_to_class.items()):
            label = f"Class {idx}: {name}"
            y_pos = i * row_height + 10
            # Draw black background for text
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([5, y_pos-5, 5 + text_width + 10, y_pos + text_height + 5], fill="black")
            draw.text((10, y_pos), label, fill="white", font=font)
        
        img.save(out_path)
        print(f"Generated samples with labels saved to {out_path}")
    except ImportError:
        print(f"Generated samples saved to {out_path} (PIL not available for labels)")
    except Exception as e:
        print(f"Generated samples saved to {out_path} (Could not add labels: {e})")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Conditional GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples per class")
    parser.add_argument("--class_id", type=int, default=None, help="Class ID to generate (None for all classes)")
    parser.add_argument("--output", type=str, default="output/cgan/cgan_generated.png", help="Output path")
    parser.add_argument("--include_real", action="store_true", help="Include real images for comparison")
    parser.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv", help="HAM metadata CSV")
    parser.add_argument("--ham_img_dir", default="dataset/HAM10000_images", help="HAM images directory")
    parser.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv", help="BCN metadata CSV")
    parser.add_argument("--bcn_img_dir", default="dataset/ISIC_images", help="BCN images directory")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    disease_classes = checkpoint["classes"]
    args_from_checkpoint = checkpoint.get("args", {})
    
    # Get parameters from checkpoint (with defaults)
    img_size = args_from_checkpoint.get("img_size", 64)
    g_ch = args_from_checkpoint.get("g_ch", 64)
    z_dim = args_from_checkpoint.get("z_dim", 128)
    
    num_classes = len(disease_classes)
    
    print(f"Loaded checkpoint with {num_classes} classes")
    print(f"Classes: {list(disease_classes.keys())}")
    print(f"Image size: {img_size}x{img_size}, Generator channels: {g_ch}, Z dim: {z_dim}")
    
    # Create model with parameters from checkpoint
    G = Generator(z_dim, num_classes, base_ch=g_ch, img_size=img_size).to(device)
    G.load_state_dict(checkpoint["G"])
    
    # Load real images if requested
    real_images_dict = None
    if args.include_real:
        try:
            dataset_dir = Path(args.ham_csv).parent
            if not ensure_dataset(dataset_dir=str(dataset_dir)):
                print("Warning: Could not ensure dataset, skipping real images")
            else:
                # Load a small batch to get one image per class
                train_loader, _, _, _ = create_data_loaders(
                    ham_metadata_path=args.ham_csv,
                    ham_img_dir=[args.ham_img_dir],
                    bcn_metadata_path=args.bcn_csv,
                    bcn_img_dir=args.bcn_img_dir,
                    batch_size=1,
                    img_size=img_size,
                    seed=42
                )
                real_images_dict = {}
                for imgs, labels, _ in train_loader:
                    for img, label in zip(imgs, labels):
                        if label.item() not in real_images_dict:
                            real_images_dict[label.item()] = img.unsqueeze(0)
                    if len(real_images_dict) == num_classes:
                        break
                print(f"Loaded {len(real_images_dict)} real images for comparison")
        except Exception as e:
            print(f"Could not load real images: {e}")
    
    # Generate samples
    generate_samples(G, disease_classes, z_dim, device, args.num_samples, args.output, real_images_dict, args.class_id)


if __name__ == "__main__":
    main()

