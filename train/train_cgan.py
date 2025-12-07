"""
Training script for Conditional GAN (CGAN)
"""

import sys
from pathlib import Path
import torch
import argparse
import json
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.cgan import Generator, Discriminator, d_hinge, g_hinge
from data.data_loader import create_data_loaders
from data.dataset_utils import ensure_dataset
from device_utils import get_device


@torch.no_grad()
def generate_comparison(G, class_to_idx, z_dim, device, real_images_dict, out_path):
    """Generate comparison grid: real vs fake images for each class."""
    G.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    imgs = []
    
    for c in range(len(idx_to_class)):
        # Real image
        real_img = real_images_dict[c]
        imgs.append(real_img.cpu())
        
        # Fake image
        z = torch.randn(1, z_dim, device=device)
        y = torch.full((1,), c, dtype=torch.long, device=device)
        fake = G(z, y)
        imgs.append(fake.cpu())
    
    grid = make_grid(torch.cat(imgs, 0), nrow=2, normalize=True, value_range=(-1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


@torch.no_grad()
def sample_grid(G, class_to_idx, z_dim, device, out_path):
    """Generate class-conditioned image grid."""
    G.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    imgs = []
    for c in range(len(idx_to_class)):
        z = torch.randn(8, z_dim, device=device)
        y = torch.full((8,), c, dtype=torch.long, device=device)
        imgs.append(G(z, y))
    grid = make_grid(torch.cat(imgs, 0), nrow=8, normalize=True, value_range=(-1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # Ensure dataset exists (download if needed)
    dataset_dir = Path(args.ham_csv).parent if Path(args.ham_csv).is_absolute() else Path("dataset")
    if not ensure_dataset(dataset_dir=str(dataset_dir)):
        print("Error: Could not ensure dataset is available. Please check the download.")
        sys.exit(1)

    # Use unified data loader
    train_loader, _, _, disease_classes = create_data_loaders(
        ham_metadata_path=args.ham_csv,
        ham_img_dir=args.ham_img_dir,
        bcn_metadata_path=args.bcn_csv,
        bcn_img_dir=args.bcn_img_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,  # training at specified resolution
        seed=args.seed,
    )

    num_classes = len(disease_classes)
    print(f"Detected {num_classes} classes.")

    G = Generator(args.z_dim, num_classes, base_ch=args.g_ch, img_size=args.img_size).to(device)
    D = Discriminator(num_classes, base_ch=args.d_ch).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == "cuda")

    # Collect one real image per class for comparison
    real_images_dict = {}
    for imgs, labels, _ in train_loader:
        for img, label in zip(imgs, labels):
            if label.item() not in real_images_dict:
                real_images_dict[label.item()] = img.unsqueeze(0)  # Add batch dim: (C,H,W) -> (1,C,H,W)
        if len(real_images_dict) == num_classes:
            break

    for epoch in range(args.epochs):
        G.train()
        D.train()
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # --- Train Discriminator ---
            z = torch.randn(imgs.size(0), args.z_dim, device=device)
            with torch.no_grad():
                fake = G(z, labels)
            d_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                real_logits = D(imgs, labels)
                fake_logits = D(fake, labels)
                d_loss = d_hinge(real_logits, fake_logits)
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)
            scaler.update()

            # --- Train Generator ---
            z = torch.randn(imgs.size(0), args.z_dim, device=device)
            g_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                fake = G(z, labels)
                fake_logits = D(fake, labels)
                g_loss = g_hinge(fake_logits)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })

        print(f"\n[Epoch {epoch+1}/{args.epochs}] D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

        # Generate comparison images
        comparison_path = f"{args.output_dir}/comparison_e{epoch+1:03d}.png"
        generate_comparison(G, disease_classes, args.z_dim, device, real_images_dict, comparison_path)

        # Save generated samples
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        sample_grid(G, disease_classes, args.z_dim, device, f"{args.output_dir}/cgan{args.img_size}_e{epoch+1:03d}.png")

        # Save checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "classes": disease_classes,
            "args": vars(args),
            "epoch": epoch + 1
        }, checkpoint_dir / f"cgan{args.img_size}_e{epoch+1:03d}.pt")

    # Save final model
    final_checkpoint_path = checkpoint_dir / f"cgan{args.img_size}_final.pt"
    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict(),
        "classes": disease_classes,
        "args": vars(args),
        "epoch": args.epochs
    }, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")

    # Save training summary as JSON
    summary = {
        "epochs": args.epochs,
        "final_d_loss": d_loss.item() if 'd_loss' in locals() else None,
        "final_g_loss": g_loss.item() if 'g_loss' in locals() else None,
        "num_classes": num_classes,
        "disease_classes": disease_classes,
        "model_config": {
            "z_dim": args.z_dim,
            "g_ch": args.g_ch,
            "d_ch": args.d_ch,
            "lr": args.lr,
            "amp": args.amp
        }
    }
    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Training summary saved: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv")
    p.add_argument("--ham_img_dir", default="dataset/HAM10000_images")
    p.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv")
    p.add_argument("--bcn_img_dir", default="dataset/ISIC_images")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--img_size", type=int, default=64, help="Image size for training (must be power of 2)")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--g_ch", type=int, default=64)
    p.add_argument("--d_ch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cgan", help="Checkpoint directory")
    p.add_argument("--output_dir", type=str, default="output/cgan", help="Output directory for generated samples")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

