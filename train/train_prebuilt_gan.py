"""
Training script for Pre-built Conditional GAN
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import argparse
import random
import math
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.prebuilt_gan import ConditionalGenerator, ConditionalDiscriminator
from data.data_loader import create_data_loaders
from data.dataset_utils import ensure_dataset
from device_utils import get_device


@torch.no_grad()
def generate_comparison_prebuilt(G, num_classes, device, real_images_dict, z_dim=128, out_path="output/comparison.png"):
    """Generate comparison grid: real vs fake images for each class."""
    G.eval()
    imgs = []
    
    for c in range(num_classes):
        # Real image
        real_img = real_images_dict[c]
        imgs.append(real_img.cpu())
        
        # Fake image
        z = torch.randn(1, z_dim, device=device)
        labels = torch.full((1,), c, dtype=torch.long, device=device)
        fake = G(z, labels)
        imgs.append(fake.cpu())
    
    grid = make_grid(torch.cat(imgs, 0), nrow=2, normalize=True, value_range=(-1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


class SkinLesionDataset(Dataset):
    def __init__(self, items, transform=None):
        """
        items: list of (image_path, class_idx)
        """
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


@torch.no_grad()
def generate_samples_for_all_classes(
    G,
    num_classes,
    device,
    z_dim=128,
    samples_per_class=4,
    out_path="output/prebuilt_gan/generated_samples.png",
):
    G.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
    G.train()


def train(args):
    device = get_device()
    print(f"[DEVICE] {device}")

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
        img_size=args.img_size,
        seed=args.seed,
    )

    num_classes = len(disease_classes)
    print(f"[INFO] num_classes={num_classes}, disease_classes={disease_classes}")

    # Models
    G = ConditionalGenerator(z_dim=args.z_dim, num_classes=num_classes, img_size=args.img_size).to(device)
    D = ConditionalDiscriminator(num_classes=num_classes, img_size=args.img_size).to(device)

    # Loading pre-built weights for fine-tuning
    if args.pretrained_g and os.path.exists(args.pretrained_g):
        print(f"[LOAD] Pretrained G from {args.pretrained_g}")
        G.load_state_dict(torch.load(args.pretrained_g, map_location=device), strict=False)

    if args.pretrained_d and os.path.exists(args.pretrained_d):
        print(f"[LOAD] Pretrained D from {args.pretrained_d}")
        D.load_state_dict(torch.load(args.pretrained_d, map_location=device), strict=False)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99))
    hinge = nn.ReLU()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect one real image per class for comparison
    real_images_dict = {}
    for real_imgs, labels, _ in train_loader:
        for img, label in zip(real_imgs, labels):
            if label.item() not in real_images_dict:
                real_images_dict[label.item()] = img.unsqueeze(0)  # Add batch dim: (C,H,W) -> (1,C,H,W)
        if len(real_images_dict) == num_classes:
            break

    step = 0
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_imgs, labels, _ in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bsz = real_imgs.size(0)

            # Train D
            z = torch.randn(bsz, args.z_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (bsz,), device=device)
            fake_imgs = G(z, fake_labels).detach()

            D_real = D(real_imgs, labels)
            D_fake = D(fake_imgs, fake_labels)

            loss_D_real = hinge(1.0 - D_real).mean()
            loss_D_fake = hinge(1.0 + D_fake).mean()
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # Training G
            z = torch.randn(bsz, args.z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (bsz,), device=device)
            gen_imgs = G(z, gen_labels)
            D_gen = D(gen_imgs, gen_labels)
            loss_G = -D_gen.mean()

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })

        print(
            f"\n[Epoch {epoch}/{args.epochs}] "
            f"loss_D={loss_D.item():.4f} | loss_G={loss_G.item():.4f}"
        )

        # Generate comparison images
        comparison_path = os.path.join(args.output_dir, f"comparison_epoch_{epoch}.png")
        generate_comparison_prebuilt(G, num_classes, device, real_images_dict, z_dim=args.z_dim, out_path=comparison_path)

        # Checkpoints + samples
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            g_path = os.path.join(args.checkpoint_dir, f"G_epoch_{epoch}.pt")
            d_path = os.path.join(args.checkpoint_dir, f"D_epoch_{epoch}.pt")
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"[CKPT] Saved {g_path} and {d_path}")

            sample_path = os.path.join(args.output_dir, f"generated_samples_epoch_{epoch}.png")
            generate_samples_for_all_classes(
                G,
                num_classes=num_classes,
                device=device,
                z_dim=args.z_dim,
                samples_per_class=4,
                out_path=sample_path,
            )

    # Save final model
    final_g_path = os.path.join(args.checkpoint_dir, "G_final.pt")
    final_d_path = os.path.join(args.checkpoint_dir, "D_final.pt")
    torch.save(G.state_dict(), final_g_path)
    torch.save(D.state_dict(), final_d_path)
    print(f"[FINAL] Saved {final_g_path} and {final_d_path}")

    # Save training summary as JSON
    summary = {
        "epochs": args.epochs,
        "final_d_loss": loss_D.item() if 'loss_D' in locals() else None,
        "final_g_loss": loss_G.item() if 'loss_G' in locals() else None,
        "num_classes": num_classes,
        "disease_classes": disease_classes,
        "model_config": {
            "z_dim": args.z_dim,
            "image_size": args.img_size,
            "lr": args.lr,
            "batch_size": args.batch_size
        }
    }
    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[SUMMARY] Saved {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Prebuilt Conditional GAN training")
    p.add_argument("--ham_csv", default="dataset/HAM10000_metadata.csv")
    p.add_argument("--ham_img_dir", default="dataset/HAM10000_images")
    p.add_argument("--bcn_csv", default="dataset/ISIC_metadata.csv")
    p.add_argument("--bcn_img_dir", default="dataset/ISIC_images")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--pretrained_g", type=str, default=None,
                   help="Path to pretrained G .pt for fine-tuning.")
    p.add_argument("--pretrained_d", type=str, default=None,
                   help="Path to pretrained D .pt for fine-tuning.")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/prebuilt_gan", help="Checkpoint directory")
    p.add_argument("--output_dir", type=str, default="output/prebuilt_gan", help="Output directory for generated samples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

