# CGan/main.py
import sys
from pathlib import Path
import torch
import argparse
from torchvision.utils import make_grid, save_image

# --- Add project root to sys.path so Python can find pb_diffusion and other modules ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Import your model and existing repo data loader ---
from CGan.CGan import Generator, Discriminator, d_hinge, g_hinge
from pb_diffusion.data_loader import create_data_loaders


# ---------------- DEVICE ----------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- SAMPLING ----------------
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


# ---------------- TRAIN LOOP ----------------
def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # Use your repo’s data loader
    train_loader, _, _, disease_classes = create_data_loaders(
        ham_metadata_path=args.ham_csv,
        ham_img_part1=args.ham_img_part1,
        ham_img_part2=args.ham_img_part2,
        bcn_metadata_path=args.bcn_csv,
        bcn_img_dir=args.bcn_img_dir,
        batch_size=args.batch_size,
        img_size=64,  # training at 64×64
        seed=args.seed,
    )

    num_classes = len(disease_classes)
    print(f"Detected {num_classes} classes.")

    G = Generator(args.z_dim, num_classes, base_ch=args.g_ch, img_size=64).to(device)
    D = Discriminator(num_classes, base_ch=args.d_ch).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    for epoch in range(args.epochs):
        G.train(); D.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # --- Train Discriminator ---
            z = torch.randn(imgs.size(0), args.z_dim, device=device)
            with torch.no_grad():
                fake = G(z, labels)
            d_opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                real_logits = D(imgs, labels)
                fake_logits = D(fake, labels)
                d_loss = d_hinge(real_logits, fake_logits)
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)
            scaler.update()

            # --- Train Generator ---
            z = torch.randn(imgs.size(0), args.z_dim, device=device)
            g_opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                fake = G(z, labels)
                fake_logits = D(fake, labels)
                g_loss = g_hinge(fake_logits)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

        print(f"[Epoch {epoch+1}] D: {d_loss.item():.3f} | G: {g_loss.item():.3f}")

        # Save generated samples
        sample_grid(G, disease_classes, args.z_dim, device, f"samples/cgan64_e{epoch+1:03d}.png")

        # Save checkpoint
        Path("runs").mkdir(exist_ok=True)
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "classes": disease_classes,
            "args": vars(args),
            "epoch": epoch + 1
        }, f"runs/cgan64_e{epoch+1:03d}.pt")


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ham_csv", default="HAM10000_metadata.csv")
    p.add_argument("--ham_img_part1", default="dataverse_files/HAM10000_images_part_1")
    p.add_argument("--ham_img_part2", default="dataverse_files/HAM10000_images_part_2")
    p.add_argument("--bcn_csv", default="ISIC_metadata.csv")
    p.add_argument("--bcn_img_dir", default="ISIC_images")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--g_ch", type=int, default=64)
    p.add_argument("--d_ch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
