import os
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

#Device selection

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#Dataset + stratified split

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


def build_stratified_splits(
    root_dir,
    nevus_name="nevus",
    nevus_cap=400,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    """
    - Uses all class subfolders under root_dir.
    - Caps Nevus at nevus_cap.
    - Uses ALL samples for other classes.
    - Splits per class: 70% train, 15% val, 15% test.
    - Returns: (train_items, val_items, test_items, class_to_idx)
    """
    random.seed(seed)
    root = Path(root_dir)

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    class_names = sorted(d.name for d in class_dirs)

    if not class_names:
        raise ValueError(f"No class folders found in {root_dir}")

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_items, val_items, test_items = [], [], []

    for cls in class_names:
        cls_idx = class_to_idx[cls]
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend((root / cls).glob(ext))

        img_paths = list(img_paths)
        if not img_paths:
            print(f"[WARN] No images for class '{cls}', skipping.")
            continue

        # Cap Nevus
        if cls.lower() == nevus_name.lower() and len(img_paths) > nevus_cap:
            img_paths = random.sample(img_paths, nevus_cap)

        random.shuffle(img_paths)
        n = len(img_paths)

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val

        cls_train = [(str(p), cls_idx) for p in img_paths[:n_train]]
        cls_val = [(str(p), cls_idx) for p in img_paths[n_train:n_train + n_val]]
        cls_test = [(str(p), cls_idx) for p in img_paths[n_train + n_val:]]

        train_items.extend(cls_train)
        val_items.extend(cls_val)
        test_items.extend(cls_test)

        print(
            f"[{cls}] idx={cls_idx} total_used={n} "
            f"train={len(cls_train)} val={len(cls_val)} test={len(cls_test)}"
        )

    print(
        f"[SPLIT DONE] train={len(train_items)} "
        f"val={len(val_items)} test={len(test_items)} "
        f"classes={len(class_to_idx)}"
    )

    return train_items, val_items, test_items, class_to_idx


#Pre-built conditional GAN

class ConditionalGenerator(nn.Module):
    """
    Pre-built conditional generator (StyleGAN-ish but simplified).

    - Input: latent z (B, z_dim), labels (B)
    - Output: (B, 3, H, W) in [-1, 1]
    - Has label embeddings; designed for fine-tuning.
    """

    def __init__(self, z_dim, num_classes, img_size=256, base_channels=256):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, z_dim)

        self.fc = nn.Linear(z_dim, base_channels * 8 * 8)

        #Upsample from 8x8 to img_size
        num_upsamples = int(round(math.log2(img_size // 8)))
        channels = base_channels
        blocks = []
        for _ in range(num_upsamples):
            blocks += [
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
                nn.BatchNorm2d(channels // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels //= 2

        self.upsample_blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = z + label_vec
        x = self.fc(x).view(x.size(0), -1, 8, 8)
        x = self.upsample_blocks(x)
        x = self.to_rgb(x)
        return x


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator:
    - Embeds label -> spatial mask.
    - Concatenates mask with image.
    """

    def __init__(self, num_classes, img_size=256, base_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        in_channels = 4  # 3 (RGB) + 1 (label mask)
        layers = []
        layers += block(in_channels, base_channels, normalize=False)
        layers += block(base_channels, base_channels * 2)
        layers += block(base_channels * 2, base_channels * 4)
        layers += block(base_channels * 4, base_channels * 8)

        self.conv = nn.Sequential(*layers)

        down_factor = 2 ** 4
        final_size = img_size // down_factor
        self.adv = nn.Conv2d(base_channels * 8, 1, final_size, 1, 0)

    def forward(self, img, labels):
        mask = self.label_emb(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, mask], dim=1)
        feat = self.conv(x)
        out = self.adv(feat).view(-1)
        return out


#Sample generation

@torch.no_grad()
def generate_samples_for_all_classes(
    G,
    num_classes,
    device,
    z_dim=128,
    samples_per_class=4,
    out_path="samples/generated_samples.png",
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

#Training loop
def train(
    data_root,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    image_size=64,
    num_timesteps=1000,  
    save_interval=10,
    z_dim=128,
    pretrained_g=None,
    pretrained_d=None,
    out_dir="checkpoints",
    samples_dir="samples",
):
    device = get_device()
    print(f"[DEVICE] {device}")

    #Stratified split with Nevus cap
    train_items, _, _, class_to_idx = build_stratified_splits(data_root)
    num_classes = len(class_to_idx)
    print(f"[INFO] num_classes={num_classes}, class_to_idx={class_to_idx}")

    #Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = SkinLesionDataset(train_items, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
        drop_last=True,
    )

    #Models
    G = ConditionalGenerator(z_dim=z_dim, num_classes=num_classes, img_size=image_size).to(device)
    D = ConditionalDiscriminator(num_classes=num_classes, img_size=image_size).to(device)

    #loading pre-built weights for fine-tuning
    if pretrained_g and os.path.exists(pretrained_g):
        print(f"[LOAD] Pretrained G from {pretrained_g}")
        G.load_state_dict(torch.load(pretrained_g, map_location=device), strict=False)

    if pretrained_d and os.path.exists(pretrained_d):
        print(f"[LOAD] Pretrained D from {pretrained_d}")
        D.load_state_dict(torch.load(pretrained_d, map_location=device), strict=False)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))
    hinge = nn.ReLU()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    step = 0
    for epoch in range(1, epochs + 1):
        G.train()
        D.train()

        for real_imgs, labels in train_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bsz = real_imgs.size(0)

            #Train D
            z = torch.randn(bsz, z_dim, device=device)
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

            #Training G
            z = torch.randn(bsz, z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (bsz,), device=device)
            gen_imgs = G(z, gen_labels)
            D_gen = D(gen_imgs, gen_labels)
            loss_G = -D_gen.mean()

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            step += 1

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"loss_D={loss_D.item():.4f} | loss_G={loss_G.item():.4f}"
        )

        #Checkpoints + samples
        if epoch % save_interval == 0 or epoch == epochs:
            g_path = os.path.join(out_dir, f"G_epoch_{epoch}.pt")
            d_path = os.path.join(out_dir, f"D_epoch_{epoch}.pt")
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"[CKPT] Saved {g_path} and {d_path}")

            sample_path = os.path.join(samples_dir, f"generated_samples_epoch_{epoch}.png")
            generate_samples_for_all_classes(
                G,
                num_classes=num_classes,
                device=device,
                z_dim=z_dim,
                samples_per_class=4,
                out_path=sample_path,
            )


#CLI

def parse_args():
    p = argparse.ArgumentParser(description="Pre-built conditional GAN + fine-tuning")
    p.add_argument("--data-root", type=str, required=True,
                   help="Root folder with class subfolders (e.g. 'dataset/GenAI Project').")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-timesteps", type=int, default=1000,
                   help="Unused here (for compatibility with diffusion configs).")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--pretrained-g", type=str, default=None,
                   help="Path to pretrained G .pt for fine-tuning.")
    p.add_argument("--pretrained-d", type=str, default=None,
                   help="Path to pretrained D .pt for fine-tuning.")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--samples-dir", type=str, default="samples")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_timesteps=args.num_timesteps,
        save_interval=args.save_interval,
        z_dim=args.z_dim,
        pretrained_g=args.pretrained_g,
        pretrained_d=args.pretrained_d,
        out_dir=args.out_dir,
        samples_dir=args.samples_dir,
    )
