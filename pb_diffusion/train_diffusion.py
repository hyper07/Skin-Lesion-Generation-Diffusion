import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import DiffusionModel
from data_loader import create_data_loaders
from datetime import datetime

# ============================
# 🔧 Configuration
# ============================

BATCH_SIZE = 64
IMG_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4 * (BATCH_SIZE / 4) ** 0.5
LEARNING_RATE = min(LEARNING_RATE, 3e-5)
TOP_N_CLASSES = None
LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
USE_LORA = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Learning Rate: {LEARNING_RATE}")

DATA_DIR = os.getenv('DATA_DIR')

# Define paths
ham_metadata = os.path.join(DATA_DIR, "dataverse_files/HAM10000_metadata")
ham_part1 = os.path.join(DATA_DIR, "dataverse_files/HAM10000_images_part_1")
ham_part2 = os.path.join(DATA_DIR, "dataverse_files/HAM10000_images_part_2")
bcn_metadata = os.path.join(DATA_DIR, "bcn20000_metadata_2025-10-21.csv")
bcn_images = os.path.join(DATA_DIR, "ISIC-images")

# ============================
# 📦 Load Data
# ============================

train_loader, val_loader, _, disease_classes = create_data_loaders(
    ham_metadata_path=ham_metadata,
    ham_img_part1=ham_part1,
    ham_img_part2=ham_part2,
    bcn_metadata_path=bcn_metadata,
    bcn_img_dir=bcn_images,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    top_n_classes=TOP_N_CLASSES
)

num_classes = len(disease_classes)

# ============================
# 🧠 Initialize Model
# ============================

model = DiffusionModel(num_classes=num_classes, use_lora=USE_LORA).to(DEVICE)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resume from latest checkpoint
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
if checkpoint_files:
    checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
    print(f"✓ Resuming from checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

    if "unet_state_dict" in checkpoint:
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    global_step = checkpoint.get("step", 0)
    start_epoch = checkpoint.get("epoch", 0)
else:
    print("⚠️ No checkpoint found — starting fresh.")
    global_step = 0
    start_epoch = 0

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

if checkpoint_files and "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

criterion = nn.MSELoss()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"run_{timestamp}_bs{BATCH_SIZE}_lora{USE_LORA}"
writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

# ============================
# 🚀 Training Loop
# ============================

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in loop:
        images, class_ids, _ = batch
        images = images.to(DEVICE)
        class_ids = class_ids.to(DEVICE)

        optimizer.zero_grad()
        predicted_noise, true_noise = model(images, class_ids)

        if torch.isnan(predicted_noise).any():
            print("⚠️ UNet output contains NaNs — skipping step")
            continue

        loss = criterion(predicted_noise, true_noise)
        if torch.isnan(loss):
            print("⚠️ Loss is NaN — skipping step")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
        loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        # # Save checkpoint every 200 steps
        # if global_step % 200 == 0 and global_step > 0:
        #     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"diffusion_epoch_{epoch+1}_step_{global_step}.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'step': global_step,
        #         'unet_state_dict': model.unet.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': loss.item()
        #     }, checkpoint_path)
        #     print(f"\n✓ Checkpoint saved: {checkpoint_path}")

        global_step += 1

    # ============================
    # 🧪 Validation
    # ============================

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images, class_ids, _ = batch
            images = images.to(DEVICE)
            class_ids = class_ids.to(DEVICE)
            predicted_noise, true_noise = model(images, class_ids)
            loss = criterion(predicted_noise, true_noise)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    print(f"\nValidation Loss: {avg_val_loss:.4f}")

    # ============================
    # 💾 End of Epoch Checkpoint
    # ============================

    final_ckpt = os.path.join(CHECKPOINT_DIR, f"diffusion_epoch_{epoch+1}_final.pt")
    torch.save({
        'epoch': epoch + 1,
        'step': global_step,
        'unet_state_dict': model.unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_val_loss
    }, final_ckpt)
    print(f"✓ Epoch checkpoint saved: {final_ckpt}")

writer.close()
print("✓ Training complete.")