# Pre-Built Conditional GAN (64×64 Skin Lesion Generator)

This model implements a **Pre-Built Conditional GAN** for generating synthetic 64×64 skin lesion images.  
It supports **fine-tuning** (transfer learning) on an existing model and **automatic per-class image sampling**.  
The GAN architecture is inspired by StyleGAN2-ADA but simplified for faster training and compatibility with CPU / MPS / CUDA.

# Training the model
python pre-builtGAN_finetune/train_prebuilt_gan.py \
  --data-root "dataset/GenAI Project" \
  --epochs 25 \
  --batch-size 16 \
  --lr 1e-4 \
  --image-size 64 \
  --save-interval 10

# Fine tuning from Saved Weights
python pre-builtGAN_finetune/train_prebuilt_gan.py \
  --data-root "dataset/GenAI Project" \
  --epochs 50 \
  --batch-size 16 \
  --lr 5e-5 \
  --image-size 64 \
  --pretrained-g checkpoints/G_epoch_25.pt \
  --pretrained-d checkpoints/D_epoch_25.pt
