"""
Debug script to test diffusers loading on macOS
"""
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

try:
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
    print("✓ imported diffusers ok")
except Exception as e:
    print("✗ failed to import diffusers:", e)
    exit(1)

model_id = "CompVis/stable-diffusion-v1-4"

try:
    print("loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    print("✓ VAE loaded")
except Exception as e:
    print("✗ VAE loading failed:", e)
    exit(1)

try:
    print("loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    print("✓ UNet loaded")
except Exception as e:
    print("✗ UNet loading failed:", e)
    exit(1)

try:
    print("loading scheduler...")
    sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    print("✓ Scheduler loaded")
except Exception as e:
    print("✗ Scheduler loading failed:", e)
    exit(1)

print("\n✓✓✓ ALL COMPONENTS LOADED SUCCESSFULLY ✓✓✓")
