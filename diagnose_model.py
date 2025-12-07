"""
Diagnostic script to check model weights and generate samples
"""

import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.conditional_diffusion import ConditionalDiffusionModel
from device_utils import get_device

def diagnose_checkpoint(checkpoint_path):
    """Check if checkpoint weights are valid"""
    print(f"Diagnosing checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint not found!")
        return

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"📋 Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Classes: {checkpoint.get('num_classes', 'unknown')}")

    # Load model
    model_config = checkpoint.get('model_config', {})
    image_size = model_config.get('image_size', 128)
    num_classes = checkpoint.get('num_classes', 7)  # Default fallback
    model_channels = model_config.get('model_channels', 128)
    num_res_blocks = model_config.get('num_res_blocks', 2)
    channel_mult_str = model_config.get('channel_mult', '1,2,4,8')
    try:
        channel_mult = tuple(int(x.strip()) for x in channel_mult_str.split(','))
    except:
        channel_mult = (1, 2, 4, 8)
    num_timesteps = model_config.get('num_timesteps', 500)
    use_lora = model_config.get('use_lora', False)
    lora_r = model_config.get('lora_r', 4)
    lora_alpha = model_config.get('lora_alpha', 16)
    lora_dropout = model_config.get('lora_dropout', 0.1)

    model = ConditionalDiffusionModel(
        image_size=image_size,
        num_classes=num_classes,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_timesteps=num_timesteps,
        beta_schedule='linear',
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Check weight statistics
    print("🔍 Weight Statistics:")
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            nan_params += torch.isnan(param).sum().item()
            inf_params += torch.isinf(param).sum().item()

    print(f"  Total parameters: {total_params:,}")
    print(f"  Zero parameters: {zero_params:,} ({zero_params/total_params*100:.1f}%)")
    print(f"  NaN parameters: {nan_params:,}")
    print(f"  Inf parameters: {inf_params:,}")

    if zero_params > total_params * 0.5:
        print("⚠️  WARNING: More than 50% of parameters are zero! Model may be collapsed.")
    if nan_params > 0 or inf_params > 0:
        print("❌ ERROR: Model has NaN or Inf values!")

    # Try a simple forward pass
    print("🧪 Testing forward pass...")
    try:
        batch_size = 2
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        class_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        with torch.no_grad():
            output = model(x, class_labels)
            predicted_noise, noise, t_out = output

        print("✅ Forward pass successful")
        print(f"  Output shape: {predicted_noise.shape}")
        print(f"  Output range: [{predicted_noise.min().item():.4f}, {predicted_noise.max().item():.4f}]")

        if predicted_noise.min() == 0 and predicted_noise.max() == 0:
            print("❌ ERROR: Model outputs all zeros!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

    # Try sampling
    print("🎨 Testing sampling...")
    try:
        class_labels = torch.tensor([0], device=device)  # Test class 0
        with torch.no_grad():
            sample = model.sample(class_labels, batch_size=1, num_inference_steps=10)

        print("✅ Sampling successful")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Sample range: [{sample.min().item():.4f}, {sample.max().item():.4f}]")

        if sample.min() == 0 and sample.max() == 0:
            print("❌ ERROR: Sample is all zeros!")
        elif sample.min() == sample.max():
            print("⚠️  WARNING: Sample has no variation!")

    except Exception as e:
        print(f"❌ Sampling failed: {e}")

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint to diagnose")
    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint)