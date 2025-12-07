#!/bin/bash
# Fix for macOS mutex crash with diffusers/PyTorch

echo "=== Fixing macOS PyTorch/Diffusers Mutex Issue ==="
echo ""
echo "This will:"
echo "1. Uninstall current torch/diffusers"
echo "2. Install CPU-only PyTorch (more stable on macOS)"
echo "3. Reinstall diffusers"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Uninstall problematic packages
pip uninstall -y torch torchvision torchaudio diffusers transformers accelerate

# Install PyTorch with MPS support (standard for macOS)
pip install --no-cache-dir torch torchvision torchaudio

# Reinstall diffusers stack
pip install --no-cache-dir diffusers transformers accelerate

echo ""
echo "=== IMPORTANT: Runtime Fix ==="
echo "To prevent the mutex error while using MPS, you must set these environment variables:"
echo "export KMP_DUPLICATE_LIB_OK=True"
echo "export OMP_NUM_THREADS=1"
echo ""
echo "You can run this in your terminal now:"
echo "  export KMP_DUPLICATE_LIB_OK=True && export OMP_NUM_THREADS=1"
echo ""
echo "Or add them to your ~/.zshrc to make it permanent."
echo "==============================" safetensors peft

echo ""
echo "✓ Installation complete!"
echo ""
echo "Now test with:"
echo "python generate/generate_prebuilt_diffusion.py \\"
echo "    --checkpoint checkpoints/probability_diffusion/prebuilt_diffusion_epoch_88_final.pt \\"
echo "    --num_classes 3 --class_id 0 --num_samples 4 \\"
echo "    --num_inference_steps 50 --img_size 128 \\"
echo "    --output output/probability_diffusion/generated.png"
