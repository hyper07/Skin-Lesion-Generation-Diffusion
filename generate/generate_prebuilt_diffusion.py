"""
Generation script for Probability Diffusion Model (Stable Diffusion based)
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Generate images with Probability Diffusion Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/prebuilt_diffusion/prebuilt_diffusion_epoch_88_final.pt",help="Path to checkpoint file")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID to generate")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps (not used with pb_diffusion)")
    parser.add_argument("--img_size", type=int, default=128, help="Image size (not used with pb_diffusion, it uses 64)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--output", type=str, default="output/prebuilt_diffusion/generated.png", help="Output path")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    # Use local generate script instead of pb_diffusion folder
    local_generate_script = project_root / "generate" / "generate_pb_diffusion.py"
    
    # Make checkpoint path absolute
    if not Path(args.checkpoint).is_absolute():
        checkpoint_path = (project_root / args.checkpoint).resolve()
    else:
        checkpoint_path = Path(args.checkpoint)
    
    # Make output path absolute - for probability diffusion, output is a directory
    if not Path(args.output).is_absolute():
        output_dir = (project_root / args.output).resolve()
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Probability Diffusion Generation ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Class ID: {args.class_id}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print(f"=" * 40)
    print()
    print(f"Image size is fixed at 64x64 (pb_diffusion default)")
    print()
    
    # Use local generate_pb_diffusion.py directly
    cmd = [
        sys.executable,
        str(local_generate_script),
        "--checkpoint", str(checkpoint_path),
        "--class_id", str(args.class_id),
        "--num_samples", str(args.num_samples),
        "--output", str(output_dir),
        "--num_classes", str(args.num_classes)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=str(local_generate_script.parent))
    
    if result.returncode == 0:
        print(f"\n✓✓✓ Generation complete! ✓✓✓")
        print(f"Check output in: {output_dir}/class_{args.class_id}/")
    else:
        print(f"\n✗ Generation failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
