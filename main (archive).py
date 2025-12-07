"""
Main entry point for apan5560-project
Conditional Diffusion Model for Skin Lesion Generation
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Conditional Diffusion Model for Skin Lesion Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the diffusion model
  python main.py train --epochs 100 --batch-size 16
  
  # Generate samples from trained model
  python main.py generate checkpoints/final_model.pt --class-name "Nevus" --num-samples 16
  
  # Generate samples for all classes
  python main.py generate checkpoints/final_model.pt --all-classes
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the diffusion model')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--image-size', type=int, default=256, help='Image size')
    train_parser.add_argument('--num-timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    train_parser.add_argument('--save-interval', type=int, default=10, help='Save checkpoint every N epochs')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--samples-dir', type=str, default='samples', help='Samples directory')
    train_parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='Device to use (default: auto-detect)')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic images')
    gen_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    gen_parser.add_argument('--class-name', type=str, default=None, help='Class name to generate')
    gen_parser.add_argument('--class-idx', type=int, default=None, help='Class index to generate')
    gen_parser.add_argument('--num-samples', type=int, default=16, help='Number of samples to generate')
    gen_parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of denoising steps')
    gen_parser.add_argument('--output-dir', type=str, default='generated_images', help='Output directory')
    gen_parser.add_argument('--all-classes', action='store_true', help='Generate samples for all classes')
    gen_parser.add_argument('--samples-per-class', type=int, default=8, help='Samples per class (when using --all-classes)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from train_diffusion import train_diffusion_model
        train_diffusion_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_size=args.image_size,
            num_timesteps=args.num_timesteps,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
            samples_dir=args.samples_dir,
            device=args.device,
        )
    elif args.command == 'generate':
        from generate_samples import generate_images, generate_all_classes
        if args.all_classes:
            generate_all_classes(
                args.checkpoint,
                num_samples_per_class=args.samples_per_class,
                num_inference_steps=args.num_inference_steps,
                output_dir=args.output_dir,
            )
        else:
            generate_images(
                args.checkpoint,
                class_name=args.class_name,
                class_idx=args.class_idx,
                num_samples=args.num_samples,
                num_inference_steps=args.num_inference_steps,
                output_dir=args.output_dir,
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
