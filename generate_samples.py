"""
Generate synthetic skin lesion images using trained conditional diffusion model
"""

import os
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from dotenv import load_dotenv

from models.conditional_diffusion import ConditionalDiffusionModel
from training_config import TrainingConfig

load_dotenv()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get metadata from checkpoint
    disease_classes = checkpoint.get('disease_classes')
    num_classes = checkpoint.get('num_classes')
    
    # Get model config from checkpoint if available, otherwise use defaults
    model_config = checkpoint.get('model_config', {})
    image_size = model_config.get('image_size', 256)
    model_channels = model_config.get('model_channels', 128)
    num_res_blocks = model_config.get('num_res_blocks', 2)
    channel_mult = model_config.get('channel_mult', (1, 2, 4, 8))
    num_timesteps = model_config.get('num_timesteps', 1000)
    beta_schedule = model_config.get('beta_schedule', 'linear')
    
    # Create model with config from checkpoint
    model = ConditionalDiffusionModel(
        image_size=image_size,
        num_classes=num_classes,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_epoch' in checkpoint:
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f} (epoch {checkpoint['best_epoch']})")
    if 'training_config' in checkpoint:
        print(f"Training config: {checkpoint['training_config']}")
    
    return model, disease_classes


def generate_images(
    checkpoint_path,
    class_name=None,
    class_idx=None,
    num_samples=16,
    num_inference_steps=50,
    output_dir="generated_images",
):
    """Generate synthetic images"""
    
    # Get device
    config = TrainingConfig()
    device = config.device
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, disease_classes = load_model(checkpoint_path, device)
    
    # Get class index
    if class_name is not None:
        if class_name not in disease_classes:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {list(disease_classes.keys())}")
        class_idx = disease_classes[class_name]
    elif class_idx is None:
        raise ValueError("Must provide either class_name or class_idx")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    print(f"Generating {num_samples} images for class {class_idx}...")
    class_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.sample(class_labels, batch_size=num_samples, num_inference_steps=num_inference_steps)
    
    # Denormalize for visualization
    generated = (generated + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    generated = torch.clamp(generated, 0.0, 1.0)
    
    # Save individual images
    class_name = [name for name, idx in disease_classes.items() if idx == class_idx][0]
    for i in range(num_samples):
        img = generated[i].cpu().permute(1, 2, 0).numpy()
        plt.imsave(
            os.path.join(output_dir, f"{class_name}_{i:03d}.png"),
            img
        )
    
    # Create grid visualization
    grid_size = int(num_samples ** 0.5)
    if grid_size * grid_size < num_samples:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = generated[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f"{class_name} (Class {class_idx})", fontsize=14, fontweight='bold')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"{class_name}_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGenerated {num_samples} images for class '{class_name}'")
    print(f"Images saved to: {output_dir}")
    print(f"Grid visualization saved to: {grid_path}")


def generate_all_classes(
    checkpoint_path,
    num_samples_per_class=8,
    num_inference_steps=50,
    output_dir="generated_images",
):
    """Generate samples for all classes"""
    
    # Get device
    config = TrainingConfig()
    device = config.device
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, disease_classes = load_model(checkpoint_path, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each class
    num_classes = len(disease_classes)
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(15, 5 * num_classes))
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in sorted(disease_classes.items(), key=lambda x: x[1]):
        print(f"Generating {num_samples_per_class} images for '{class_name}' (class {class_idx})...")
        
        class_labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.sample(class_labels, batch_size=num_samples_per_class, num_inference_steps=num_inference_steps)
        
        # Denormalize
        generated = (generated + 1.0) / 2.0
        generated = torch.clamp(generated, 0.0, 1.0)
        
        # Plot
        for i in range(num_samples_per_class):
            ax = axes[class_idx, i] if num_classes > 1 else axes[i]
            img = generated[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"{class_name}", fontsize=12, fontweight='bold')
        
        # Save individual images for this class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_samples_per_class):
            img = generated[i].cpu().permute(1, 2, 0).numpy()
            plt.imsave(os.path.join(class_dir, f"{class_name}_{i:03d}.png"), img)
    
    plt.tight_layout()
    all_classes_path = os.path.join(output_dir, "all_classes_grid.png")
    plt.savefig(all_classes_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGenerated samples for all {num_classes} classes")
    print(f"Images saved to: {output_dir}")
    print(f"Grid visualization saved to: {all_classes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic skin lesion images")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--class-name", type=str, default=None, help="Class name to generate")
    parser.add_argument("--class-idx", type=int, default=None, help="Class index to generate")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--output-dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--all-classes", action="store_true", help="Generate samples for all classes")
    parser.add_argument("--samples-per-class", type=int, default=8, help="Samples per class (when using --all-classes)")
    
    args = parser.parse_args()
    
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

