"""
Improved sample generation functions for Conditional Diffusion Model
Uses EMA weights for better quality generations
"""
import os
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from model import ConditionalDiffusionModel
from tqdm import tqdm
import torch.nn.functional as F
import re


def sanitize_filename(name: str) -> str:
    """
    Convert class names to filesystem-safe filenames

    Args:
        name: Original class name

    Returns:
        Sanitized filename-safe string
    """
    # Replace problematic characters
    name = name.replace(' ', '_')
    name = name.replace('(', '').replace(')', '')
    name = name.replace('/', '_')
    name = re.sub(r'[^\w\-_\.]', '', name)  # Remove any remaining special chars except - _ .
    return name


def create_experiment_folder(base_dir: str = "generated", experiment_name: str = None) -> Path:
    """
    Create a timestamped experiment folder for organized outputs

    Args:
        base_dir: Base directory for generations
        experiment_name: Optional experiment identifier

    Returns:
        Path to the experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        experiment_name = sanitize_filename(experiment_name)
        folder_name = f"{timestamp}_{experiment_name}"
    else:
        folder_name = timestamp

    experiment_dir = Path(base_dir) / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir


def remove_compile_prefix(state_dict):
    """Remove torch.compile prefix from state_dict keys"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # Remove '_orig_mod.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint with proper configuration"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model config
    model_config = checkpoint.get('model_config', {})

    # Remove torch.compile prefixes from state dicts
    checkpoint['model_state_dict'] = remove_compile_prefix(checkpoint['model_state_dict'])
    if 'ema_state_dict' in checkpoint:
        checkpoint['ema_state_dict'] = remove_compile_prefix(checkpoint['ema_state_dict'])

    # Create model with saved configuration
    model = ConditionalDiffusionModel(
        image_size=model_config.get('image_size', 256),
        num_classes=model_config.get('num_classes', 13),
        model_channels=model_config.get('model_channels', 128),
        num_res_blocks=model_config.get('num_res_blocks', 2),
        channel_mult=tuple(model_config.get('channel_mult', (1, 2, 3, 4))),
        num_timesteps=model_config.get('num_timesteps', 1000),
        beta_schedule=model_config.get('beta_schedule', 'cosine'),
        time_emb_dim=model_config.get('time_emb_dim', 512),
        class_emb_dim=model_config.get('class_emb_dim', 256),
    )

    # Load model weights (prefer EMA if available)
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA weights for better quality...")
        # Load EMA shadow parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        for name, param in model.named_parameters():
            if name in checkpoint['ema_state_dict']:
                param.data = checkpoint['ema_state_dict'][name]
    else:
        print("Loading regular model weights...")
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Get disease classes
    disease_classes = checkpoint.get('disease_classes', {})

    print(f"Model loaded successfully!")
    print(f"Number of classes: {len(disease_classes)}")
    print(f"Classes: {list(disease_classes.keys())}")

    return model, disease_classes, checkpoint


def generate_samples_for_class(model, class_idx, class_name, num_samples, device, output_dir, num_inference_steps=50, target_size=512):
    """Generate samples for a specific class"""
    print(f"\nGenerating {num_samples} samples for class: {class_name}")

    # Sanitize class name for filesystem
    safe_class_name = sanitize_filename(class_name)

    # Create output directory for this class
    class_dir = Path(output_dir) / safe_class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Generate in batches to avoid memory issues
    batch_size = min(8, num_samples)
    all_images = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            class_labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)

            # Generate images
            generated = model.sample(
                class_labels,
                batch_size=current_batch_size,
                num_inference_steps=num_inference_steps,
                use_ddim=True  # Use DDIM for better quality
            )

            # Denormalize
            generated = (generated + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            generated = torch.clamp(generated, 0.0, 1.0)

            # Upscale to higher resolution for better detail
            if generated.shape[-1] != target_size:
                generated = F.interpolate(generated, size=(target_size, target_size), mode='bilinear', align_corners=False)

            all_images.append(generated)

    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)

    # Save individual images with improved naming
    timestamp = datetime.now().strftime("%H%M%S")
    for i in range(num_samples):
        img = all_images[i].cpu().permute(1, 2, 0).numpy()
        filename = f"{safe_class_name}_{i:03d}_{timestamp}_{target_size}px.png"
        plt.imsave(class_dir / filename, img)

    print(f"Saved {num_samples} images to {class_dir}")

    return all_images


def create_grid_visualization(images, class_name, output_path, grid_size=None, target_size=512):
    """Create a grid visualization of generated images"""
    num_images = len(images)

    if grid_size is None:
        # Auto-calculate grid size
        cols = min(8, num_images)
        rows = (num_images + cols - 1) // cols
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(cols * (target_size / 128), rows * (target_size / 128)), squeeze=False)

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        if i < num_images:
            img = images[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f"Generated {class_name} Images", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid visualization to {output_path}")


def generate_all_classes(model, disease_classes, device, output_dir, samples_per_class=8, num_inference_steps=50, target_size=512):
    """Generate samples for all classes"""
    print("\n" + "="*60)
    print("Generating samples for all classes")
    print("="*60)

    all_class_images = []
    class_names_list = []
    safe_class_names = []

    for class_name, class_idx in sorted(disease_classes.items(), key=lambda x: x[1]):
        safe_name = sanitize_filename(class_name)
        images = generate_samples_for_class(
            model, class_idx, class_name, samples_per_class,
            device, output_dir, num_inference_steps, target_size
        )

        # Save grid for this class
        grid_path = Path(output_dir) / safe_name / f"{safe_name}_grid_{target_size}px.png"
        create_grid_visualization(images, class_name, grid_path, target_size=target_size)

        # Keep first few images for combined grid
        all_class_images.append(images[:min(4, samples_per_class)])
        class_names_list.append(class_name)
        safe_class_names.append(safe_name)

    # Create combined grid showing all classes
    print("\nCreating combined grid for all classes...")
    fig, axes = plt.subplots(len(disease_classes), min(4, samples_per_class),
                             figsize=(24, 6 * len(disease_classes)), squeeze=False)

    for class_idx, (images, class_name) in enumerate(zip(all_class_images, class_names_list)):
        for img_idx in range(len(images)):
            ax = axes[class_idx, img_idx]

            img = images[img_idx].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')

            if img_idx == 0:
                ax.set_title(class_name, fontsize=10, fontweight='bold', loc='left')

    plt.suptitle("Generated Skin Lesion Images - All Classes", fontsize=16, fontweight='bold')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = Path(output_dir) / f"all_classes_grid_{timestamp}_{target_size}px.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined grid to {combined_path}")

    # Save metadata file
    metadata_path = Path(output_dir) / "generation_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Generation timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Samples per class: {samples_per_class}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Target size: {target_size}px\n")
        f.write(f"Total classes: {len(disease_classes)}\n")
        f.write("\nClass mapping:\n")
        for orig, safe in zip(class_names_list, safe_class_names):
            f.write(f"  {orig} -> {safe}\n")

    print(f"Saved metadata to {metadata_path}")

    print("\n" + "="*60)
    print("Generation complete!")
    print(f"All images saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate improved samples from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", 
                        help="Path to checkpoint file")
    parser.add_argument("--output-dir", type=str, default="generated", 
                        help="Base output directory for generated images")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name for organized folder structure")
    parser.add_argument("--samples-per-class", type=int, default=1, 
                        help="Number of samples to generate per class")
    parser.add_argument("--num-inference-steps", type=int, default=50, 
                        help="Number of inference steps (more = better quality, slower)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use: 'cpu', 'cuda', 'mps' (default: auto-detect)")
    parser.add_argument("--class-name", type=str, default=None, 
                        help="Generate only for specific class (default: all classes)")
    parser.add_argument("--target-size", type=int, default=256, 
                        help="Target image size for upscaling (default: 256)")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon) acceleration")
        else:
            # Try CUDA, fall back to CPU if error
            try:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print("Using CUDA (NVIDIA GPU) acceleration")
                else:
                    device = torch.device('cpu')
                    print("Using CPU")
            except RuntimeError as e:
                print(f"CUDA check failed: {e}")
                device = torch.device('cpu')
                print("Falling back to CPU")
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    
    # Load model
    model, disease_classes, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create organized experiment folder
    experiment_dir = create_experiment_folder(args.output_dir, args.experiment_name)
    print(f"Output directory: {experiment_dir}")
    
    # Generate samples
    if args.class_name:
        # Generate for specific class
        if args.class_name not in disease_classes:
            print(f"Error: Class '{args.class_name}' not found!")
            print(f"Available classes: {list(disease_classes.keys())}")
            return
        
        class_idx = disease_classes[args.class_name]
        images = generate_samples_for_class(
            model, class_idx, args.class_name, args.samples_per_class,
            device, experiment_dir, args.num_inference_steps, args.target_size
        )
        
        # Save grid
        safe_name = sanitize_filename(args.class_name)
        grid_path = experiment_dir / safe_name / f"{safe_name}_grid_{args.target_size}px.png"
        create_grid_visualization(images, args.class_name, grid_path, target_size=args.target_size)
    else:
        # Generate for all classes
        generate_all_classes(
            model, disease_classes, device, experiment_dir,
            args.samples_per_class, args.num_inference_steps, args.target_size
        )


if __name__ == "__main__":
    main()
