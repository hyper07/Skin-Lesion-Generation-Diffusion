"""
Batch generation script for evaluation images
Generates 200 images per class for each model
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Model configurations
MODELS = {
    "cgan": {
        "script": "generate/generate_cgan.py",
        "checkpoint": "checkpoints/cgan/cgan128_final.pt",
        "samples_per_batch": 50,  # Generate in batches to avoid memory issues
    },
    "conditional_diffusion": {
        "script": "generate/generate_diffusion.py",
        "checkpoint": "checkpoints/conditional_diffusion/diffusion_final.pt",
        "samples_per_batch": 20,  # Smaller batches for diffusion
    },
    "probability_diffusion": {
        "script": "generate/generate_pb_diffusion.py",
        "checkpoint": "checkpoints/probability_diffusion/prebuilt_diffusion_epoch_88_final.pt",
        "samples_per_batch": 10,  # Even smaller for probability diffusion
    },
    "prebuilt_gan": {
        "script": "generate/generate_prebuilt_gan.py",
        "checkpoint": "checkpoints/prebuilt_gan/G_final.pt",
        "samples_per_batch": 50,
    }
}

DISEASE_CLASSES = {
    "Actinic keratosis": 0,
    "Basal cell carcinoma": 1,
    "Benign keratosis (HAM)": 2,
    "Dermatofibroma": 3,
    "Melanoma (BCN)": 4,
    "Melanoma (HAM)": 5,
    "Melanoma metastasis": 6,
    "Nevus": 7,
    "Scar": 8,
    "Seborrheic keratosis": 9,
    "Solar lentigo": 10,
    "Squamous cell carcinoma": 11,
    "Vascular lesion": 12
}

def generate_for_model(model_name, model_config, output_base, num_samples_per_class=200):
    """Generate images for a specific model"""
    print(f"\n{'='*60}")
    print(f"Generating images for {model_name.upper()}")
    print(f"{'='*60}")

    model_output_dir = output_base / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    script_path = project_root / model_config["script"]
    checkpoint_path = project_root / model_config["checkpoint"]
    samples_per_batch = model_config["samples_per_batch"]

    # Check if checkpoint exists
    if not os.path.exists(str(checkpoint_path)):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    total_generated = 0

    for class_name, class_id in DISEASE_CLASSES.items():
        print(f"\n📁 Generating {num_samples_per_class} images for {class_name} (class {class_id})")

        class_dir = model_output_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images
        existing_images = len(list(class_dir.glob("*.png")))
        if existing_images >= num_samples_per_class:
            print(f"  ✅ Already have {existing_images} images, skipping")
            total_generated += existing_images
            continue

        images_needed = num_samples_per_class - existing_images
        print(f"  📊 Need {images_needed} more images (already have {existing_images})")

        # Generate in batches
        batch_num = 0
        while images_needed > 0:
            current_batch_size = min(samples_per_batch, images_needed)

            # Create temporary output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if model_name == "probability_diffusion":
                # Probability diffusion creates its own directory structure
                temp_output = model_output_dir / f"temp_{timestamp}"
            else:
                # Other models generate individual images to a temp directory
                temp_output = model_output_dir / f"temp_{timestamp}"
                temp_output.mkdir(parents=True, exist_ok=True)
                actual_output = temp_output / "temp.png"  # Dummy file for compatibility

            print(f"  🔄 Batch {batch_num + 1}: Generating {current_batch_size} images...")

            # Prepare command
            if model_name == "cgan":
                cmd = [
                    sys.executable, str(script_path),
                    "--checkpoint", str(checkpoint_path),
                    "--num_samples", str(current_batch_size),
                    "--class_id", str(class_id),
                    "--output", str(actual_output)
                ]
            elif model_name == "conditional_diffusion":
                cmd = [
                    sys.executable, str(script_path),
                    "--checkpoint", str(checkpoint_path),
                    "--class_id", str(class_id),
                    "--num_samples", str(current_batch_size),
                    "--num_inference_steps", "50",
                    "--output", str(actual_output)
                ]
            elif model_name == "probability_diffusion":
                cmd = [
                    sys.executable, str(script_path),
                    "--checkpoint", str(checkpoint_path),
                    "--class_id", str(class_id),
                    "--num_samples", str(current_batch_size),
                    "--num_classes", "13",
                    "--output", str(temp_output)
                ]
            elif model_name == "prebuilt_gan":
                cmd = [
                    sys.executable, str(script_path),
                    "--checkpoint", str(checkpoint_path),
                    "--num_classes", "13",
                    "--samples_per_class", str(current_batch_size),
                    "--class_id", str(class_id),
                    "--output", str(actual_output)
                ]

            # Run command
            try:
                result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print(f"    ✅ Batch completed successfully")

                    # Move/copy generated images to class directory
                    if model_name == "probability_diffusion":
                        # Probability diffusion creates individual files in class subdirectories
                        temp_class_dir = temp_output / f"class_{class_id}"
                        if temp_class_dir.exists():
                            for img_file in temp_class_dir.glob("*.png"):
                                # Rename to avoid conflicts
                                existing_count = len(list(class_dir.glob("*.png")))
                                new_name = class_dir / f"{existing_count:03d}.png"
                                img_file.rename(new_name)
                    else:
                        # Other models generate individual images to the temp directory
                        existing_count = len(list(class_dir.glob("*.png")))
                        for img_file in temp_output.glob("*.png"):
                            # Rename to avoid conflicts
                            new_name = class_dir / f"{existing_count:03d}.png"
                            img_file.rename(new_name)
                            existing_count += 1

                    # Count newly generated images
                    new_count = len(list(class_dir.glob("*.png")))
                    batch_generated = new_count - existing_images
                    images_needed = num_samples_per_class - new_count
                    total_generated += batch_generated

                    print(f"    📊 Progress: {new_count}/{num_samples_per_class} images")

                else:
                    print(f"    ❌ Batch failed: {result.stderr}")
                    break

            except subprocess.TimeoutExpired:
                print(f"    ⏰ Batch timed out after 5 minutes")
                break
            except Exception as e:
                print(f"    ❌ Batch error: {str(e)}")
                break

            batch_num += 1

        # Final count for this class
        final_count = len(list(class_dir.glob("*.png")))
        print(f"  ✅ Completed {class_name}: {final_count}/{num_samples_per_class} images")

    print(f"\n📊 {model_name.upper()} Summary:")
    print(f"  Total images generated: {total_generated}")
    return total_generated > 0

def split_grid_to_individual_images(grid_path, output_dir, num_images):
    """Split a grid image into individual images - DEPRECATED: scripts now generate individual images"""
    pass

def main():
    parser = argparse.ArgumentParser(description="Batch generate evaluation images")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()) + ["all"],
                       default=["all"], help="Models to generate for")
    parser.add_argument("--output", type=str, default="output/evaluation_images",
                       help="Output base directory")
    parser.add_argument("--samples_per_class", type=int, default=200,
                       help="Number of samples to generate per class")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip classes that already have enough images")

    args = parser.parse_args()

    # Determine which models to process
    if "all" in args.models:
        selected_models = list(MODELS.keys())
    else:
        selected_models = args.models

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("🎨 Batch Image Generation for Evaluation")
    print(f"📁 Output directory: {output_base}")
    print(f"📊 Target: {args.samples_per_class} images per class")
    print(f"🤖 Models: {', '.join(selected_models)}")
    print(f"📋 Classes: {len(DISEASE_CLASSES)} disease types")
    print(f"🎯 Total images needed: {len(selected_models) * len(DISEASE_CLASSES) * args.samples_per_class:,}")

    total_generated = 0
    successful_models = 0

    for model_name in selected_models:
        if generate_for_model(model_name, MODELS[model_name], output_base, args.samples_per_class):
            successful_models += 1
        else:
            print(f"❌ Failed to generate for {model_name}")

    print(f"\n{'='*60}")
    print("🎉 Batch generation completed!")
    print(f"✅ Successful models: {successful_models}/{len(selected_models)}")
    print(f"📁 Check results in: {output_base}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()