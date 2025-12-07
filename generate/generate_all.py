# generate_all.py
import torch
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import (
    load_cgan, load_conditional_diffusion, load_prebuilt_gan, load_prebuilt_diffusion,
    generate_cgan_images, generate_conditional_diffusion_images,
    generate_prebuilt_gan_images, generate_prebuilt_diffusion_images,
    CHECKPOINT_PATHS, IDX_TO_CLASS, OUTPUT_BASE_DIR, device
)

def generate_for_model(model_name, num_samples=200):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(OUTPUT_BASE_DIR) / model_name / timestamp
    
    print(f"\n{'='*60}")
    print(f"Generating {model_name}")
    print(f"{'='*60}")
    
    if model_name == "cgan":
        model, z_dim = load_cgan(CHECKPOINT_PATHS["cgan"])
        for class_id in range(13):
            class_name = IDX_TO_CLASS[class_id]
            class_dir = output_base / f"class_{class_id}"
            generate_cgan_images(model, z_dim, class_id, class_name, num_samples, class_dir)
    
    elif model_name == "conditional_diffusion":
        model = load_conditional_diffusion(CHECKPOINT_PATHS["conditional_diffusion"])
        for class_id in range(13):
            class_name = IDX_TO_CLASS[class_id]
            class_dir = output_base / f"class_{class_id}_{class_name}"
            generate_conditional_diffusion_images(model, class_id, class_name, num_samples, class_dir)
    
    elif model_name == "prebuilt_gan":
        model, z_dim = load_prebuilt_gan(CHECKPOINT_PATHS["prebuilt_gan"])
        for class_id in range(13):
            class_name = IDX_TO_CLASS[class_id]
            class_dir = output_base / f"class_{class_id}"
            generate_prebuilt_gan_images(model, z_dim, class_id, class_name, num_samples, class_dir)
    
    elif model_name == "prebuilt_diffusion":
        model = load_prebuilt_diffusion(CHECKPOINT_PATHS["prebuilt_diffusion"])
        for class_id in range(13):
            class_name = IDX_TO_CLASS[class_id]
            class_dir = output_base / f"class_{class_id}"
            generate_prebuilt_diffusion_images(model, class_id, class_name, num_samples, class_dir)
    
    print(f"✓ Completed: {output_base}")

if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "cgan"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    generate_for_model(model, num_samples)