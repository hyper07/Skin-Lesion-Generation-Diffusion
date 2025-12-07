# download_models.py
import os
from download_dataset import download_from_google_drive

# Mapping of model names to their Google Drive file IDs
MODEL_IDS = {
    "cgan": "1P2FAeHJgSvo65LlfQVLWxJzWG0gchjuQ",
    "conditional_diffusion": "1ackjP5uXU6Yi1zLrc-d94cMI3J9Ejo0Q",
    "prebuilt_gan": "13UxLA1xzRnHOAKu-AiOzWswuIcPzfN5O",
    "prebuilt_diffusion": "1QRYPKxACk10lYjPy4zWik3SrILM8HYz9",
}

MODEL_NAMES = {
    "cgan": "cgan128_final",
    "conditional_diffusion": "diffusion_final",
    "prebuilt_gan": "G_final",
    "prebuilt_diffusion": "prebuilt_diffusion_epoch_88_final",
}

def download_models(model_name=None, output_dir="checkpoints"):
    """
    Download one or all models into the specified directory.
    
    Args:
        model_name (str): One of 'cgan', 'conditional_diffusion', 'prebuilt_gan', 'prebuilt_diffusion'.
                         If None, downloads all models.
        output_dir (str): Directory where checkpoints will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_name:
        if model_name not in MODEL_IDS:
            raise ValueError(f"Invalid model name: {model_name}")
        ids_to_download = {model_name: MODEL_IDS[model_name]}
    else:
        ids_to_download = MODEL_IDS

    for name, file_id in ids_to_download.items():
        # All models save to their own folder
        folder_name = name
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
        destination = os.path.join(output_dir, f"{folder_name}/{MODEL_NAMES[name]}.pt")

        if os.path.exists(destination):
            print(f"Model '{name}' already exists at {destination}, skipping download.")
            continue

        print(f"Downloading {name} model into {destination}...")

        success = download_from_google_drive(file_id, destination)
        if not success:
            print(f"Failed to download {name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download model checkpoints from Google Drive")
    parser.add_argument("--model", type=str, choices=list(MODEL_IDS.keys()),
                        help="Model name to download (default: all)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    download_models(model_name=args.model, output_dir=args.output_dir)
