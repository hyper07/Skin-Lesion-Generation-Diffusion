# PB Diffusion Model

This project implements a conditional diffusion model for generating synthetic skin disease images based on the HAM10000 and BCN20000 datasets. The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and is built with PyTorch and Hugging Face Diffusers.

## Prerequisites

- Python 3.8 or higher
- PyTorch (with CUDA support if using GPU)
- torchvision
- diffusers
- tqdm
- tensorboard
- Other dependencies: numpy, pillow, etc. (install via `pip install torch torchvision diffusers tqdm tensorboard`)

## Data Preparation

1. Download the HAM10000 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
2. Download the BCN20000 dataset (metadata and images).
3. Set the `DATA_DIR` environment variable to the root directory containing the datasets. The expected structure is:
   ```
   $DATA_DIR/
   ├── HAM10000_metadata.csv
   ├── HAM10000_images/
   │   └── ISIC_*.jpg
   ├── ISIC_metadata.csv
   └── ISIC_images/
       └── ISIC_*.jpg
   ```

## Training the Model

1. Ensure all dependencies are installed.
2. Set the `DATA_DIR` environment variable.
3. Run the training script with desired parameters:

   ```bash
   # Basic training with auto device detection
   python train_diffusion.py

   # Training with custom parameters
   python train_diffusion.py --device cuda --batch-size 16 --num-epochs 10 --img-size 256
   ```

### Configuration

The script supports the following command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | auto | Device to use: auto, cpu, cuda, mps |
| `--batch-size` | 32 | Batch size (adjust based on GPU memory) |
| `--img-size` | 128 | Image resolution (128x128) |
| `--num-epochs` | 5 | Number of training epochs |
| `--learning-rate` | auto | Learning rate (auto-calculated if not specified) |
| `--top-n-classes` | 3 | Number of disease classes to use |
| `--log-dir` | ./logs | TensorBoard log directory |
| `--checkpoint-dir` | ../checkpoints | Checkpoint directory |
| `--use-lora` | True | Use LoRA for efficient training |

The script will:
- Load and preprocess the data
- Initialize the diffusion model with LoRA
- Resume from the latest checkpoint if available (with robust error handling)
- Train the model with AdamW optimizer and cosine annealing scheduler
- Save checkpoints every 200 steps and at the end of each epoch
- Log training progress to TensorBoard with tqdm status bars

### Monitoring Training

- Training logs are saved to `./logs/`
- Checkpoints are saved to `./checkpoints/` (parent directory)
- Use TensorBoard to monitor loss curves: `tensorboard --logdir ./logs`

## Generating Images

After training, use the `generate.py` script to generate synthetic images conditioned on specific disease classes.

### Basic Usage

```bash
python generate.py --class_id 0 --output ./generated
```

### Command Line Arguments

- `--checkpoint`: Path to a specific checkpoint file (optional, defaults to latest in `../checkpoints/`)
- `--class_id`: Disease class ID to condition on (required, 0 to num_classes-1)
- `--output`: Output directory for generated images (default: `./generated`)
- `--num_classes`: Total number of classes (default: 3)

### Example

Generate an image for class 1 and save to a custom directory:

```bash
python generate.py --class_id 1 --output ./my_generated_images --num_classes 3
```

The generated images will be saved as PNG files with timestamps, e.g., `class_1_20251111_120000.png`.

## Model Architecture

- **DiffusionModel**: Custom model class that combines a pre-trained VAE encoder/decoder with a UNet conditioned on class labels
- **LoRA**: Applied to the UNet for parameter-efficient fine-tuning
- **Scheduler**: DDPM scheduler for the diffusion process

## Troubleshooting

- **Checkpoint loading issues**: The script includes robust error handling and will fall back to fresh training if checkpoint loading fails
- **No checkpoint found**: If starting fresh, the script will initialize a new model
- **CUDA out of memory**: Reduce `--batch-size` or `--img-size`
- **Data loading errors**: Verify `DATA_DIR` is set correctly and datasets are in the expected locations
- **NaN losses**: The script skips steps with NaN outputs to prevent training instability
- **Device selection**: Use `--device auto` for automatic detection, or specify `cpu`, `cuda`, or `mps`

## License

[Add license information if applicable]
