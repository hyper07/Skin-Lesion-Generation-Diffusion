# Training and Generation Guide

This guide explains how to train and generate synthetic skin lesion images using the conditional diffusion model with the new modular code structure.

## 📁 Project Structure

The code has been organized into modular files in the `models/` directory:

```
models/
├── model.py              # Diffusion model architecture
├── dataset.py            # Dataset classes
├── data_loader.py        # Data loading functions
├── generate.py           # Generation functions and main script
├── train_diffusion.py    # Training functions
```

## 🔧 Prerequisites

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# .env file
DATA_DIR=/path/to/your/dataset/directory
```

Example:
```bash
DATA_DIR=/home/kibaek/Documents/Github/apan5560-project/dataset/GenAI Project
```

### 2. Dataset Structure

Ensure your dataset is organized as follows:

```
<DATA_DIR>/
├── dataverse_files/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images/
│   │   ├── ISIC_*.jpg
│   │   └── ...
│   ├── ISIC_metadata.csv
│   └── ISIC_images/
│       ├── ISIC_*.jpg
│       └── ...
```

### 3. Dependencies

Install required packages:

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install pandas scikit-learn matplotlib seaborn
pip install python-dotenv tqdm
```

## 🏋️ Training the Model

### Basic Training

```bash
cd models
python train_diffusion.py
```

### Advanced Training Options

```bash
cd models
python train_diffusion.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --image-size 256 \
    --num-timesteps 1000 \
    --save-interval 10 \
    --checkpoint-dir ../checkpoints \
    --samples-dir ../samples \
    --device cuda
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--learning-rate` | 1e-4 | Learning rate |
| `--image-size` | 256 | Image resolution |
| `--num-timesteps` | 1000 | Number of diffusion timesteps |
| `--save-interval` | 10 | Save checkpoint every N epochs |
| `--checkpoint-dir` | checkpoints | Directory to save checkpoints |
| `--samples-dir` | samples | Directory to save sample images |
| `--device` | auto | Device to use (cpu/cuda/mps) |

### Quick Training Script

For the easiest training experience:

```bash
./train_quick.sh
```

This script uses optimized parameters and handles device detection automatically.

## 🎨 Generating Images

### Generate All Classes

```bash
cd models
python generate.py
```

### Generate Specific Class

```bash
cd models
python generate.py --class-name "Melanoma (HAM)"
```

### Advanced Generation Options

```bash
cd models
python generate.py \
    --checkpoint ../checkpoints/final_model.pt \
    --output-dir ../generated \
    --experiment-name my_experiment \
    --samples-per-class 8 \
    --num-inference-steps 50 \
    --target-size 512 \
    --device cuda
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | checkpoints/final_model.pt | Path to trained model checkpoint |
| `--output-dir` | generated | Output directory for generated images |
| `--experiment-name` | None | Optional experiment name for organized folders |
| `--samples-per-class` | 1 | Number of samples to generate per class |
| `--num-inference-steps` | 50 | Number of denoising steps (higher = better quality) |
| `--target-size` | 256 | Output image resolution |
| `--device` | auto | Device to use (cpu/cuda/mps) |
| `--class-name` | None | Generate only specific class (default: all classes) |

## 📊 Understanding the Output

### Generated Images Structure

```
generated/
├── YYYYMMDD_HHMMSS_experiment_name/  # Experiment folder
│   ├── generation_metadata.txt       # Generation details
│   ├── all_classes_grid_*.png        # Combined grid of all classes
│   ├── Actinic_keratosis/
│   │   ├── actinic_keratosis_000_*.png
│   │   ├── actinic_keratosis_001_*.png
│   │   └── actinic_keratosis_grid_*.png
│   ├── Basal_cell_carcinoma/
│   │   └── ...
│   └── ... (other classes)
```

### Available Classes

The model generates images for these skin lesion classes:

- Actinic keratosis
- Basal cell carcinoma
- Benign keratosis (HAM)
- Dermatofibroma
- Melanoma (BCN)
- Melanoma (HAM)
- Melanoma metastasis
- Nevus
- Scar
- Seborrheic keratosis
- Solar lentigo
- Squamous cell carcinoma
- Vascular lesion

## 🔍 Monitoring Training

### TensorBoard (Optional)

```bash
pip install tensorboard
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

### Training Logs

The training script outputs:
- Loss values per epoch
- Validation metrics
- Sample generation during training
- Checkpoint saving progress

## 🛠️ Using Individual Modules

You can also use the modular components directly:

### Load Model and Generate

```python
from models.model import ConditionalDiffusionModel
from models.generate import load_model_from_checkpoint, generate_all_classes
import torch

# Load trained model
model, disease_classes, _ = load_model_from_checkpoint('checkpoints/final_model.pt')

# Generate images
generate_all_classes(
    model=model,
    disease_classes=disease_classes,
    device=torch.device('cuda'),
    output_dir='generated',
    samples_per_class=4,
    num_inference_steps=50,
    target_size=256
)
```

### Custom Training

```python
from models.model import ConditionalDiffusionModel
from models.data_loader import load_data_loaders_from_notebook
from models.train_diffusion import train_diffusion_model

# Load data
train_dataset, val_dataset, test_dataset, disease_classes = load_data_loaders_from_notebook()

# Train model
train_diffusion_model(
    epochs=50,
    batch_size=16,
    learning_rate=2e-4,
    image_size=256,
    device='cuda'
)
```

## 🚀 Performance Tips

### GPU Training
- Use `--device cuda` for NVIDIA GPUs
- Use `--device mps` for Apple Silicon Macs
- Reduce batch size if you get CUDA out of memory errors

### Memory Optimization
- Lower `--image-size` (128 or 256) for less memory usage
- Reduce `--batch-size` during training
- Use `--num-inference-steps 25` for faster generation (lower quality)

### Quality vs Speed
- Higher `--num-inference-steps` = better quality but slower
- Higher `--target-size` = higher resolution but more memory
- EMA weights provide better quality than regular checkpoints

## 🔧 Troubleshooting

### Common Issues

1. **"DATA_DIR not found"**
   - Make sure your `.env` file exists and contains `DATA_DIR=/path/to/data`

2. **CUDA out of memory**
   - Reduce batch size: `--batch-size 8`
   - Use smaller images: `--image-size 128`
   - Use CPU: `--device cpu`

3. **Import errors**
   - Make sure you're in the project root directory
   - Install missing dependencies: `pip install -r requirements.txt`

4. **No checkpoint found**
   - Train the model first or specify correct checkpoint path
   - Check that training completed successfully

### Getting Help

- Check the training logs for error messages
- Verify your dataset structure matches the requirements
- Ensure all dependencies are installed
- Try running with `--device cpu` for debugging

## 📈 Expected Results

After successful training and generation:

- **Training**: Loss should decrease steadily, validation loss should be stable
- **Generation**: High-quality synthetic skin lesion images for each class
- **Grid visualization**: Organized comparison of generated images across classes

The model typically achieves realistic skin textures and class-specific morphological features after 50-100 epochs of training.
