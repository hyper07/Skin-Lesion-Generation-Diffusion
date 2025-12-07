# Image Generation and Training Pages

## Overview

Two new pages have been added to the Streamlit app for synthetic medical image generation:

1. **4_Image_Generation.py** - Generate synthetic skin lesion images
2. **5_Training.py** - Train and fine-tune generative models

## Features

### Image Generation Page

- **Model Selection**: Choose from three different generative models:
  - Conditional GAN (CGAN)
  - Conditional Diffusion Model
  - Pre-built GAN

- **Disease Class Selection**: Generate images for specific skin lesion types:
  - Nevus (common moles)
  - Melanoma (malignant cancer)
  - Basal cell carcinoma (common cancer)

- **Customizable Parameters**:
  - Number of images to generate (1-16)
  - Model-specific settings (latent dimensions, inference steps)
  - Custom checkpoint paths

- **Interactive Output**:
  - View generated images in a grid layout
  - Download individual images as PNG files
  - Real-time generation with progress indicators

### Training Page

- **Training Modes**:
  - Train from scratch
  - Fine-tune existing models

- **Configurable Parameters**:
  - Batch size
  - Number of epochs
  - Learning rate
  - Image resolution (64, 128, 256)

- **Model-Specific Settings**:
  - **CGAN**: Latent dimension, discriminator learning rate
  - **Diffusion**: Number of timesteps, beta schedule
  - **Pre-built GAN**: Latent dimension

- **Training Management**:
  - Generate training commands
  - Monitor training progress
  - View training logs
  - Save and load checkpoints

## Installation

1. Install the required dependencies:

```bash
cd app-streamlit
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:

```bash
DATA_DIR=/path/to/your/dataset
```

3. Ensure you have trained model checkpoints in the appropriate directories:
   - `checkpoints/cgan/generator_final.pt`
   - `checkpoints/conditional_diffusion/best_model.pt`
   - `checkpoints/prebuilt_gan/generator_final.pt`

## Usage

### Running the Streamlit App

```bash
cd app-streamlit
streamlit run Home.py
```

Then navigate to:
- **Image Generation**: Page 4 in the sidebar
- **Training**: Page 5 in the sidebar

### Generating Images

1. Select a model from the dropdown
2. Choose a disease class
3. Set the number of images to generate
4. Adjust model-specific parameters (optional)
5. Click "Generate Images"
6. Download individual images using the download buttons

### Training Models

1. Select a model to train
2. Choose training mode (from scratch or fine-tune)
3. Configure training parameters
4. Set data and output paths
5. Click "Start Training" to generate the training command
6. Copy and run the command in your terminal

**Note**: Training runs in the terminal for better monitoring and control.

## Model Information

### Conditional GAN (CGAN)
- **Speed**: Fast (~1 second per batch)
- **Quality**: Good quality for synthetic training data
- **Best for**: Quick generation, data augmentation

### Conditional Diffusion Model
- **Speed**: Slower (depends on inference steps)
- **Quality**: High quality with fine details
- **Best for**: Research-grade synthetic images

### Pre-built GAN
- **Speed**: Fast (~1 second per batch)
- **Quality**: Good baseline quality
- **Best for**: Transfer learning scenarios

## Troubleshooting

### Checkpoint Not Found Error
- Ensure models are trained and saved in the correct directories
- Use custom checkpoint paths in the sidebar
- Check file permissions

### Out of Memory Error
- Reduce batch size
- Decrease image size
- Reduce number of samples to generate

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify that the project root is in Python path
- Check that model files exist in the `models/` directory

## Device Support

The app automatically detects and uses the best available device:
- **Mac M1/M2/M3**: Metal Performance Shaders (MPS)
- **NVIDIA GPU**: CUDA acceleration
- **CPU**: Fallback for compatibility

## File Structure

```
app-streamlit/
├── pages/
│   ├── 4_Image_Generation.py    # Image generation interface
│   └── 5_Training.py             # Training interface
├── functions/
│   └── model_utils.py            # Helper functions for models
├── requirements.txt              # Updated with torch dependencies
└── README_GENERATION.md          # This file
```

## Additional Resources

- Model architectures: See `models/` directory
- Training scripts: See `train/` directory
- Generation scripts: See `generate/` directory
- Training configuration: See `training_config.py`

## Notes

- Training requires significant computational resources
- Generation is faster but still requires GPU/MPS for best performance
- Checkpoints can be large (100MB+), ensure adequate storage
- For production use, consider adding authentication and rate limiting
