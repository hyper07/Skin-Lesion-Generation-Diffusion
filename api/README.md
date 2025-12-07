# APAN5560 GenAI API

This FastAPI application provides comprehensive generative AI capabilities for the skin lesion analysis project, including probability calculations, text processing, word embeddings, neural network training, and multiple generative models (GANs, Energy-based models, and Diffusion models).

## Project Structure

```
api/
├── Dockerfile                    # Container configuration for FastAPI app
├── README.md                     # API documentation
├── main.py                       # Main FastAPI application with router architecture
├── pyproject.toml                # Python project configuration and dependencies
├── start.sh                      # Application startup script
├── commands/                     # Command-line training scripts
│   ├── train_cnn.py              # Script to train CNN models
│   └── train_gan.py              # Script to train GAN models
├── checkpoints/                  # Model checkpoints and saved models
│   ├── cnn_{dataset}/            # CNN model checkpoints by dataset
│   ├── gan_{dataset}/            # GAN model checkpoints by dataset
│   ├── energy_{dataset}/         # Energy-based model checkpoints
│   └── diffusion_{dataset}/      # Diffusion model checkpoints
├── help_lib/                     # Core functionality and helper functions
│   ├── __init__.py
│   ├── probability_lib.py        # Probability theory implementations
│   ├── text_processing_lib.py    # Text processing and bigram functionality
│   ├── embeddings_lib.py         # Word embedding operations
│   ├── neural_networks.py        # Neural network training and inference
│   ├── checkpoints.py            # Checkpoint management utilities
│   ├── model.py                  # Model training utilities
│   └── data_loader.py            # Data loading utilities
├── results/                      # Output from models and functions
├── models/                       # Model definitions and Pydantic schemas
│   ├── __init__.py
│   ├── requests.py               # Request models for API endpoints
│   ├── responses.py              # Response models for API endpoints
│   ├── gan_models.py             # GAN model implementations
│   └── energy_diffusion_models.py # Energy and diffusion model implementations
└── routers/                      # FastAPI routers for organized API structure
    ├── __init__.py
    ├── probability.py            # Probability & statistics endpoints (/probability/*)
    ├── embedding.py              # Text processing & embedding endpoints (/embedding/*)
    └── neural_networks.py        # Neural network endpoints (/neural-networks/*)
```

## Features

### 📊 Probability and Information Theory
- **Entropy calculation** using Shannon's formula
- **Cross-entropy** between distributions
- **KL divergence** computation
- **Information content** calculations

### 📝 Text Generation and Analysis
- **Bigram model training** from text corpora
- **Markov chain text generation**
- **Probability matrix visualization**
- **Integration with Project Gutenberg** for training data

### 🧠 Word Embeddings and NLP
- **Word embeddings** using spaCy (300-dimensional vectors)
- **Word similarity** calculations
- **Sentence similarity** computations
- **Word algebra** operations (e.g., king - man + woman ≈ queen)

### 🔬 Neural Networks and Deep Learning
- **Activation functions**: sigmoid, softmax, ReLU
- **2D convolution** with configurable stride and padding
- **2D max pooling** with pool size and stride
- **Loss functions**: cross-entropy, mean squared error (MSE)
- **Model training**: CNN training pipeline with progress tracking
- **Model inference**: Single image and batch prediction capabilities
- **Model persistence**: Automatic saving and loading of trained models
- **Device info**: Check available devices (CPU, CUDA, MPS)
- **Model info**: CNN metadata and current model status
- **Output size calculator** for conv/pool layers
- **Checkpoint management**: Save, load, and manage model checkpoints

### 🎨 Generative Models
- **GAN training**: Train Generative Adversarial Networks on various datasets
- **Energy-based models**: Train energy-based generative models
- **Diffusion models**: Train and use diffusion models for image generation
- **Image generation**: Generate synthetic images using trained models
- **Model persistence**: Automatic saving and loading of model checkpoints
- **Training monitoring**: Real-time progress tracking with loss metrics
- **Multi-dataset support**: Flexible dataset configuration for different use cases

## Quick Start

### Using Docker (Recommended)
```bash
cd api
docker build -t apan5560-genai-api .
docker run -p 8888:8888 apan5560-genai-api
```

### Local Development
```bash
cd api
pip install -e .
python -m spacy download en_core_web_lg
uvicorn main:app --reload --port 8888
```

The FastAPI app will be available at [http://localhost:8888](http://localhost:8888) and the interactive API docs at [http://localhost:8888/docs](http://localhost:8888/docs).

## API Documentation

The API is organized into three main routers:

### API Structure (Router-Based Architecture)

#### 🧮 Probability & Statistics (`/probability/*`)
- POST `/probability/information` – Calculate information content
- POST `/probability/entropy` – Calculate entropy of a distribution
- POST `/probability/cross-entropy` – Calculate cross-entropy between distributions
- POST `/probability/kl-divergence` – Calculate KL divergence between distributions
- POST `/probability/independent-events` – Calculate intersection and union probabilities
- POST `/probability/check-independence` – Check if two events are independent
- POST `/probability/bayes-posterior` – Calculate P(A|B) using Bayes' rule
- POST `/probability/medical-test-probability` – Medical test probability calculations
- POST `/probability/distribution-statistics` – Expected value, variance, sample statistics
- POST `/probability/compare-entropy` – Compare entropy distributions
- POST `/probability/expected-value` – Calculate expected value
- POST `/probability/variance` – Calculate variance
- POST `/probability/bayes-rule` – Apply Bayes' rule
- POST `/probability/test-independence` – Test event independence

#### 📝 Text Processing & Embeddings (`/embedding/*`)
- POST `/embedding/generate` – Generate text using default bigram model
- POST `/embedding/generate-from-book` – Generate text from book-trained bigram model
- POST `/embedding/analyze-bigrams` – Analyze text and compute bigram probabilities
- GET `/embedding/bigram-matrix` – Get bigram probability matrix
- GET `/embedding/vocabulary` – Current bigram model vocabulary
- POST `/embedding/word-embedding` – Get embedding for a word
- POST `/embedding/word-similarity` – Similarity between two words
- POST `/embedding/sentence-similarity` – Similarity between two sentences
- POST `/embedding/word-algebra` – word1 + word2 − word3 vs word4
- GET `/embedding/health` – Health check with spaCy model status

#### 🧠 Neural Networks & Deep Learning (`/neural-networks/*`)
- POST `/neural-networks/activation-function` – Apply activation to input array
- POST `/neural-networks/convolution` – Apply 2D convolution
- POST `/neural-networks/pooling` – Apply 2D max pooling
- POST `/neural-networks/calculate-loss` – Compute loss (cross_entropy, mse)
- POST `/neural-networks/train` – Train CNN model on CIFAR-10 dataset
- POST `/neural-networks/train/resume` – Resume training from checkpoint
- GET `/neural-networks/training/status` – Get current training status
- POST `/neural-networks/predict` – Make prediction on single image
- POST `/neural-networks/predict-batch` – Make predictions on multiple images
- GET `/neural-networks/device-info` – Available devices and PyTorch version
- GET `/neural-networks/model-inference/{model_type}` – Info for `fcnn` or `cnn`
- GET `/neural-networks/model/info` – Current loaded model information
- GET `/neural-networks/output-size-calculator` – Compute output size for conv/pool layer
- GET `/neural-networks/checkpoints` – List available checkpoints
- POST `/neural-networks/checkpoints/load` – Load a specific checkpoint
- GET `/neural-networks/checkpoints/latest` – Get latest checkpoint info
- DELETE `/neural-networks/checkpoints` – Delete all checkpoints
- POST `/neural-networks/gan/train` – Train GAN model
- POST `/neural-networks/gan/generate` – Generate images using trained GAN
- GET `/neural-networks/gan/info` – Get information about available GAN models
- POST `/neural-networks/gan/load` – Load a specific GAN checkpoint
- POST `/neural-networks/energy/train` – Train energy-based model
- POST `/neural-networks/energy/generate` – Generate images using energy model
- GET `/neural-networks/energy/info` – Get energy model information
- POST `/neural-networks/energy/load` – Load energy model checkpoint
- POST `/neural-networks/diffusion/train` – Train diffusion model
- POST `/neural-networks/diffusion/generate` – Generate images using diffusion model
- GET `/neural-networks/diffusion/info` – Get diffusion model information
- POST `/neural-networks/diffusion/load` – Load diffusion model checkpoint

#### 🏠 Root Endpoints
- GET `/` – API info and version

### Quick Examples

#### Calculate Entropy (Probability)
```bash
curl -X POST "http://localhost:8888/probability/entropy" \
  -H "Content-Type: application/json" \
  -d '{"probabilities": [0.5, 0.3, 0.2]}'
```

#### Information Content (Probability)
```bash
curl -X POST "http://localhost:8888/probability/information" \
  -H "Content-Type: application/json" \
  -d '{"prob": 0.5}'
```

#### Generate Text (Embedding)
```bash
curl -X POST "http://localhost:8888/embedding/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
```

#### Word Similarity (Embedding)
```bash
curl -X POST "http://localhost:8888/embedding/word-similarity" \
  -H "Content-Type: application/json" \
  -d '{"word1": "apple", "word2": "orange"}'
```

#### Cross Entropy (Probability)
```bash
curl -X POST "http://localhost:8888/probability/cross-entropy" \
  -H "Content-Type: application/json" \
  -d '{"true_distribution": [0.7, 0.2, 0.1], "predicted_distribution": [0.6, 0.3, 0.1]}'
```

#### Activation Function (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/activation-function" \
  -H "Content-Type: application/json" \
  -d '{"function_name": "softmax", "input_values": [1.0, -2.0, 0.5, 3.0]}'
```

#### Convolution (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/convolution" \
  -H "Content-Type: application/json" \
  -d '{"image": [[1,2,3],[4,5,6],[7,8,9]], "kernel": [[1,0],[0,1]], "stride": 1, "padding": 0}'
```

#### Device Info (Neural Networks)
```bash
curl -X GET "http://localhost:8888/neural-networks/device-info"
```

#### Train CNN Model (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset": "cifar10", "epochs": 5, "model_type": "cnn", "batch_size": 32, "learning_rate": 0.001}'
```

#### Make Prediction (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": [[[255,0,0],[0,255,0]],[[0,0,255],[128,128,128]]], "dataset": "cifar10", "model_type": "cnn"}'
```

#### Get Model Info (Neural Networks)
```bash
curl -X GET "http://localhost:8888/neural-networks/model/info"
```

#### Train GAN Model (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/gan/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mnist", "epochs": 10, "batch_size": 64, "learning_rate": 0.0002}'
```

#### Generate Images with GAN (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/gan/generate" \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mnist", "num_images": 5}'
```

#### Get GAN Info (Neural Networks)
```bash
curl -X GET "http://localhost:8888/neural-networks/gan/info?dataset=mnist"
```

#### Train Energy-Based Model (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/energy/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mnist", "epochs": 10, "batch_size": 64}'
```

#### Train Diffusion Model (Neural Networks)
```bash
curl -X POST "http://localhost:8888/neural-networks/diffusion/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mnist", "epochs": 10, "batch_size": 64}'
```

## Dependencies

### Core Dependencies
- **FastAPI, Uvicorn** (web framework and server)
- **PyTorch, TorchVision** (deep learning framework)
- **NumPy, SciPy** (numerical computing)
- **Pandas** (data manipulation)
- **scikit-learn** (machine learning utilities)
- **spaCy** (natural language processing)
- **Pydantic** (data validation)
- **Pillow** (image processing)

### Installation
For local development:
```bash
cd api
pip install -e .
python -m spacy download en_core_web_lg
```

For production deployment using Docker:
```bash
cd api
docker build -t apan5560-genai-api .
docker run -p 8888:8888 apan5560-genai-api
```
## Testing

### Interactive API Documentation

Visit the interactive Swagger UI to explore all API endpoints:
```
http://localhost:8888/docs
```

The documentation shows three main endpoint groups:
- **Probability APIs** (`/probability/*`)
- **Embedding & Text Processing APIs** (`(/embedding/*`) 
- **Neural Network APIs** (`/neural-networks/*`)

## Command Line Training Scripts

The project includes command-line scripts for training models directly from the terminal:

### Train CNN Model
```bash
python commands/train_cnn.py --epochs 10 --model_type simple --batch_size 32
```

### Train GAN Model
```bash
python commands/train_gan.py --device mps --epochs 20 --dataset mnist
```

Available options for `train_cnn.py`:
- `--epochs`: Number of training epochs (default: 5)
- `--model_type`: Model architecture (`simple` or `resnet`, default: `simple`)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)

Available options for `train_gan.py`:
- `--device`: Device to use (`cpu`, `cuda`, `mps`, default: `cpu`)
- `--epochs`: Number of training epochs (default: 10)
- `--dataset`: Dataset to use (`mnist` or `fashion_mnist`, default: `mnist`)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate (default: 0.0002)

## Dependencies

## Enabling MPS (Apple Silicon GPU) Support

If you want to enable MPS (Metal Performance Shaders) for PyTorch on Apple Silicon (M1/M2/M3/M4), you can pass an environment variable to the Docker container:

```bash
docker run -p 8888:8888 -e PYTORCH_ENABLE_MPS=1 apan5560-genai-api
```

This sets the `PYTORCH_ENABLE_MPS` environment variable inside the container, allowing supported libraries to use the MPS backend for GPU acceleration.

> **Note:** Ensure your code and dependencies support MPS. For PyTorch, check the [official documentation](https://pytorch.org/docs/stable/notes/mps.html).

## Implementation Details

This API implements comprehensive GenAI functionality using a modular, router-based architecture:

### Help Library Module (`help_lib/`)
1. **`probability_lib.py`** → Information theory calculations (entropy, cross-entropy, KL divergence, Bayes rule)
2. **`text_processing_lib.py`** → Bigram analysis, tokenization, and text generation
3. **`embeddings_lib.py`** → Word embedding management and similarity calculations
4. **`neural_networks.py`** → CNN operations, activation functions, convolution/pooling
5. **`checkpoints.py`** → Model checkpoint saving, loading, and management
6. **`model.py`** → Model training utilities and pipelines
7. **`data_loader.py`** → Data loading and preprocessing utilities

### Models Module (`models/`)
1. **`requests.py`** → Pydantic models for API request validation
2. **`responses.py`** → Pydantic models for API response formatting
3. **`gan_models.py`** → GAN model implementations
4. **`energy_diffusion_models.py`** → Energy-based and diffusion model implementations

### Routers Module (`routers/`)
1. **`probability.py`** → All probability and statistics endpoints
2. **`embedding.py`** → Text processing, bigrams, and word embedding endpoints
3. **`neural_networks.py`** → Neural network training, inference, and generative model endpoints

### Generative Model Capabilities
- **GANs**: Train generator and discriminator networks for image synthesis
- **Energy-Based Models**: Train energy functions for data generation
- **Diffusion Models**: Implement forward and reverse diffusion processes
- **Multi-Model Support**: Switch between different generative approaches
- **Checkpoint Management**: Save and restore model states across training sessions

### Benefits of Modular Architecture
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Reusability**: Functions can be imported and used independently
- **Maintainability**: Easier to test, debug, and extend
- **Clean Imports**: Main application file is streamlined and readable
- **Scalability**: Easy to add new models and capabilities

---

**Part of the APAN5560 Final Project** - For full project documentation, see the [main README](../README.md).
