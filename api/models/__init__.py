"""
Models package for SPS GenAI API.
Contains Pydantic models for request/response validation and neural network architectures.
"""

# Import GAN models for easy access
from .gan_models import (
    Generator,
    Discriminator,
    GAN,
    GANFactory,
    get_gan,
    get_generator_only,
    get_discriminator_only,
    load_trained_gan,
    get_dataloader,
    train_gan
)

# Import CNN models for easy access
from .cnn_models import (
    SimpleCNN,
    EnhancedCNN,
    CustomCNN,
    CNNFactory,
    get_cifar10_model,
    get_assignment_model
)

from .bigram_model import (
    BigramModel,
    analyze_bigrams,
    generate_text,
    simple_tokenizer,
    print_bigram_probs_matrix_python
)

# Import Energy and Diffusion models
from .energy_diffusion_models import (
    EnergyModel,
    EnergyModelTrainer,
    DiffusionModel,
    DiffusionTrainer,
    get_energy_model,
    get_diffusion_model,
    train_energy_model,
    train_diffusion_model,
    get_cifar10_dataloader as get_cifar10_dataloader_ed
)

# Re-export request/response models
from .requests import *
from .responses import *