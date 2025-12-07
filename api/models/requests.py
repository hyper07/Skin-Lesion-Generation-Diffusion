"""
Request models for SPS GenAI API.
Pydantic models for validating incoming API requests.
"""

from pydantic import BaseModel
from typing import List, Optional


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class ProbabilityDistribution(BaseModel):
    probabilities: List[float]


class CrossEntropyRequest(BaseModel):
    true_distribution: List[float]
    predicted_distribution: List[float]


class KLDivergenceRequest(BaseModel):
    p_distribution: List[float]
    q_distribution: List[float]


class BigramAnalysisRequest(BaseModel):
    text: str
    frequency_threshold: int = 5


class WordEmbeddingRequest(BaseModel):
    word: str


class WordSimilarityRequest(BaseModel):
    word1: str
    word2: str


class SentenceSimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str


class WordAlgebraRequest(BaseModel):
    word1: str  # A
    word2: str  # B
    word3: str  # C
    word4: str  # D for comparison (A + B - C compared to D)


# Neural Network related requests
class ActivationFunctionRequest(BaseModel):
    function_name: str  # sigmoid, softmax, relu
    input_values: List[float]


class ConvolutionRequest(BaseModel):
    image: List[List[float]]  # 2D image as nested list
    kernel: List[List[float]]  # 2D kernel as nested list
    stride: int = 1
    padding: int = 0


class PoolingRequest(BaseModel):
    image: List[List[float]]  # 2D image as nested list
    pool_size: int = 2
    stride: int = 2


class LossCalculationRequest(BaseModel):
    predictions: List[List[float]]  # 2D array: batch_size x num_classes
    targets: List[int]  # 1D array: batch_size (class indices)
    loss_type: str  # "cross_entropy" or "mse"


class ModelInferenceRequest(BaseModel):
    model_type: str  # "fcnn" or "cnn"
    input_data: List[List[List[float]]]  # 3D array for images or 2D for flatten data
    model_params: Optional[dict] = None

# Basic training request that defaults to CIFAR-10
class TrainingRequest(BaseModel):
    epochs: int = 1
    dataset: str = "cifar10"  # Default to CIFAR-10
    model_type: str = "assignmentcnn"  # "simple" or "enhanced" CNN
    batch_size: int = 32  # From Module 4 Practical
    learning_rate: float = 0.0005  # From Module 4 Practical
    checkpoint_dir: Optional[str] = "checkpoints"  # Directory to save checkpoints
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    save_checkpoints: bool = True  # Whether to save checkpoints during training


class CheckpointLoadRequest(BaseModel):
    checkpoint_path: str
    device: str = "cpu"


class CheckpointSaveRequest(BaseModel):
    model_path: str
    checkpoint_dir: str = "checkpoints"
    epoch: int
    loss: float
    accuracy: float


class ImagePredictionRequest(BaseModel):
    image: List[List[List[float]]]  # 3D array: [channels, height, width] for 32x32x3 image
    dataset: str = "cifar10"  # Default to CIFAR-10
    model_type: str = "assignmentcnn"  # "simple" or "enhanced" CNN


class BatchPredictionRequest(BaseModel):
    images: List[List[List[List[float]]]]  # 4D array: [batch, channels, height, width]
    dataset: str = "cifar10"  # Default to CIFAR-10
    model_type: str = "assignmentcnn"  # "simple" or "enhanced" CNN


# GAN related requests
class GANTrainingRequest(BaseModel):
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 0.0002
    latent_dim: int = 100
    dataset: str = "mnist"
    save_path: str = "gan_mnist/gan_mnist.pth"


class GANGenerateRequest(BaseModel):
    num_images: int = 4
    latent_dim: int = 100
    dataset: str = "mnist"
    force_retrain: bool = False  # If True, force re-training even if model exists


class GANLoadRequest(BaseModel):
    model_path: str = "gan_mnist/gan_mnist.pth"
    dataset: str = "mnist"


# Energy Model related requests
class EnergyTrainingRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.0001
    dataset: str = "cifar10"
    save_path: Optional[str] = None  # Defaults to sps_genai/checkpoints/energy_cifar/energy_cifar.pth


class EnergyGenerateRequest(BaseModel):
    num_images: int = 4
    num_steps: int = 1000
    dataset: str = "cifar10"
    force_retrain: bool = False


class EnergyLoadRequest(BaseModel):
    model_path: Optional[str] = None  # Defaults to sps_genai/checkpoints/energy_cifar/energy_cifar.pth
    dataset: str = "cifar10"


# Diffusion Model related requests
class DiffusionTrainingRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.0001
    dataset: str = "cifar10"
    num_timesteps: int = 1000
    save_path: Optional[str] = None  # Defaults to sps_genai/checkpoints/diffusion_cifar/diffusion_cifar.pth


class DiffusionGenerateRequest(BaseModel):
    num_images: int = 1
    dataset: str = "cifar10"
    force_retrain: bool = False


class DiffusionLoadRequest(BaseModel):
    model_path: Optional[str] = None  # Defaults to sps_genai/checkpoints/diffusion_cifar/diffusion_cifar.pth
    dataset: str = "cifar10"