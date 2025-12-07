"""
Response models for SPS GenAI API.
Pydantic models for structuring API responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


class EntropyResponse(BaseModel):
    entropy: float


class CrossEntropyResponse(BaseModel):
    cross_entropy: float


class KLDivergenceResponse(BaseModel):
    kl_divergence: float


class BigramAnalysisResponse(BaseModel):
    vocab: List[str]
    bigram_probabilities: Dict[str, Dict[str, float]]


class WordEmbeddingResponse(BaseModel):
    word: str
    embedding: List[float]


class WordSimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float


class SentenceSimilarityResponse(BaseModel):
    sentence1: str
    sentence2: str
    similarity: float


class WordAlgebraResponse(BaseModel):
    expression: str
    comparison_word: str
    similarity: float


class TextGenerationResponse(BaseModel):
    generated_text: str


class BookTextGenerationResponse(BaseModel):
    generated_text: str
    vocab_size: int
    source: str


class BigramMatrixResponse(BaseModel):
    vocab: List[str]
    probability_matrix: List[List[float]]


class VocabularyResponse(BaseModel):
    vocabulary: List[str]
    size: int


class HealthResponse(BaseModel):
    status: str
    spacy_available: bool
    bigram_model_vocab_size: int


# Neural Network related responses
class ActivationFunctionResponse(BaseModel):
    function_name: str
    input_values: List[float]
    output_values: List[float]


class ConvolutionResponse(BaseModel):
    input_shape: List[int]
    kernel_shape: List[int]
    output_shape: List[int]
    output: List[List[float]]
    stride: int
    padding: int


class PoolingResponse(BaseModel):
    input_shape: List[int]
    output_shape: List[int]
    output: List[List[float]]
    pool_size: int
    stride: int


class LossCalculationResponse(BaseModel):
    loss_type: str
    loss_value: float
    batch_size: int


class ModelInferenceResponse(BaseModel):
    model_type: str
    input_shape: List[int]
    predictions: List[List[float]]
    predicted_classes: List[int]


class DeviceInfoResponse(BaseModel):
    device: str
    available_devices: List[str]
    pytorch_version: str


# Generic training and prediction responses (defaults to CIFAR-10)
class TrainingResponse(BaseModel):
    status: str
    dataset: str
    epochs_completed: int
    final_train_accuracy: float
    final_val_accuracy: float
    training_history: Dict[str, List[float]]
    model_path: str
    training_time: float
    checkpoint_dir: Optional[str] = None
    latest_checkpoint: Optional[str] = None
    resumed_from_checkpoint: Optional[str] = None


class PredictionResponse(BaseModel):
    dataset: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    all_probabilities: List[float]


class BatchPredictionResponse(BaseModel):
    dataset: str
    predictions: List[PredictionResponse]
    batch_size: int


class ModelInfoResponse(BaseModel):
    model_type: str
    dataset: str
    classes: List[str]
    input_size: List[int]
    num_classes: int
    architecture: List[str]
    model_exists: bool


class CheckpointResponse(BaseModel):
    status: str
    checkpoint_path: str
    epoch: int
    loss: float
    accuracy: float
    message: str


class CheckpointListResponse(BaseModel):
    checkpoint_dir: str
    available_checkpoints: List[str]
    latest_checkpoint: Optional[str]
    total_checkpoints: int


# GAN related responses
class GANTrainingResponse(BaseModel):
    status: str
    message: str
    epochs_completed: int
    final_generator_loss: float
    final_discriminator_loss: float
    model_path: str
    training_time: float
    sample_images: Optional[List[List[List[float]]]] = None  # 4 sample generated images
    sample_image_urls: Optional[List[str]] = None  # URLs to saved sample images
    sample_base64_images: Optional[List[str]] = None  # Base64 encoded sample images


class GANGenerateResponse(BaseModel):
    status: str
    num_images: int
    images: List[List[List[float]]]  # List of 28x28 images (legacy)
    image_urls: Optional[List[str]] = None  # URLs to saved images
    base64_images: Optional[List[str]] = None  # Base64 encoded images
    message: str


class GANInfoResponse(BaseModel):
    model_type: str = "GAN"
    dataset: str = "MNIST"
    latent_dim: int
    generator_architecture: List[str]
    discriminator_architecture: List[str]
    model_exists: bool


# Energy Model related responses
class EnergyTrainingResponse(BaseModel):
    status: str
    message: str
    epochs_completed: int
    final_loss: float
    model_path: str
    training_time: float
    sample_images: Optional[List[List[List[float]]]] = None
    sample_image_urls: Optional[List[str]] = None
    sample_base64_images: Optional[List[str]] = None


class EnergyGenerateResponse(BaseModel):
    status: str
    num_images: int
    images: List[List[List[float]]]
    image_urls: Optional[List[str]] = None
    base64_images: Optional[List[str]] = None
    message: str


class EnergyInfoResponse(BaseModel):
    model_type: str = "Energy"
    dataset: str = "CIFAR-10"
    architecture: List[str]
    model_exists: bool


# Diffusion Model related responses
class DiffusionTrainingResponse(BaseModel):
    status: str
    message: str
    epochs_completed: int
    final_loss: float
    model_path: str
    training_time: float
    num_timesteps: int
    sample_images: Optional[List[List[List[float]]]] = None
    sample_image_urls: Optional[List[str]] = None
    sample_base64_images: Optional[List[str]] = None


class DiffusionGenerateResponse(BaseModel):
    status: str
    num_images: int
    images: List[List[List[float]]]
    image_urls: Optional[List[str]] = None
    base64_images: Optional[List[str]] = None
    message: str


class DiffusionInfoResponse(BaseModel):
    model_type: str = "Diffusion"
    dataset: str = "CIFAR-10"
    num_timesteps: int
    architecture: List[str]
    model_exists: bool