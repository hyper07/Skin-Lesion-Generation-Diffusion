"""
Neural Networks API Router
Handles all neural network, activation function, and deep learning endpoints.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import torch
import time
import os
import base64
from io import BytesIO
from PIL import Image
from typing import List
from torchinfo import summary

from help_lib.neural_networks import (
    sigmoid,
    softmax,
    relu,
    calculate_cross_entropy_loss,
    calculate_mse_loss,
    apply_convolution_2d,
    apply_max_pooling_2d,
    SimpleFC,
    SimpleCNN,
    AssignmentCNN,
    get_device,
    calculate_output_size
)
from help_lib.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    get_latest_checkpoint,
    get_checkpoint_base_dir,
    get_checkpoint_metadata
)

from models import GAN, get_dataloader, train_gan
from models.gan_models import save_generated_images, images_to_base64
from models.energy_diffusion_models import (
    EnergyModel, EnergyModelTrainer, train_energy_model,
    DiffusionModel, DiffusionTrainer, train_diffusion_model
)


# Import CIFAR-10 utilities from consolidated helpers (no global flag)
try:
    from help_lib.model import train_cifar10_classifier, load_and_predict, get_cifar10_model_info
    from help_lib.data_loader import CIFAR10_CLASSES
except Exception as e:
    print(f"Warning: CIFAR-10 helpers not available: {e}")
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from models.requests import (
    ActivationFunctionRequest,
    ConvolutionRequest,
    PoolingRequest,
    LossCalculationRequest,
    TrainingRequest,
    ImagePredictionRequest,
    BatchPredictionRequest,
    CheckpointLoadRequest,
    CheckpointSaveRequest,
    GANTrainingRequest,
    GANGenerateRequest,
    GANLoadRequest,
    EnergyTrainingRequest,
    EnergyGenerateRequest,
    EnergyLoadRequest,
    DiffusionTrainingRequest,
    DiffusionGenerateRequest,
    DiffusionLoadRequest
)
from models.responses import (
    ActivationFunctionResponse,
    ConvolutionResponse,
    PoolingResponse,
    LossCalculationResponse,
    ModelInferenceResponse,
    DeviceInfoResponse,
    TrainingResponse,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    CheckpointResponse,
    CheckpointListResponse,
    GANTrainingResponse,
    GANGenerateResponse,
    GANInfoResponse,
    EnergyTrainingResponse,
    EnergyGenerateResponse,
    EnergyInfoResponse,
    DiffusionTrainingResponse,
    DiffusionGenerateResponse,
    DiffusionInfoResponse
)

# Create router
router = APIRouter(prefix="/neural-networks", tags=["Neural Networks"])

@router.post("/activation-function", response_model=ActivationFunctionResponse)
def apply_activation_function(request: ActivationFunctionRequest):
    """Apply activation function to input values."""
    try:
        input_array = np.array(request.input_values)
        if request.function_name.lower() == "sigmoid":
            output = sigmoid(input_array).tolist()
        elif request.function_name.lower() == "softmax":
            output = softmax(input_array).tolist()
        elif request.function_name.lower() == "relu":
            output = relu(input_array).tolist()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown activation function: {request.function_name}")
        return ActivationFunctionResponse(
            function_name=request.function_name,
            input_values=request.input_values,
            output_values=output
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying activation function: {str(e)}")

@router.post("/convolution", response_model=ConvolutionResponse)
def apply_convolution(request: ConvolutionRequest):
    """Apply 2D convolution operation."""
    try:
        image = np.array(request.image)
        kernel = np.array(request.kernel)
        output = apply_convolution_2d(image, kernel, request.stride, request.padding)
        return ConvolutionResponse(
            input_shape=list(image.shape),
            kernel_shape=list(kernel.shape),
            output_shape=list(output.shape),
            output=output.tolist(),
            stride=request.stride,
            padding=request.padding
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying convolution: {str(e)}")

@router.post("/pooling", response_model=PoolingResponse)
def apply_pooling(request: PoolingRequest):
    """Apply 2D max pooling operation."""
    try:
        image = np.array(request.image)
        output = apply_max_pooling_2d(image, request.pool_size, request.stride)
        return PoolingResponse(
            input_shape=list(image.shape),
            output_shape=list(output.shape),
            output=output.tolist(),
            pool_size=request.pool_size,
            stride=request.stride
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying pooling: {str(e)}")

@router.post("/calculate-loss", response_model=LossCalculationResponse)
def calculate_loss(request: LossCalculationRequest):
    """Calculate loss for neural network training."""
    try:
        predictions = np.array(request.predictions)
        targets = np.array(request.targets)
        if request.loss_type.lower() == "cross_entropy":
            loss = calculate_cross_entropy_loss(predictions, targets)
        elif request.loss_type.lower() == "mse":
            if len(targets.shape) == 1:
                num_classes = predictions.shape[1]
                targets_one_hot = np.zeros((len(targets), num_classes))
                targets_one_hot[np.arange(len(targets)), targets] = 1
                targets = targets_one_hot
            loss = calculate_mse_loss(predictions, targets)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown loss type: {request.loss_type}")
        return LossCalculationResponse(
            loss_type=request.loss_type,
            loss_value=float(loss),
            batch_size=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating loss: {str(e)}")

@router.get("/device-info", response_model=DeviceInfoResponse)
def get_device_info():
    """Get information about available computing devices."""
    try:
        device = get_device()
        available_devices = []
        available_devices.append("cpu")
        if torch.cuda.is_available():
            available_devices.append("cuda")
        if torch.backends.mps.is_available():
            available_devices.append("mps")
        return DeviceInfoResponse(
            device=str(device),
            available_devices=available_devices,
            pytorch_version=torch.__version__
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="PyTorch not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting device info: {str(e)}")

@router.get("/model-info/{model_type}")
def get_model_info(model_type: str):
    """Get information about available model types."""
    try:
        if model_type.lower() == "fcnn":
            model_info = {
                "model_type": "fcnn",
                "description": "Fully Connected Neural Network",
                "input_format": "Flattened input (e.g., 28*28 = 784 for MNIST)",
                "example_params": {
                    "input_size": 784,
                    "hidden_size": 128,
                    "num_classes": 10
                }
            }
        elif model_type.lower() == "cnn":
            model_info = {
                "model_type": "cnn",
                "description": "Convolutional Neural Network",
                "input_format": "3D tensor (channels, height, width)",
                "example_params": {
                    "num_classes": 10,
                    "input_channels": 3,
                    "input_size": "32x32"
                }
            }
        elif model_type.lower() in ("assignmentcnn"):
            model_info = {
                "model_type": "assignment",
                "description": "AssignmentCNN for 64x64 inputs",
                "input_format": "3D tensor (channels, height, width)",
                "example_params": {
                    "num_classes": 10,
                    "input_channels": 3,
                    "input_size": "64x64"
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        return model_info
    except ImportError:
        raise HTTPException(status_code=503, detail="PyTorch not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/output-size-calculator")
def calculate_layer_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
):
    """Calculate output size after convolution or pooling layer."""
    try:
        output_size = calculate_output_size(input_size, kernel_size, stride, padding)
        return {
            "input_size": input_size,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "output_size": output_size,
            "formula": "(input_size + 2*padding - (kernel_size-1) - 1) // stride + 1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating output size: {str(e)}")

# Generic Training Endpoint (defaults to CIFAR-10)
@router.post("/train", response_model=TrainingResponse)
def train_model(request: TrainingRequest):
    """Train a classifier model with checkpoint support. Defaults to CIFAR-10 dataset."""
    # Default to CIFAR-10 if no dataset specified
    dataset = request.dataset.lower()
    
    if dataset != "cifar10":
        raise HTTPException(status_code=400, detail="Currently only CIFAR-10 dataset is supported")
    
    try:
        start_time = time.time()
        
        # Prepare checkpoint parameters
        checkpoint_params = {
            'checkpoint_dir': request.checkpoint_dir,
            'save_checkpoints': request.save_checkpoints,
            'resume_from_checkpoint': request.resume_from_checkpoint
        }
        
        history = train_cifar10_classifier(
            epochs=request.epochs,
            model_type=request.model_type,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            **checkpoint_params
        )
        
        training_time = time.time() - start_time
        
        # Format training history to match TrainingResponse model
        # Remove non-list items and ensure all values are lists of floats
        formatted_history = {}
        for key, value in history.items():
            if isinstance(value, list):
                # Convert to list of floats if needed
                formatted_history[key] = [float(v) for v in value]
            elif key == 'epochs':
                # Convert epochs to a list for compatibility
                formatted_history[key] = [float(i+1) for i in range(value)]
        
        # Get checkpoint information
        latest_checkpoint = None
        if request.save_checkpoints and request.checkpoint_dir:
            latest_checkpoint = get_latest_checkpoint(request.checkpoint_dir)
            if latest_checkpoint:
                latest_checkpoint = os.path.basename(latest_checkpoint)
        
        return TrainingResponse(
            status="completed",
            dataset="cifar10",
            epochs_completed=request.epochs,
            final_train_accuracy=history['train_acc'][-1] if history['train_acc'] else 0.0,
            final_val_accuracy=history['val_acc'][-1] if history['val_acc'] else 0.0,
            training_history=formatted_history,
            model_path="../models/cifar10_classifier.pth",
            training_time=training_time,
            checkpoint_dir=request.checkpoint_dir if request.save_checkpoints else None,
            latest_checkpoint=latest_checkpoint,
            resumed_from_checkpoint=request.resume_from_checkpoint
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.post("/train/resume", response_model=TrainingResponse)
def resume_training(
    epochs: int = 10,
    checkpoint_dir: str = "checkpoints",
    dataset: str = "cifar10",
    model_type: str = "assignmentcnn"
):
    """Resume training from the latest checkpoint."""
    try:
        # Find latest checkpoint
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No checkpoints found in {checkpoint_dir}. Use /train endpoint to start fresh training."
            )
        
        # Create training request with checkpoint resumption
        request = TrainingRequest(
            epochs=epochs,
            dataset=dataset,
            model_type=model_type,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=latest_checkpoint,
            save_checkpoints=True
        )
        
        # Use the existing train_model function
        return train_model(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming training: {str(e)}")


@router.get("/training/status")
def get_training_status(checkpoint_dir: str = "checkpoints"):
    """Get current training status from checkpoints."""
    try:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint is None:
            return {
                "status": "no_training",
                "message": "No checkpoints found. Training not started.",
                "checkpoint_dir": checkpoint_dir,
                "can_resume": False
            }
        
        # Load checkpoint to get training information
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        accuracy = checkpoint.get('accuracy', 0.0)
        
        return {
            "status": "training_available",
            "message": f"Training can be resumed from epoch {epoch}",
            "checkpoint_dir": checkpoint_dir,
            "latest_checkpoint": os.path.basename(latest_checkpoint),
            "last_epoch": epoch,
            "last_loss": float(loss),
            "last_accuracy": float(accuracy),
            "can_resume": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImagePredictionRequest):
    """Make prediction on an image. Defaults to CIFAR-10 dataset."""
    # Default to CIFAR-10 if no dataset specified
    dataset = request.dataset.lower()
    
    if dataset != "cifar10":
        raise HTTPException(status_code=400, detail="Currently only CIFAR-10 dataset is supported")
    
    try:
        # Convert image to tensor
        image_array = np.array(request.image, dtype=np.float32)
        image_tensor = torch.from_numpy(image_array)
        
        # Make prediction
        class_name, confidence, probabilities = load_and_predict(image_tensor, model_type=request.model_type)
        
        # Create class probabilities dictionary
        class_probs = {CIFAR10_CLASSES[i]: float(probabilities[i]) for i in range(len(CIFAR10_CLASSES))}
        
        return PredictionResponse(
            dataset="cifar10",
            predicted_class=class_name,
            confidence=float(confidence),
            class_probabilities=class_probs,
            all_probabilities=probabilities.tolist()
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_images_batch(request: BatchPredictionRequest):
    """Make predictions on multiple images. Defaults to CIFAR-10 dataset."""
    # Default to CIFAR-10 if no dataset specified
    dataset = request.dataset.lower()
    
    if dataset != "cifar10":
        raise HTTPException(status_code=400, detail="Currently only CIFAR-10 dataset is supported")
    
    try:
        predictions = []
        
        for image_data in request.images:
            # Convert image to tensor
            image_array = np.array(image_data, dtype=np.float32)
            image_tensor = torch.from_numpy(image_array)
            
            # Make prediction
            class_name, confidence, probabilities = load_and_predict(image_tensor, model_type=request.model_type)
            
            # Create class probabilities dictionary
            class_probs = {CIFAR10_CLASSES[i]: float(probabilities[i]) for i in range(len(CIFAR10_CLASSES))}
            
            predictions.append(PredictionResponse(
                dataset="cifar10",
                predicted_class=class_name,
                confidence=float(confidence),
                class_probabilities=class_probs,
                all_probabilities=probabilities.tolist()
            ))
        
        return BatchPredictionResponse(
            dataset="cifar10",
            predictions=predictions,
            batch_size=len(predictions)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {str(e)}")

@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info_endpoint():
    """Get information about the trained model. Defaults to CIFAR-10."""
    try:
        try:
            model_info = get_cifar10_model_info("simple")  # Default to simple model info
        except Exception:
            # Provide basic info even when helpers aren't available
            model_info = {
                'model_type': 'SimpleCNN',
                'dataset': 'CIFAR-10',
                'classes': CIFAR10_CLASSES,
                'input_size': [3, 32, 32],
                'num_classes': 10,
                'architecture': ['Conv2d', 'ReLU', 'MaxPool2d', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear', 'ReLU', 'Linear']
            }
        model_exists = os.path.exists('models/cifar10_classifier.pth')
        
        return ModelInfoResponse(
            model_type=model_info['model_type'],
            dataset=model_info['dataset'],
            classes=model_info['classes'],
            input_size=model_info['input_size'],
            num_classes=model_info['num_classes'],
            architecture=model_info['architecture'],
            model_exists=model_exists
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


# Checkpoint Management Endpoints
@router.get("/checkpoints", response_model=CheckpointListResponse)
def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """List all available checkpoints in a directory."""
    try:
        if not os.path.exists(checkpoint_dir):
            return CheckpointListResponse(
                checkpoint_dir=checkpoint_dir,
                available_checkpoints=[],
                latest_checkpoint=None,
                total_checkpoints=0
            )
        
        # Get all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        checkpoint_files.sort()
        
        # Get latest checkpoint
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        latest_name = os.path.basename(latest_checkpoint) if latest_checkpoint else None
        
        return CheckpointListResponse(
            checkpoint_dir=checkpoint_dir,
            available_checkpoints=checkpoint_files,
            latest_checkpoint=latest_name,
            total_checkpoints=len(checkpoint_files)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing checkpoints: {str(e)}")


@router.post("/checkpoints/load", response_model=CheckpointResponse)
def load_checkpoint_endpoint(request: CheckpointLoadRequest):
    """Load a checkpoint and return its information."""
    try:
        if not os.path.exists(request.checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")
        
        # Load checkpoint to get information (without model/optimizer)
        checkpoint = torch.load(request.checkpoint_path, map_location=request.device)
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        accuracy = checkpoint.get('accuracy', 0.0)
        
        return CheckpointResponse(
            status="loaded",
            checkpoint_path=request.checkpoint_path,
            epoch=epoch,
            loss=float(loss),
            accuracy=float(accuracy),
            message=f"Checkpoint loaded successfully from epoch {epoch}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading checkpoint: {str(e)}")


@router.get("/checkpoints/latest")
def get_latest_checkpoint_info(checkpoint_dir: str = "checkpoints"):
    """Get information about the latest checkpoint."""
    try:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint is None:
            raise HTTPException(status_code=404, detail=f"No checkpoints found in {checkpoint_dir}")
        
        # Load checkpoint to get information
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        accuracy = checkpoint.get('accuracy', 0.0)
        
        return CheckpointResponse(
            status="found",
            checkpoint_path=latest_checkpoint,
            epoch=epoch,
            loss=float(loss),
            accuracy=float(accuracy),
            message=f"Latest checkpoint from epoch {epoch}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latest checkpoint: {str(e)}")


@router.delete("/checkpoints")
def delete_checkpoints(checkpoint_dir: str = "checkpoints", confirm: bool = False):
    """Delete all checkpoints in a directory."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Please set confirm=true to delete all checkpoints"
            )
        
        if not os.path.exists(checkpoint_dir):
            return {"message": f"Checkpoint directory {checkpoint_dir} does not exist"}
        
        # Delete all .pth files
        deleted_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth'):
                file_path = os.path.join(checkpoint_dir, filename)
                os.remove(file_path)
                deleted_files.append(filename)
        
        return {
            "message": f"Deleted {len(deleted_files)} checkpoint files",
            "deleted_files": deleted_files,
            "checkpoint_dir": checkpoint_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting checkpoints: {str(e)}")


# GAN Endpoints
@router.post("/gan/train", response_model=GANTrainingResponse)
def train_gan_model(request: GANTrainingRequest):
    """Train a GAN model on specified dataset. If model exists, use it; otherwise train it."""
    if GAN is None:
        raise HTTPException(status_code=503, detail="GAN implementation not available")

    try:
        model_path = f"checkpoints/gan_{request.dataset}/gan_{request.dataset}.pth"
        model_exists = os.path.exists(model_path)

        start_time = time.time()
        training_time = 0.0
        epochs_completed = 0
        final_g_loss = 0.0
        final_d_loss = 0.0

        if model_exists:
            print(f"Trained GAN model found for {request.dataset.upper()}. Using existing model...")
            message = f"Using existing trained GAN model on {request.dataset.upper()}"
            
            # Try to load losses from the latest checkpoint
            try:
                from help_lib.checkpoints import load_checkpoint
                checkpoint_dir = f"checkpoints/gan_{request.dataset}"
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    # Create dummy models/optimizers just to load checkpoint info
                    dummy_gan = GAN(dataset=request.dataset, latent_dim=request.latent_dim)
                    models = {'generator': dummy_gan.generator, 'discriminator': dummy_gan.discriminator}
                    optimizers = {'generator': dummy_gan.g_optimizer, 'discriminator': dummy_gan.d_optimizer}
                    
                    _, losses, _ = load_checkpoint(models, optimizers, latest_checkpoint, device=dummy_gan.device)
                    final_g_loss = losses.get('g_loss', 0.0)
                    final_d_loss = losses.get('d_loss', 0.0)
                else:
                    final_g_loss = 0.0
                    final_d_loss = 0.0
            except Exception as e:
                print(f"Could not load losses from checkpoint: {e}")
                final_g_loss = 0.0
                final_d_loss = 0.0
        else:
            print(f"Training GAN on {request.dataset.upper()} for {request.epochs} epochs...")

            # Use the new train_gan function
            trained_gan, d_losses, g_losses = train_gan(
                dataset=request.dataset,
                epochs=request.epochs,
                batch_size=request.batch_size,
                latent_dim=request.latent_dim,
                learning_rate=request.learning_rate
            )

            training_time = time.time() - start_time
            epochs_completed = request.epochs
            message = f"GAN training on {request.dataset.upper()} completed successfully"
            final_g_loss = g_losses[-1] if g_losses else 0.0
            final_d_loss = d_losses[-1] if d_losses else 0.0

        print("Generating 4 sample images...")
        gan = GAN(dataset=request.dataset, latent_dim=request.latent_dim)
        gan.load_models(model_path)

        # Generate 4 sample images
        sample_images = gan.generate_images(4)

        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save sample images to disk
        sample_image_paths = save_generated_images(sample_images, request.dataset.lower(), timestamp, prefix="sample")

        # Convert images to base64 for API response
        sample_base64_images = images_to_base64(sample_images)

        # Convert to list format for JSON response
        sample_images_list = []
        for img in sample_images:
            # Denormalize from [-1, 1] to [0, 1] and convert to list
            img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
            sample_images_list.append(img_denorm.squeeze().tolist())

        return GANTrainingResponse(
            status="completed",
            message=message,
            epochs_completed=epochs_completed,
            final_generator_loss=final_g_loss,
            final_discriminator_loss=final_d_loss,
            model_path=model_path,
            training_time=training_time,
            sample_images=sample_images_list,
            sample_image_urls=sample_image_paths,
            sample_base64_images=sample_base64_images
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training GAN: {str(e)}")


@router.post("/gan/generate", response_model=GANGenerateResponse)
def generate_gan_images(request: GANGenerateRequest):
    """Generate images using trained GAN model."""
    if GAN is None:
        raise HTTPException(status_code=503, detail="GAN implementation not available")

    try:
        # Initialize GAN with dataset
        gan = GAN(dataset=request.dataset, latent_dim=request.latent_dim)

        # Determine model path
        model_path = f"checkpoints/gan_{request.dataset}/gan_{request.dataset}.pth"

        # Check if we should force retrain or use existing model
        if request.force_retrain:
            # Force retrain the model
            print(f"Forcing retrain of GAN on {request.dataset.upper()} for {5} epochs...")
            train_gan(dataset=request.dataset, epochs=5, batch_size=128, latent_dim=request.latent_dim)
            gan.load_models(model_path)
            message = f"Generated {request.num_images} images using newly retrained GAN model on {request.dataset.upper()} (5 epochs)"
        else:
            # Try to use existing model, train if none exists
            try:
                gan.load_models(model_path)
                message = f"Generated {request.num_images} images using trained GAN model on {request.dataset.upper()}"
            except FileNotFoundError:
                # Train a quick model if none exists
                print(f"No trained GAN model found for {request.dataset}. Training a quick model...")
                train_gan(dataset=request.dataset, epochs=5, batch_size=128, latent_dim=request.latent_dim)
                gan.load_models(model_path)
                message = f"Generated {request.num_images} images using newly trained GAN model on {request.dataset.upper()} (5 epochs)"

        # Generate images
        generated_images = gan.generate_images(request.num_images)

        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save images to disk
        image_paths = save_generated_images(generated_images, request.dataset.lower(), timestamp)

        # Convert images to base64 for API response
        base64_images = images_to_base64(generated_images)

        # Convert to list format for JSON response (legacy)
        images_list = []
        for img in generated_images:
            # Denormalize from [-1, 1] to [0, 1] and convert to list
            img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
            images_list.append(img_denorm.squeeze().tolist())

        return GANGenerateResponse(
            status="success",
            num_images=request.num_images,
            images=images_list,  # Legacy format
            image_urls=image_paths,  # File paths to saved images
            base64_images=base64_images,  # Base64 encoded images for display
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating GAN images: {str(e)}")


@router.get("/gan/info", response_model=GANInfoResponse)
def get_gan_info(dataset: str = "mnist"):
    """Get information about the GAN model."""
    if GAN is None:
        raise HTTPException(status_code=503, detail="GAN implementation not available")

    try:
        model_path = f"checkpoints/gan_{dataset}/gan_{dataset}.pth"
        model_exists = os.path.exists(model_path)

        # Instantiate GAN to get model architectures
        gan = GAN(dataset=dataset.lower())

        # Load trained model if it exists
        if model_exists:
            try:
                gan.load_models(model_path)
            except Exception as e:
                # If loading fails, continue with instantiated model
                pass

        # Get input sizes based on dataset
        if dataset.lower() == "mnist":
            img_size = 28
            channels = 1
            gen_input_size = (1, 100)
            disc_input_size = (1, 1, 28, 28)
        elif dataset.lower() == "cifar":
            img_size = 32
            channels = 3
            gen_input_size = (1, 100)
            disc_input_size = (1, 3, 32, 32)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")

        # Get architecture using torchinfo summary
        generator_summary = summary(gan.generator, input_size=gen_input_size, device='cpu', verbose=0)
        discriminator_summary = summary(gan.discriminator, input_size=disc_input_size, device='cpu', verbose=0)

        # Convert summary to list of strings (split by lines)
        generator_arch = [line.strip() for line in str(generator_summary).split('\n') if line.strip()]
        discriminator_arch = [line.strip() for line in str(discriminator_summary).split('\n') if line.strip()]

        return GANInfoResponse(
            model_type="GAN",
            dataset=dataset.upper(),
            latent_dim=100,
            generator_architecture=generator_arch,
            discriminator_architecture=discriminator_arch,
            model_exists=model_exists
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting GAN info: {str(e)}")


@router.post("/gan/load")
def load_gan_model(request: GANLoadRequest):
    """Load a trained GAN model."""
    if GAN is None:
        raise HTTPException(status_code=503, detail="GAN implementation not available")

    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"GAN model not found at {request.model_path}")

        # Initialize GAN and load model
        gan = GAN(dataset=request.dataset)
        gan.load_models(request.model_path)

        return {
            "status": "loaded",
            "message": f"GAN model loaded successfully from {request.model_path}",
            "model_path": request.model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading GAN model: {str(e)}")


# Energy Model Endpoints
@router.post("/energy/train", response_model=EnergyTrainingResponse)
def train_energy_model_endpoint(request: EnergyTrainingRequest):
    """Train an Energy Model on CIFAR-10."""
    try:
        start_time = time.time()
        training_time = 0.0
        epochs_completed = 0
        final_loss = 0.0

        if request.dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Energy Model currently only supports CIFAR-10")

        # Set default paths if not provided
        if request.save_path is None:
            base_dir = get_checkpoint_base_dir()
            request.save_path = os.path.join(base_dir, "energy_cifar", "energy_cifar.pth")
        model_path = request.save_path
        model_exists = os.path.exists(model_path)

        if model_exists:
            print(f"Trained Energy Model found. Using existing model...")
            message = f"Using existing trained Energy Model on CIFAR-10"
            
            try:
                checkpoint_dir = os.path.dirname(model_path)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    model = EnergyModel(input_channels=3, img_size=32)
                    trainer = EnergyModelTrainer(model, learning_rate=request.learning_rate)
                    models = {'energy_model': trainer.model}
                    optimizers = {'energy_optimizer': trainer.optimizer}
                    _, losses, _ = load_checkpoint(models, optimizers, latest_checkpoint, device=trainer.device)
                    final_loss = losses.get('loss', 0.0)
            except Exception as e:
                print(f"Could not load losses from checkpoint: {e}")
                final_loss = 0.0
        else:
            print(f"Training Energy Model on CIFAR-10 for {request.epochs} epochs...")
            trainer, losses_history = train_energy_model(
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                save_path=request.save_path
            )
            training_time = time.time() - start_time
            epochs_completed = request.epochs
            message = f"Energy Model training on CIFAR-10 completed successfully"
            final_loss = losses_history[-1] if losses_history else 0.0

        # Generate sample images (only if model exists)
        sample_images = None
        sample_image_paths = []
        sample_base64_images = []
        sample_images_list = []
        
        if model_exists:
            print("Generating 4 sample images...")
            model = EnergyModel(input_channels=3, img_size=32)
            trainer = EnergyModelTrainer(model, learning_rate=request.learning_rate)
            # Try to load from latest_checkpoint.pth first, then fallback to model_path
            checkpoint_dir = os.path.dirname(model_path)
            latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint):
                checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dicts']['energy_model'])
            else:
                checkpoint = torch.load(model_path, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            sample_images = trainer.generate_samples(4, num_steps=100)
            
            # Create timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save sample images to disk
            sample_image_paths = save_generated_images(sample_images, "energy_cifar", timestamp, prefix="sample")
            
            # Convert images to base64
            sample_base64_images = images_to_base64(sample_images)
            
            # Convert to list format for JSON response
            for img in sample_images:
                img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
                if img_denorm.shape[0] == 3:
                    gray = (0.299 * img_denorm[0] + 0.587 * img_denorm[1] + 0.114 * img_denorm[2]).clamp(0, 1)
                    sample_images_list.append(gray.numpy().tolist())
                else:
                    sample_images_list.append(img_denorm.squeeze(0).numpy().tolist())
        else:
            # After training, generate samples
            print("Generating 4 sample images...")
            sample_images = trainer.generate_samples(4, num_steps=100)
            
            # Create timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save sample images to disk
            sample_image_paths = save_generated_images(sample_images, "energy_cifar", timestamp, prefix="sample")
            
            # Convert images to base64
            sample_base64_images = images_to_base64(sample_images)
            
            # Convert to list format for JSON response
            for img in sample_images:
                img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
                sample_images_list.append(img_denorm.permute(1, 2, 0).numpy().tolist())

        return EnergyTrainingResponse(
            status="completed",
            message=message,
            epochs_completed=epochs_completed,
            final_loss=final_loss,
            model_path=model_path,
            training_time=training_time,
            sample_images=sample_images_list,
            sample_image_urls=sample_image_paths,
            sample_base64_images=sample_base64_images
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training Energy Model: {str(e)}")


@router.post("/energy/generate", response_model=EnergyGenerateResponse)
def generate_energy_images(request: EnergyGenerateRequest):
    """Generate images using trained Energy Model."""
    try:
        if request.dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Energy Model currently only supports CIFAR-10")

        # Use checkpoint base directory
        base_dir = get_checkpoint_base_dir()
        model_path = os.path.join(base_dir, "energy_cifar", "energy_cifar.pth")

        if request.force_retrain:
            print(f"Forcing retrain of Energy Model for 5 epochs...")
            train_energy_model(epochs=5, batch_size=128)
            message = f"Generated {request.num_images} images using newly retrained Energy Model (5 epochs)"
        else:
            if not os.path.exists(model_path):
                print(f"No trained Energy Model found. Training a quick model...")
                train_energy_model(epochs=5, batch_size=128)
                message = f"Generated {request.num_images} images using newly trained Energy Model (5 epochs)"
            else:
                message = f"Generated {request.num_images} images using trained Energy Model"

        # Load model and generate
        model = EnergyModel(input_channels=3, img_size=32)
        trainer = EnergyModelTrainer(model)
        # Try to load from latest_checkpoint.pth first, then fallback to model_path
        checkpoint_dir = os.path.dirname(model_path)
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dicts']['energy_model'])
        else:
            checkpoint = torch.load(model_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        generated_images = trainer.generate_samples(request.num_images, num_steps=request.num_steps)

        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save images to disk
        image_paths = save_generated_images(generated_images, "energy_cifar", timestamp)

        # Convert images to base64
        base64_images = images_to_base64(generated_images)

        # Convert to list format for JSON response
        images_list = []
        for img in generated_images:
            img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
            if img_denorm.shape[0] == 3:
                gray = (0.299 * img_denorm[0] + 0.587 * img_denorm[1] + 0.114 * img_denorm[2]).clamp(0, 1)
                images_list.append(gray.numpy().tolist())
            else:
                images_list.append(img_denorm.squeeze(0).numpy().tolist())

        return EnergyGenerateResponse(
            status="success",
            num_images=request.num_images,
            images=images_list,
            image_urls=image_paths,
            base64_images=base64_images,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Energy Model images: {str(e)}")


@router.get("/energy/info", response_model=EnergyInfoResponse)
def get_energy_info(dataset: str = "cifar10"):
    """Get information about the Energy Model."""
    try:
        if dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Energy Model currently only supports CIFAR-10")

        # Use checkpoint base directory
        base_dir = get_checkpoint_base_dir()
        model_path = os.path.join(base_dir, "energy_cifar", "energy_cifar.pth")
        model_exists = os.path.exists(model_path)

        model = EnergyModel(input_channels=3, img_size=32)
        architecture = [str(model)]

        return EnergyInfoResponse(
            model_type="Energy",
            dataset="CIFAR-10",
            architecture=architecture,
            model_exists=model_exists
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Energy Model info: {str(e)}")


@router.post("/energy/load")
def load_energy_model(request: EnergyLoadRequest):
    """Load a trained Energy Model."""
    try:
        # Set default path if not provided
        if request.model_path is None:
            base_dir = get_checkpoint_base_dir()
            request.model_path = os.path.join(base_dir, "energy_cifar", "energy_cifar.pth")
        
        # Try to load from latest_checkpoint.pth first, then fallback to model_path
        checkpoint_dir = os.path.dirname(request.model_path)
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            model = EnergyModel(input_channels=3, img_size=32)
            trainer = EnergyModelTrainer(model)
            trainer.model.load_state_dict(checkpoint['model_state_dicts']['energy_model'])
        elif os.path.exists(request.model_path):
            checkpoint = torch.load(request.model_path, map_location='cpu')
            model = EnergyModel(input_channels=3, img_size=32)
            trainer = EnergyModelTrainer(model)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise HTTPException(status_code=404, detail=f"Energy Model not found at {request.model_path} or {latest_checkpoint}")

        return {
            "status": "loaded",
            "message": f"Energy Model loaded successfully from {request.model_path}",
            "model_path": request.model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Energy Model: {str(e)}")


# Diffusion Model Endpoints
@router.post("/diffusion/train", response_model=DiffusionTrainingResponse)
def train_diffusion_model_endpoint(request: DiffusionTrainingRequest):
    """Train a Diffusion Model on CIFAR-10."""
    try:
        start_time = time.time()
        training_time = 0.0
        epochs_completed = 0
        final_loss = 0.0

        if request.dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Diffusion Model currently only supports CIFAR-10")

        # Set default paths if not provided
        if request.save_path is None:
            base_dir = get_checkpoint_base_dir()
            request.save_path = os.path.join(base_dir, "diffusion_cifar", "diffusion_cifar.pth")
        model_path = request.save_path
        model_exists = os.path.exists(model_path)

        if model_exists:
            print(f"Trained Diffusion Model found. Using existing model...")
            message = f"Using existing trained Diffusion Model on CIFAR-10"
            
            try:
                checkpoint_dir = os.path.dirname(model_path)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    model = DiffusionModel(input_channels=3, img_size=32)
                    trainer = DiffusionTrainer(model, learning_rate=request.learning_rate)
                    models = {'diffusion_model': trainer.model.network}
                    optimizers = {'diffusion_optimizer': trainer.optimizer}
                    _, losses, _ = load_checkpoint(models, optimizers, latest_checkpoint, device=trainer.device)
                    final_loss = losses.get('loss', 0.0)
            except Exception as e:
                print(f"Could not load losses from checkpoint: {e}")
                final_loss = 0.0
        else:
            print(f"Training Diffusion Model on CIFAR-10 for {request.epochs} epochs...")
            trainer, losses_history = train_diffusion_model(
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                save_path=request.save_path
            )
            training_time = time.time() - start_time
            epochs_completed = request.epochs
            message = f"Diffusion Model training on CIFAR-10 completed successfully"
            final_loss = losses_history[-1] if losses_history else 0.0

        # Generate sample images (only if model exists)
        sample_images = None
        sample_image_paths = []
        sample_base64_images = []
        sample_images_list = []
        
        if model_exists:
            print("Generating 4 sample images...")
            model = DiffusionModel(input_channels=3, img_size=32)
            trainer = DiffusionTrainer(model)
            # Try to load from latest_checkpoint.pth first, then fallback to model_path
            checkpoint_dir = os.path.dirname(model_path)
            latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint):
                checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
                # Load the network state dict
                trainer.model.network.load_state_dict(checkpoint['model_state_dicts']['diffusion_model'])
                # Load normalizer from metadata or final model
                metadata = get_checkpoint_metadata(checkpoint_dir)
                if metadata is None:
                    # Fallback: try to load from final model file
                    final_model = torch.load(model_path, map_location=trainer.device)
                    if 'normalizer_mean' in final_model:
                        trainer.model.set_normalizer(final_model['normalizer_mean'], final_model['normalizer_std'])
            else:
                checkpoint = torch.load(model_path, map_location=trainer.device)
                trainer.model.network.load_state_dict(checkpoint['model_state_dict'])
                if 'normalizer_mean' in checkpoint:
                    trainer.model.set_normalizer(checkpoint['normalizer_mean'], checkpoint['normalizer_std'])
            
            sample_images = trainer.model.generate(4, diffusion_steps=20, image_size=32)
        else:
            # After training, generate samples
            print("Generating 4 sample images...")
            sample_images = trainer.model.generate(4, diffusion_steps=20, image_size=32)
        
        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save sample images to disk
        sample_image_paths = save_generated_images(sample_images, "diffusion_cifar", timestamp, prefix="sample")

        # Convert images to base64
        sample_base64_images = images_to_base64(sample_images)

        # Convert to list format for JSON response
        for img in sample_images:
            img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
            if img_denorm.shape[0] == 3:
                gray = (0.299 * img_denorm[0] + 0.587 * img_denorm[1] + 0.114 * img_denorm[2]).clamp(0, 1)
                sample_images_list.append(gray.numpy().tolist())
            else:
                sample_images_list.append(img_denorm.squeeze(0).numpy().tolist())

        return DiffusionTrainingResponse(
            status="completed",
            message=message,
            epochs_completed=epochs_completed,
            final_loss=final_loss,
            model_path=model_path,
            training_time=training_time,
            num_timesteps=1000,  # Diffusion steps for sampling
            sample_images=sample_images_list,
            sample_image_urls=sample_image_paths,
            sample_base64_images=sample_base64_images
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training Diffusion Model: {str(e)}")


@router.post("/diffusion/generate", response_model=DiffusionGenerateResponse)
def generate_diffusion_images(request: DiffusionGenerateRequest):
    """Generate images using trained Diffusion Model."""
    try:
        if request.dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Diffusion Model currently only supports CIFAR-10")

        # Use checkpoint base directory
        base_dir = get_checkpoint_base_dir()
        model_path = os.path.join(base_dir, "diffusion_cifar", "diffusion_cifar.pth")

        if request.force_retrain:
            print(f"Forcing retrain of Diffusion Model for 5 epochs...")
            train_diffusion_model(epochs=5, batch_size=128)
            message = f"Generated {request.num_images} images using newly retrained Diffusion Model (5 epochs)"
        else:
            if not os.path.exists(model_path):
                print(f"No trained Diffusion Model found. Training a quick model...")
                train_diffusion_model(epochs=5, batch_size=128)
                message = f"Generated {request.num_images} images using newly trained Diffusion Model (5 epochs)"
            else:
                message = f"Generated {request.num_images} images using trained Diffusion Model"

        # Load model and generate
        model = DiffusionModel(input_channels=3, img_size=32)
        trainer = DiffusionTrainer(model)
        # Try to load from latest_checkpoint.pth first, then fallback to model_path
        checkpoint_dir = os.path.dirname(model_path)
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
            # Load the network state dict
            trainer.model.network.load_state_dict(checkpoint['model_state_dicts']['diffusion_model'])
            # Load normalizer from metadata or final model
            metadata = get_checkpoint_metadata(checkpoint_dir)
            if metadata is None:
                # Fallback: try to load from final model file
                final_model = torch.load(model_path, map_location=trainer.device)
                if 'normalizer_mean' in final_model:
                    trainer.model.set_normalizer(final_model['normalizer_mean'], final_model['normalizer_std'])
        else:
            checkpoint = torch.load(model_path, map_location=trainer.device)
            trainer.model.network.load_state_dict(checkpoint['model_state_dict'])
            if 'normalizer_mean' in checkpoint:
                trainer.model.set_normalizer(checkpoint['normalizer_mean'], checkpoint['normalizer_std'])
        
        generated_images = trainer.model.generate(request.num_images, diffusion_steps=20, image_size=32)

        # Create timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save images to disk
        image_paths = save_generated_images(generated_images, "diffusion_cifar", timestamp)

        # Convert images to base64
        base64_images = images_to_base64(generated_images)

        # Convert to list format for JSON response
        images_list = []
        for img in generated_images:
            img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
            if img_denorm.shape[0] == 3:
                gray = (0.299 * img_denorm[0] + 0.587 * img_denorm[1] + 0.114 * img_denorm[2]).clamp(0, 1)
                images_list.append(gray.numpy().tolist())
            else:
                images_list.append(img_denorm.squeeze(0).numpy().tolist())

        return DiffusionGenerateResponse(
            status="success",
            num_images=request.num_images,
            images=images_list,
            image_urls=image_paths,
            base64_images=base64_images,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Diffusion Model images: {str(e)}")


@router.get("/diffusion/info", response_model=DiffusionInfoResponse)
def get_diffusion_info(dataset: str = "cifar10"):
    """Get information about the Diffusion Model."""
    try:
        if dataset != "cifar10":
            raise HTTPException(status_code=400, detail="Diffusion Model currently only supports CIFAR-10")

        # Use checkpoint base directory
        base_dir = get_checkpoint_base_dir()
        model_path = os.path.join(base_dir, "diffusion_cifar", "diffusion_cifar.pth")
        model_exists = os.path.exists(model_path)

        model = DiffusionModel(input_channels=3, img_size=32)
        architecture = [str(model)]

        return DiffusionInfoResponse(
            model_type="Diffusion",
            dataset="CIFAR-10",
            num_timesteps=1000,  # Default timesteps
            architecture=architecture,
            model_exists=model_exists
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Diffusion Model info: {str(e)}")


@router.post("/diffusion/load")
def load_diffusion_model(request: DiffusionLoadRequest):
    """Load a trained Diffusion Model."""
    try:
        # Set default path if not provided
        if request.model_path is None:
            base_dir = get_checkpoint_base_dir()
            request.model_path = os.path.join(base_dir, "diffusion_cifar", "diffusion_cifar.pth")
        
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Diffusion Model not found at {request.model_path}")

        # Try to load from latest_checkpoint.pth first, then fallback to model_path
        checkpoint_dir = os.path.dirname(request.model_path)
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            model = DiffusionModel(input_channels=3, img_size=32)
            trainer = DiffusionTrainer(model)
            trainer.model.network.load_state_dict(checkpoint['model_state_dicts']['diffusion_model'])
            # Load normalizer from metadata or final model
            metadata = get_checkpoint_metadata(checkpoint_dir)
            if metadata is None:
                final_model = torch.load(request.model_path, map_location='cpu')
                if 'normalizer_mean' in final_model:
                    trainer.model.set_normalizer(final_model['normalizer_mean'], final_model['normalizer_std'])
        else:
            checkpoint = torch.load(request.model_path, map_location='cpu')
            model = DiffusionModel(input_channels=3, img_size=32)
            trainer = DiffusionTrainer(model)
            trainer.model.network.load_state_dict(checkpoint['model_state_dict'])
            if 'normalizer_mean' in checkpoint:
                trainer.model.set_normalizer(checkpoint['normalizer_mean'], checkpoint['normalizer_std'])

        return {
            "status": "loaded",
            "message": f"Diffusion Model loaded successfully from {request.model_path}",
            "model_path": request.model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Diffusion Model: {str(e)}")
