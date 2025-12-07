import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from typing import Optional, Dict, Any, Tuple
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from models.cnn_models import CNNFactory
from .neural_networks import AssignmentCNN
from .data_loader import get_cifar10_loaders, CIFAR10_CLASSES
from .trainer import train_model as run_training


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Return a model by name using the shared CNNFactory.
    Supported names: 'FCNN' (alias of simple CNN), 'CNN' (simple), 'EnhancedCNN', 'Flexible'.
    Additional kwargs are passed to the underlying model constructor.
    """
    name = (model_name or '').lower()
    if name in ('fcnn', 'cnn', 'simple'):
        return CNNFactory.create_model('simple', **kwargs)
    if name in ('enhanced', 'enhancedcnn'):
        return CNNFactory.create_model('enhanced', **kwargs)
    if name in ('flexible', 'custom'):
        return CNNFactory.create_model('flexible', **kwargs)
    if name in ('assignment', 'assignmentcnn'):
        # Default AssignmentCNN expects 64x64 inputs and 10 classes
        # kwargs may include custom parameters if needed
        return AssignmentCNN()
    # Example of swapping in torchvision models if desired
    if name == 'resnet18':
        model = resnet18(num_classes=kwargs.get('num_classes', 10))
        return model
    raise ValueError(f"Unknown model_name: {model_name}")


def create_model_and_optimizer(
    model_type: str = "simple",
    model_kwargs: Optional[Dict[str, Any]] = None,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    device: str = 'cpu'
) -> Tuple[nn.Module, optim.Optimizer]:
    """Create model and optimizer with practical defaults.
    
    Args:
        model_type: 'simple', 'enhanced', 'flexible', or 'resnet18'
        model_kwargs: Additional parameters for model constructor
        learning_rate: Learning rate for optimizer
        optimizer_type: 'adam' or 'sgd'
        device: Device to move model to
        
    Returns:
        Tuple of (model, optimizer)
    """
    model_kwargs = model_kwargs or {}
    model = get_model(model_type, **model_kwargs).to(device)
    
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=5e-4
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer


def get_model_info(model_type: str = "simple") -> Dict[str, Any]:
    """Get high-level information about the model type."""
    name = (model_type or '').lower()
    if name in ('assignment', 'assignmentcnn'):
        return {
            'model_type': 'assignment',
            'input_size': (3, 64, 64),
            'num_classes': 10,
            'architecture': ['Conv2d(3→16)', 'ReLU', 'MaxPool2d', 'Conv2d(16→32)', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear(8192→100)', 'ReLU', 'Linear(100→10)'],
            'description': 'Assignment CNN for 64×64 CIFAR-10 variant'
        }
    info = {
        'model_type': model_type,
        'input_size': (3, 32, 32),
        'num_classes': 10,
        'architecture': [],
    }
    factory_info = CNNFactory.get_model_info(model_type)
    if factory_info:
        info['architecture'] = factory_info.get('features', [])
        info['description'] = factory_info.get('description', '')
    return info


# ---- CIFAR-10 High-level helpers (migrated from cifar10_classifier.py) ----

def train_cifar10_classifier(
    epochs: int = 10,
    model_type: str = "simple",
    batch_size: int = 128,
    learning_rate: float = 0.001,
    data_dir: str = './data',
    num_workers: int = 2,
    augment: bool = True,
    normalize: bool = True,
    pin_memory: bool = True,
    optimizer_type: str = 'adam',
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch
    import time
    from .neural_networks import get_device

    device = get_device()
    resize_to = 64 if model_type in ("assignment", "assignmentcnn") else None
    trainloader, testloader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        normalize=normalize,
        pin_memory=pin_memory,
        resize_to=resize_to,
    )

    model, optimizer = create_model_and_optimizer(
        model_type=model_type,
        model_kwargs=model_kwargs or {},
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    history = run_training(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )
    total_time = time.time() - start_time

    # Save model
    os.makedirs('../models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'classes': CIFAR10_CLASSES,
        'model_type': model_type,
        'model_kwargs': model_kwargs or {},
    }, '../models/cifar10_classifier.pth')

    # Attach metadata
    history = dict(history)
    history['training_time'] = total_time
    return history


def load_and_predict(
    image_tensor: Any,
    model_path: str = '../models/cifar10_classifier.pth',
    model_type: str = "simple",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, np.ndarray]:
    import torch
    from .neural_networks import get_device

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get('model_type', model_type)
    model_kwargs = checkpoint.get('model_kwargs', model_kwargs or {})

    model, _ = create_model_and_optimizer(
        model_type=model_type,
        model_kwargs=model_kwargs,
        learning_rate=0.001,
        optimizer_type='adam',
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        if hasattr(image_tensor, 'dim') and image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_name = CIFAR10_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        prob_array = probabilities.cpu().numpy()[0]
    return class_name, confidence_score, prob_array


def get_cifar10_model_info(model_type: str = "simple") -> Dict[str, Any]:
    info = get_model_info(model_type)
    info.update({
        'dataset': 'CIFAR-10',
        'classes': CIFAR10_CLASSES,
    })
    return info