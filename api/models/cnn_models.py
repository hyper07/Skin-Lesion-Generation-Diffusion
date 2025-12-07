"""
CNN Model Definitions
Practical and flexible CNN architectures for various use cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class BaseCNN(nn.Module):
    """Base CNN class with common functionality."""
    
    def __init__(self, num_classes: int = 10):
        super(BaseCNN, self).__init__()
        self.num_classes = num_classes
    
    def _calculate_conv_output_size(self, input_size: int, kernel_size: int, 
                                   stride: int = 1, padding: int = 0) -> int:
        """Calculate output size after convolution or pooling."""
        return (input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    
    def get_device(self):
        """Get device of the model."""
        return next(self.parameters()).device


class SimpleCNN(BaseCNN):
    """
    Simple CNN architecture suitable for CIFAR-10 (32x32x3 images) or MNIST (28x28x1).
    Based on Module 4 Practical.
    
    Architecture:
    - Conv2d(input_channels→16) → ReLU → MaxPool2d
    - Conv2d(16→32) → ReLU → MaxPool2d  
    - FC(flattened→128) → ReLU → FC(128→classes)
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3, input_size: int = 32, device: Optional[str] = None):
        super(SimpleCNN, self).__init__(num_classes)
        # Auto-detect best available device: MPS (Mac) > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size: input_size // 4, since two pools halve twice
        size_after_pools = input_size // 4
        flattened_size = 32 * size_after_pools * size_after_pools
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.conv1(x)))  # input_size → input_size/2
        x = self.pool(F.relu(self.conv2(x)))  # input_size/2 → input_size/4
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EnhancedCNN(BaseCNN):
    """
    Enhanced CNN with BatchNorm and Dropout for better performance.
    Based on Module 4 Practical.
    
    Architecture:
    - 4 Conv blocks with BatchNorm: 3→16→32→64→128
    - Dropout regularization
    - Progressive feature extraction
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3, dropout_rate: float = 0.5):
        super(EnhancedCNN, self).__init__(num_classes)
        
        # Convolutional blocks with BatchNorm
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # For 32x32 input: 32→16→8→4→2, so 128*2*2=512
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv blocks with BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8→4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4→2
        
        # Flatten and fully connected with dropout
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CustomCNN(BaseCNN):
    """
    Flexible CNN that adapts to different input sizes.
    Good for assignments with varying requirements.
    """
    
    def __init__(self, input_size: int = 32, input_channels: int = 3, 
                 num_classes: int = 10, hidden_dim: int = 100):
        super(CustomCNN, self).__init__(num_classes)
        
        self.input_size = input_size
        
        # Standard conv layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after two pool operations
        size_after_pools = input_size // 4  # Two pools, each halves the size
        flattened_size = 32 * size_after_pools * size_after_pools
        
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNFactory:
    """Factory class to create CNN models based on requirements."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseCNN:
        """
        Create CNN model based on type.
        
        Args:
            model_type: 'simple', 'enhanced', or 'custom'
            **kwargs: Model-specific parameters
        """
        if model_type.lower() == 'simple':
            return SimpleCNN(**kwargs)
        elif model_type.lower() == 'enhanced':
            return EnhancedCNN(**kwargs)
        elif model_type.lower() == 'custom' or model_type.lower() == 'flexible':
            return CustomCNN(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model_type: str) -> dict:
        """Get information about model architecture."""
        info = {
            'simple': {
                'description': 'Basic CNN for CIFAR-10',
                'layers': 4,
                'parameters': '~85K',
                'features': ['2 Conv layers', 'Max pooling', 'ReLU activation']
            },
            'enhanced': {
                'description': 'Advanced CNN with regularization',
                'layers': 8,
                'parameters': '~180K', 
                'features': ['4 Conv layers', 'BatchNorm', 'Dropout', 'Progressive channels']
            },
            'flexible': {
                'description': 'Adaptable CNN for various input sizes',
                'layers': 4,
                'parameters': 'Variable',
                'features': ['Flexible input size', 'Configurable dimensions']
            }
        }
        return info.get(model_type.lower(), {})


# Example usage and configuration
def get_cifar10_model(model_type: str = 'simple') -> BaseCNN:
    """Get a pre-configured model for CIFAR-10."""
    configs = {
        'simple': {'num_classes': 10, 'input_channels': 3},
        'enhanced': {'num_classes': 10, 'input_channels': 3, 'dropout_rate': 0.5},
        'flexible': {'input_size': 32, 'input_channels': 3, 'num_classes': 10, 'hidden_dim': 128}
    }
    
    config = configs.get(model_type, configs['simple'])
    return CNNFactory.create_model(model_type, **config)


def get_assignment_model(input_size: int = 64, hidden_dim: int = 100) -> BaseCNN:
    """Get model specifically for assignment requirements."""
    return CustomCNN(
        input_size=input_size,
        input_channels=3,
        num_classes=10,
        hidden_dim=hidden_dim
    )


def get_cifar10_dataloader(batch_size: int = 128, download: bool = True) -> DataLoader:
    """Get CIFAR-10 dataloader with proper transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader


def get_mnist_dataloader(batch_size: int = 128, download: bool = True) -> DataLoader:
    """Get MNIST dataloader with proper transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader


def train_cnn(model_type: str, dataset: str, epochs: int = 50, batch_size: int = 128, 
              learning_rate: float = 0.001, save_path: str = "checkpoints/cnn_{dataset}/cnn_{dataset}.pth", 
              device: Optional[str] = None, **kwargs) -> BaseCNN:
    """
    Train CNN on the specified dataset.
    
    Args:
        model_type: 'simple', 'enhanced', or 'custom'
        dataset: 'cifar10' or 'mnist'
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
        **kwargs: Additional arguments for model creation
    """
    dataset = dataset.lower()
    
    # Get dataloader
    if dataset == 'cifar10' or dataset == 'cifar':
        dataloader = get_cifar10_dataloader(batch_size)
        num_classes = 10
        input_channels = 3
        input_size = 32
    elif dataset == 'mnist':
        dataloader = get_mnist_dataloader(batch_size)
        num_classes = 10
        input_channels = 1
        input_size = 28
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Create model
    model_kwargs = {
        'num_classes': num_classes,
        'input_channels': input_channels,
        'input_size': input_size,
        **kwargs
    }
    model = CNNFactory.create_model(model_type, **model_kwargs)
    
    # Auto-detect best available device: MPS (Mac) > CUDA > CPU
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    device = torch.device(device)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training {model_type.upper()} CNN on {dataset.upper()} for {epochs} epochs...")
    print(f"Using device: {device}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                ncols=0,
                dynamic_ncols=True,
                leave=False,
                position=0
            )
        except ImportError:
            iterator = dataloader
        
        for batch_idx, (images, labels) in enumerate(iterator):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        # Update progress bar
        if hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({
                "Loss": f"{epoch_loss:.4f}",
                "Acc": f"{epoch_acc:.2f}%",
            })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
        # Save checkpoint every epoch
        checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'model_type': model_type,
            'dataset': dataset,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'input_channels': input_channels,
            'input_size': input_size,
            'num_classes': num_classes
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    save_path = save_path.format(dataset=dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'input_channels': input_channels,
        'input_size': input_size,
        'num_classes': num_classes
    }, save_path)
    print(f"Training completed. Final model saved to {save_path}")
    
    return model