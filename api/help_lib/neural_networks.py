"""
Neural network utility functions.
Implementation of activation functions, loss functions, and neural network components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def calculate_cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate cross-entropy loss for classification."""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    if targets.ndim == 1:  # Integer labels
        return -np.mean(np.log(predictions[np.arange(len(targets)), targets]))
    else:  # One-hot encoded
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def calculate_mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate mean squared error loss."""
    return np.mean((predictions - targets) ** 2)


def apply_convolution_2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Apply 2D convolution operation."""
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    output_h = (h - kh) // stride + 1
    output_w = (w - kw) // stride + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(0, output_h):
        for j in range(0, output_w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kh
            end_j = start_j + kw
            
            output[i, j] = np.sum(image[start_i:end_i, start_j:end_j] * kernel)
    
    return output


def apply_max_pooling_2d(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling operation."""
    h, w = image.shape
    
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + pool_size
            end_j = start_j + pool_size
            
            output[i, j] = np.max(image[start_i:end_i, start_j:end_j])
    
    return output


class SimpleFC(nn.Module):
    """Simple Fully Connected Neural Network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network from Module 4 Practical."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )  # Input channels = 3, Output channels = 16
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )  # Input channels = 16, Output channels = 32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Convolutional Layer 1 with BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2 with BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )  # Output channels = 64
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )  # Output channels = 128
        self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conf and pooling layers
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn4(self.conv4(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x

class AssignmentCNN(nn.Module):
    """
    CNN architecture for assignment requirements:
    - Input: 64×64×3 RGB image
    - Conv2D (16 filters, 3×3 kernel, stride=1, padding=1) -> ReLU -> MaxPool2D (2×2, stride=2)
    - Conv2D (32 filters, 3×3 kernel, stride=1, padding=1) -> ReLU -> MaxPool2D (2×2, stride=2)
    - Flatten -> FC (100 units) -> ReLU -> FC (10 units)
    """
    def __init__(self):
        super(AssignmentCNN, self).__init__()
        
        # First convolutional layer: 3 -> 16 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Second convolutional layer: 16 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer (shared for both pooling operations)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 64×64, after conv1+pool: 32×32, after conv2+pool: 16×16
        # Flattened size: 32 * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        # First conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 16, 32, 32)
        
        # Second conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 32, 16, 16)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)  # Output: (batch_size, 32*16*16)
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))  # Output: (batch_size, 100)
        
        # Second fully connected layer (output layer)
        x = self.fc2(x)  # Output: (batch_size, 10)
        
        return x


def get_device():
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def calculate_output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """Calculate output size after convolution or pooling."""
    return (input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
