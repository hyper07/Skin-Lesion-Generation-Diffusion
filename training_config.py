"""
Training configuration template with MPS support for Mac M1/M2/M3.
This module provides easy-to-use configurations for image training on Apple Silicon.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

class TrainingConfig:
    """
    Training configuration that automatically detects and configures MPS support.
    """
    
    def __init__(self, 
                 batch_size=32, 
                 learning_rate=1e-3, 
                 epochs=10,
                 image_size=224,
                 num_classes=10):
        
        # Device configuration
        self.device = self._get_optimal_device()
        self.use_mps = torch.backends.mps.is_available()
        
        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Device-optimized settings
        if self.use_mps:
            # MPS works better with smaller batch sizes and num_workers=0
            self.dataloader_num_workers = 0
            self.pin_memory = False  # Not needed for MPS
            self.mixed_precision = True  # MPS supports mixed precision
        elif torch.cuda.is_available():
            # NVIDIA GPU (Intel PC) optimization
            import os
            cpu_count = os.cpu_count() or 4
            self.dataloader_num_workers = min(8, cpu_count)  # Scale with CPU cores
            self.pin_memory = True  # Faster GPU memory transfer
            self.mixed_precision = True  # Use AMP for better performance
        else:
            # CPU-only (Intel PC without GPU)
            import os
            cpu_count = os.cpu_count() or 4
            self.dataloader_num_workers = min(4, cpu_count // 2)  # Don't overwhelm CPU
            self.pin_memory = False  # No GPU to pin to
            self.mixed_precision = False  # CPU doesn't benefit from mixed precision
        
        # Optimization settings
        self.weight_decay = 1e-4
        self.scheduler_patience = 5
        self.early_stopping_patience = 10
        
        print(f"✅ Training configured for device: {self.device}")
        if self.use_mps:
            print("🚀 Metal Performance Shaders (MPS) acceleration enabled!")
    
    def _get_optimal_device(self):
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def create_dataloader(self, dataset, shuffle=True, drop_last=True):
        """
        Create a DataLoader with optimal settings for the current device.
        
        Args:
            dataset: PyTorch Dataset
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DataLoader: Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last
        )
    
    def create_optimizer(self, model):
        """
        Create an optimizer for the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def create_scheduler(self, optimizer):
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            torch.optim.lr_scheduler: Configured scheduler
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.scheduler_patience
        )
    
    def move_to_device(self, *items):
        """
        Move tensors or models to the configured device.
        
        Args:
            *items: Tensors or models to move
            
        Returns:
            Moved items (single item if only one passed, tuple otherwise)
        """
        moved_items = [item.to(self.device) for item in items]
        return moved_items[0] if len(moved_items) == 1 else tuple(moved_items)
    
    def get_scaler(self):
        """
        Get a GradScaler for mixed precision training if supported.
        
        Returns:
            torch.cuda.amp.GradScaler or None: Scaler for mixed precision
        """
        if self.mixed_precision and not self.use_mps:
            # Note: MPS doesn't need GradScaler, it handles mixed precision automatically
            return torch.cuda.amp.GradScaler()
        return None


def create_image_transforms(image_size=224, is_training=True):
    """
    Create image transforms for training or validation.
    
    Args:
        image_size (int): Target image size
        is_training (bool): Whether these are training transforms
        
    Returns:
        torchvision.transforms.Compose: Image transforms
    """
    from torchvision import transforms
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def example_training_loop(model, train_loader, val_loader, config):
    """
    Example training loop that works with MPS and other devices.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: TrainingConfig instance
    """
    # Move model to device
    model = config.move_to_device(model)
    
    # Create optimizer and scheduler
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()
    scaler = config.get_scaler()
    
    print(f"🏋️  Starting training for {config.epochs} epochs")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = config.move_to_device(data, target)
            
            optimizer.zero_grad()
            
            if config.mixed_precision and scaler:
                # Mixed precision training (CUDA)
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training (MPS automatically handles mixed precision)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{config.epochs}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = config.move_to_device(data, target)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{config.epochs}: '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
        
        # Step scheduler
        scheduler.step(val_loss)


# Example usage
if __name__ == "__main__":
    # Create training configuration
    config = TrainingConfig(
        batch_size=64,
        learning_rate=1e-3,
        epochs=5,
        image_size=224,
        num_classes=10
    )
    
    print("\n📋 Training Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  DataLoader workers: {config.dataloader_num_workers}")