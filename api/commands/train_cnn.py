import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.cnn_models import train_cnn as train_cnn_func

def train_cnn(model_type='simple', dataset='cifar10', epochs=10, batch_size=64, learning_rate=0.001, checkpoint_dir='checkpoints', device=None):
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    device = torch.device(device)

    # Use the train_cnn function from cnn_models
    cnn = train_cnn_func(
        model_type=model_type, 
        dataset=dataset, 
        epochs=epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        save_path=f"{checkpoint_dir}/cnn_{dataset}/cnn_{dataset}.pth",
        device=device
    )

    # Additional logic can be added here if needed
    # For now, just return the trained CNN
    return cnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN on specified dataset')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'enhanced', 'custom'], help='CNN model type')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default=None, help='Device to use (auto-detect if not specified: mps > cuda > cpu)')
    args = parser.parse_args()

    train_cnn(args.model_type, args.dataset, args.epochs, args.batch_size, args.lr, args.checkpoint_dir, args.device)