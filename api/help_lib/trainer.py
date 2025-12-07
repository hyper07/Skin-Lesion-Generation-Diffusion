import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from .evaluator import evaluate as evaluate_model_metrics


def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader: DataLoader, criterion, device) -> Tuple[float, float]:
    # Backwards-compatible wrapper; now implemented in evaluator.py
    return evaluate_model_metrics(model, dataloader, criterion, device)


def train_model(
    model,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion,
    optimizer,
    device='cpu',
    epochs: int = 10,
) -> Dict[str, Any]:
    """Train and evaluate a model for the given number of epochs."""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': epochs,
    }

    for _ in range(epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return history