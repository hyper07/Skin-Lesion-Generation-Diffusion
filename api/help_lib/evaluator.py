import torch


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate a model on a dataloader, returning (avg_loss, accuracy_percent)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Calculate average loss and accuracy on the provided dataset."""
    return evaluate(model, data_loader, criterion, device)