import torch
import os
import json
from datetime import datetime

# Get the base directory for checkpoints (sps_genai/checkpoints)
def get_checkpoint_base_dir():
    """Get the base checkpoint directory path (sps_genai/checkpoints)."""
    # Get the directory where this file is located (help_lib/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to sps_genai/, then to checkpoints/
    sps_genai_dir = os.path.dirname(current_dir)
    checkpoint_base = os.path.join(sps_genai_dir, 'checkpoints')
    return checkpoint_base

def save_checkpoint(models, optimizers, epoch, losses=None, accuracies=None, checkpoint_dir='checkpoints', keep_only_latest=True):
    """
    Save model checkpoint. By default, only keeps the latest checkpoint to save space.
    Metadata is saved in a separate JSON file.

    Args:
        models: Dict of PyTorch models to save, e.g., {'generator': gen_model, 'discriminator': disc_model}
        optimizers: Dict of optimizers corresponding to the models
        epoch: Current epoch number
        losses: Dict of loss values for each model (optional)
        accuracies: Dict of accuracy values for each model (optional)
        checkpoint_dir: Directory to save checkpoints (relative to sps_genai/checkpoints or absolute path)
        keep_only_latest: If True, only keep the latest checkpoint and delete old ones (default: True)
    """
    # If relative path, make it relative to sps_genai/checkpoints
    if not os.path.isabs(checkpoint_dir):
        base_dir = get_checkpoint_base_dir()
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and optimizer states (this is the large file)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    checkpoint = {
        'model_state_dicts': {name: model.state_dict() for name, model in models.items()},
        'optimizer_state_dicts': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved: {checkpoint_path}")
    
    # Save metadata (epoch, losses, accuracies) in a lightweight JSON file
    metadata = {
        'epoch': epoch,
        'losses': losses or {},
        'accuracies': accuracies or {},
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': checkpoint_path
    }
    metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Checkpoint metadata saved: {metadata_path}")
    
    # If keep_only_latest is True, delete old epoch-based checkpoints
    if keep_only_latest:
        # Delete old epoch-based checkpoints (model_epoch_XXX.pth)
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('model_epoch_') and filename.endswith('.pth'):
                old_checkpoint_path = os.path.join(checkpoint_dir, filename)
                try:
                    os.remove(old_checkpoint_path)
                    print(f"Deleted old checkpoint: {filename}")
                except Exception as e:
                    print(f"Warning: Could not delete old checkpoint {filename}: {e}")

def load_checkpoint(models, optimizers, checkpoint_path, device='cpu'):
    """
    Load model checkpoint and restore training state.
    Automatically loads metadata from JSON file if available.

    Args:
        models: Dict of PyTorch models to load state into
        optimizers: Dict of optimizers to load state into
        checkpoint_path: Path to checkpoint file (or checkpoint directory)
        device: Device to load tensors to ('cpu' or 'cuda')
    
    Returns:
        tuple: (epoch, losses, accuracies) from the checkpoint
    
    Note: Saving optimizer state is crucial for resuming training properly.
    It preserves learning rate schedules, momentum buffers, and other
    optimizer-specific state that affects convergence.
    """
    # If checkpoint_path is a directory, look for latest_checkpoint.pth
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, 'latest_checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model states
    for name, model in models.items():
        model.load_state_dict(checkpoint['model_state_dicts'][name])
    
    # Restore optimizer states
    if optimizers:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][name])
    
    # Try to load metadata from JSON file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        epoch = metadata.get('epoch', 0)
        losses = metadata.get('losses', {})
        accuracies = metadata.get('accuracies', {})
        print(f"Checkpoint loaded - Epoch: {epoch}, Losses: {losses}, Accuracies: {accuracies}")
    else:
        # Fallback to old format (if metadata was in checkpoint file)
        epoch = checkpoint.get('epoch', 0)
        losses = checkpoint.get('losses', {})
        accuracies = checkpoint.get('accuracies', {})
        print(f"Checkpoint loaded (legacy format) - Epoch: {epoch}, Losses: {losses}, Accuracies: {accuracies}")
    
    return epoch, losses, accuracies


def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Get the path to the latest checkpoint file
    
    Args:
        checkpoint_dir: Directory containing checkpoints (relative to sps_genai/checkpoints or absolute path)
        
    Returns:
        str: Path to the latest checkpoint file, or None if no checkpoints found
    """
    # If relative path, make it relative to sps_genai/checkpoints
    if not os.path.isabs(checkpoint_dir):
        base_dir = get_checkpoint_base_dir()
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for latest_checkpoint.pth first (new format)
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Fallback: If no latest checkpoint, find the highest numbered epoch checkpoint (legacy format)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and return the latest
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def get_checkpoint_metadata(checkpoint_dir='checkpoints'):
    """
    Get checkpoint metadata from JSON file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        dict: Metadata dictionary with epoch, losses, accuracies, etc., or None if not found
    """
    # If relative path, make it relative to sps_genai/checkpoints
    if not os.path.isabs(checkpoint_dir):
        base_dir = get_checkpoint_base_dir()
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir)
    
    metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def load_latest_checkpoint(models, optimizers, checkpoint_dir='checkpoints', device='cpu'):
    """
    Load the most recent checkpoint from a directory
    
    Args:
        models: Dict of PyTorch models to load state into
        optimizers: Dict of optimizers to load state into
        checkpoint_dir: Directory containing checkpoints
        device: Device to load tensors to ('cpu' or 'cuda')
    
    Returns:
        tuple: (epoch, losses, accuracies) from the checkpoint, or (0, {}, {}) if no checkpoint found
    """
    # If relative path, make it relative to sps_genai/checkpoints
    if not os.path.isabs(checkpoint_dir):
        base_dir = get_checkpoint_base_dir()
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir)
    
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        print(f"No checkpoints found in {checkpoint_dir}")
        return 0, {}, {}
    
    return load_checkpoint(models, optimizers, latest_checkpoint, device)