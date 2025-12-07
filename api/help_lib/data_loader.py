import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import os


# CIFAR-10 class names (made globally available for reuse)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar10_transforms(
    augment: bool = True,
    normalize: bool = True,
    resize_to: Optional[int] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms for CIFAR-10 with optional augmentation/normalization and resizing.
    
    Args:
        augment: Whether to apply data augmentation to the training set.
        normalize: Whether to normalize using CIFAR-10 mean/std.
        resize_to: If provided, resize images to (resize_to, resize_to).
    """
    normalize_transform = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ) if normalize else None

    train_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] if augment else []
    if resize_to:
        train_transforms.append(transforms.Resize((resize_to, resize_to)))
    train_transforms += [transforms.ToTensor()]
    if normalize_transform:
        train_transforms.append(normalize_transform)

    test_transforms = []
    if resize_to:
        test_transforms.append(transforms.Resize((resize_to, resize_to)))
    test_transforms.append(transforms.ToTensor())
    if normalize_transform:
        test_transforms.append(normalize_transform)

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    normalize: bool = True,
    pin_memory: bool = True,
    resize_to: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test dataloaders with common sensible defaults."""
    transform_train, transform_test = get_cifar10_transforms(
        augment=augment,
        normalize=normalize,
        resize_to=resize_to,
    )

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return trainloader, testloader


def get_data_loader(data_dir, batch_size=32, train=True):
    """Create a general-purpose DataLoader from a directory using ImageFolder.

    This expects `data_dir` to contain class subdirectories. If `train` is True,
    and a `train/` subdirectory exists, it will load from `data_dir/train`; if
    False and `test/` exists, it will load from `data_dir/test`. Otherwise, it
    loads directly from `data_dir`.
    """
    # Choose split directory if present
    split_dir = 'train' if train else 'test'
    candidate_path = os.path.join(data_dir, split_dir)
    root = candidate_path if os.path.isdir(candidate_path) else data_dir

    # Basic transforms; add light augmentation for training if desired
    transform_list = []
    if train and os.path.isdir(candidate_path):
        # Only apply augmentation when explicit train split exists
        transform_list.extend([transforms.RandomHorizontalFlip()])
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True,
    )
    return loader