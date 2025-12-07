"""
Unified Data Loader for HAM10000 + BCN20000 Skin Lesion Datasets
Handles the existing directory structure without reorganization
"""

import os
import random
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from .dataset import SkinLesionDataset

# Load environment variables
load_dotenv()


def load_ham10000_metadata(metadata_path, img_dir):
    """Load HAM10000 metadata and create image path mappings"""
    
    # HAM10000 diagnosis code mapping
    ham_dx_map = {
        'nv': 'Nevus',
        'mel': 'Melanoma (HAM)',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratosis',
        'vasc': 'Vascular lesion',
        'df': 'Dermatofibroma',
        'bkl': 'Benign keratosis (HAM)'
    }
    
    # Try reading the metadata - could be CSV with or without header
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading HAM metadata: {e}")
        return [], []
    
    image_paths = []
    labels = []
    skipped_not_found = 0
    
    for _, row in df.iterrows():
        # Get image_id
        if 'image_id' in df.columns:
            image_id = row['image_id']
        elif 'lesion_id' in df.columns:
            image_id = row['lesion_id']
        else:
            image_id = row.iloc[0]
        
        # Get diagnosis code
        if 'dx' in df.columns:
            dx_code = row['dx']
        elif 'diagnosis' in df.columns:
            dx_code = row['diagnosis']
        else:
            dx_code = row.iloc[1]
        
        # Map to full diagnosis name
        diagnosis = ham_dx_map.get(dx_code, dx_code)
        
        # Check image in the directory
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(diagnosis)
        else:
            skipped_not_found += 1
    
    print(f"\nHAM10000 loaded: {len(image_paths)} images")
    if skipped_not_found > 0:
        print(f"  Skipped {skipped_not_found} images not found on disk ({skipped_not_found/len(df)*100:.1f}%)")
    
    return image_paths, labels


def load_bcn20000_metadata(metadata_path, img_dir):
    """Load BCN20000 metadata - uses 'isic_id' and 'diagnosis_3' columns"""
    
    # BCN diagnosis mapping (standardize names)
    bcn_diagnosis_map = {
        'Nevus': 'Nevus',
        'Melanoma, NOS': 'Melanoma (BCN)',
        'Melanoma metastasis': 'Melanoma metastasis',
        'Basal cell carcinoma': 'Basal cell carcinoma',
        'Seborrheic keratosis': 'Seborrheic keratosis',
        'Solar or actinic keratosis': 'Actinic keratosis',
        'Squamous cell carcinoma, NOS': 'Squamous cell carcinoma',
        'Scar': 'Scar',
        'Solar lentigo': 'Solar lentigo',
        'Dermatofibroma': 'Dermatofibroma'
    }
    
    df = pd.read_csv(metadata_path)
    
    # Verify expected columns exist
    if 'isic_id' not in df.columns:
        raise ValueError(f"Expected 'isic_id' column in BCN metadata. Found: {df.columns.tolist()}")
    if 'diagnosis_3' not in df.columns:
        raise ValueError(f"Expected 'diagnosis_3' column in BCN metadata. Found: {df.columns.tolist()}")
    
    image_paths = []
    labels = []
    skipped_missing = 0
    skipped_not_found = 0
    
    for _, row in df.iterrows():
        image_id = row['isic_id']
        raw_diagnosis = row['diagnosis_3']
        
        # Skip rows with missing diagnosis
        if pd.isna(raw_diagnosis):
            skipped_missing += 1
            continue
        
        # Standardize diagnosis name
        diagnosis = bcn_diagnosis_map.get(raw_diagnosis, raw_diagnosis)
        
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(diagnosis)
        else:
            skipped_not_found += 1
    
    print(f"\nBCN20000 loaded: {len(image_paths)} images")
    if skipped_missing > 0:
        print(f"  Skipped {skipped_missing} images with missing diagnosis ({skipped_missing/len(df)*100:.1f}%)")
    if skipped_not_found > 0:
        print(f"  Skipped {skipped_not_found} images not found on disk")
    
    return image_paths, labels


def create_data_loaders(
    ham_metadata_path,
    ham_img_dir,
    bcn_metadata_path,
    bcn_img_dir,
    batch_size=32,
    img_size=256,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    top_n_classes=None,
    max_samples_per_class=2000,
    seed=42,
    num_workers=None,
    pin_memory=None,
    persistent_workers=None,
    prefetch_factor=None,
    balance_classes=True,
):
    """
    Create train, validation, and test data loaders
    
    Args:
        ham_metadata_path: Path to HAM10000 metadata CSV
        ham_img_dir: Path to HAM10000 images directory
        bcn_metadata_path: Path to BCN20000 metadata CSV
        bcn_img_dir: Path to BCN20000 images directory
        batch_size: Batch size for data loaders
        img_size: Target image size (will be resized to img_size x img_size)
        train_split: Fraction of data for training (default: 0.7)
        val_split: Fraction of data for validation (default: 0.15)
        test_split: Fraction of data for testing (default: 0.15)
        top_n_classes: If specified, only keep the top N most frequent classes
        max_samples_per_class: Maximum number of samples to keep per class (default: 2000)
        seed: Random seed for reproducibility
        num_workers: DataLoader workers (default: auto based on CPU cores)
        pin_memory: Enable CUDA pinned memory (default: True when CUDA available)
        persistent_workers: Keep workers alive between epochs (default: True when workers > 0)
        prefetch_factor: Number of batches prefetched per worker (default: 2 when workers > 0)
        balance_classes: Whether to use WeightedRandomSampler to balance classes (default: True)
    
    Returns:
        train_loader, val_loader, test_loader, disease_classes
    """
    
    # Load both datasets
    print("Loading datasets...")
    ham_paths, ham_labels = load_ham10000_metadata(ham_metadata_path, ham_img_dir)
    bcn_paths, bcn_labels = load_bcn20000_metadata(bcn_metadata_path, bcn_img_dir)
    
    # Combine
    all_paths = ham_paths + bcn_paths
    all_labels = ham_labels + bcn_labels
    
    print(f"\nTotal combined: {len(all_paths)} images from HAM10000 + BCN20000")
    
    label_counts = pd.Series(all_labels).value_counts()
    print("\nClass distribution (before balancing):")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # FILTER TO TOP N CLASSES
    if top_n_classes is not None:
        # Get top N classes
        top_classes = label_counts.nlargest(top_n_classes).index.tolist()
        
        # Filter data
        filtered_paths = []
        filtered_labels = []
        
        for path, label in zip(all_paths, all_labels):
            if label in top_classes:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        all_paths = filtered_paths
        all_labels = filtered_labels
        label_counts = pd.Series(all_labels).value_counts()
    
    # BALANCE CLASSES BY LIMITING MAX SAMPLES PER CLASS
    print(f"\nBalancing classes by limiting to max {max_samples_per_class} samples per class...")
    
    balanced_paths = []
    balanced_labels = []
    class_counts = {}
    
    for path, label in zip(all_paths, all_labels):
        if label not in class_counts:
            class_counts[label] = 0
        
        if class_counts[label] < max_samples_per_class:
            balanced_paths.append(path)
            balanced_labels.append(label)
            class_counts[label] += 1
    
    all_paths = balanced_paths
    all_labels = balanced_labels
    
    print(f"  After balancing: {len(all_paths)} images")
    balanced_counts = pd.Series(all_labels).value_counts()
    print("  Balanced class distribution:")
    for label, count in balanced_counts.items():
        print(f"    {label}: {count}")
    
    # Create class mapping
    unique_diseases = sorted(list(set(all_labels)))
    disease_classes = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=(val_split + test_split),
        stratify=all_labels,
        random_state=seed
    )
    
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=seed
    )
    
    # Add rotated versions for training data (90°, 180°, 270°)
    print(f"\nSkipping rotation augmentation - using original images only...")
    augmented_train_paths = train_paths.copy()
    augmented_train_labels = train_labels.copy()
    train_rotation_angles = [0] * len(train_paths)
    
    print(f"  Training images: {len(train_paths)} (no augmentation)")
    
    # Validation and test use original images only (no rotation)
    val_rotation_angles = [0] * len(val_paths)
    test_rotation_angles = [0] * len(test_paths)
    
    # Transforms
    # Simple resize to square: directly resize to img_size x img_size
    # Note: No augmentation - using original images only
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to square img_size x img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to square img_size x img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        augmented_train_paths, 
        augmented_train_labels, 
        disease_classes, 
        train_transform,
        rotation_angles=train_rotation_angles
    )
    val_dataset = SkinLesionDataset(
        val_paths, 
        val_labels, 
        disease_classes, 
        val_test_transform,
        rotation_angles=val_rotation_angles
    )
    test_dataset = SkinLesionDataset(
        test_paths, 
        test_labels, 
        disease_classes, 
        val_test_transform,
        rotation_angles=test_rotation_angles
    )
    
    # DataLoader performance configuration
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(8, cpu_count)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 2

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor

    print("\nDataLoader configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    if num_workers > 0:
        print(f"  persistent_workers: {persistent_workers}")
        print(f"  prefetch_factor: {prefetch_factor}")

    # Configure sampler for class balancing
    sampler = None
    shuffle = True
    
    if balance_classes:
        print("\nCalculating class weights for balancing...")
        # Count classes in training set
        train_counts = pd.Series(augmented_train_labels).value_counts()
        print("  Training set class distribution:")
        for label, count in train_counts.items():
            print(f"    {label}: {count}")
            
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / train_counts
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in augmented_train_labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False # Shuffle is mutually exclusive with sampler
        print("  Class balancing enabled (WeightedRandomSampler)")

    # Create loaders
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, disease_classes

