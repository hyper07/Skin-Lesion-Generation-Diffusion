"""
Data Loader for HAM10000 + BCN20000 Skin Lesion Datasets
Handles the existing directory structure without reorganization
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from dataset import SkinLesionDataset

# Load environment variables
load_dotenv()

def load_ham10000_metadata(metadata_path, img_dir_part1, img_dir_part2):
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
    
    # print(f"HAM metadata columns: {df.columns.tolist()}")
    # print(f"HAM metadata shape: {df.shape}")
    
    image_paths = []
    labels = []
    
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
        
        # Check both part1 and part2 directories
        path1 = os.path.join(img_dir_part1, f"{image_id}.jpg")
        path2 = os.path.join(img_dir_part2, f"{image_id}.jpg")
        
        if os.path.exists(path1):
            image_paths.append(path1)
            labels.append(diagnosis)
        elif os.path.exists(path2):
            image_paths.append(path2)
            labels.append(diagnosis)
        else:
            print(f"Warning: Image {image_id} not found")
    
    # print(f"\nHAM10000 diagnosis distribution:")
    # for dx, count in pd.Series(labels).value_counts().items():
    #     print(f"  {dx}: {count}")
    
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
    
    # print(f"\nBCN20000 metadata shape: {df.shape}")
    # print(f"BCN20000 columns: {df.columns.tolist()}")
    
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
    
    # print("BCN20000 diagnosis distribution:")
    # for dx, count in pd.Series(labels).value_counts().items():
    #     print(f"  {dx}: {count}")
    
    return image_paths, labels

def create_data_loaders(
    ham_metadata_path,
    ham_img_part1,
    ham_img_part2,
    bcn_metadata_path,
    bcn_img_dir,
    batch_size=32,
    img_size=256,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    top_n_classes=None,
    seed=42
):
    """
    Create train, validation, and test data loaders
    
    Args:
        top_n_classes: If specified, only keep the top N most frequent classes
    """
    
    # print("=" * 60)
    # print("Loading Datasets")
    # print("=" * 60)
    
    # Load both datasets
    ham_paths, ham_labels = load_ham10000_metadata(ham_metadata_path, ham_img_part1, ham_img_part2)
    bcn_paths, bcn_labels = load_bcn20000_metadata(bcn_metadata_path, bcn_img_dir)
    
    # Combine
    all_paths = ham_paths + bcn_paths
    all_labels = ham_labels + bcn_labels
    
    # print(f"\n{'='*60}")
    # print(f"COMBINED: {len(all_paths)} total images")
    # print(f"{'='*60}")
    
    # # Show full distribution
    # print("\nFull class distribution:")
    label_counts = pd.Series(all_labels).value_counts()
    # for dx, count in label_counts.items():
    #     pct = (count / len(all_labels)) * 100
    #     print(f"  {dx}: {count} ({pct:.2f}%)")
    
    # FILTER TO TOP N CLASSES
    if top_n_classes is not None:
        # print(f"\n{'='*60}")
        # print(f"FILTERING TO TOP {top_n_classes} CLASSES")
        # print(f"{'='*60}")
        
        # Get top N classes
        top_classes = label_counts.nlargest(top_n_classes).index.tolist()
        # print(f"\nTop {top_n_classes} classes selected:")
        # for i, cls in enumerate(top_classes, 1):
        #     print(f"  {i}. {cls} ({label_counts[cls]} images)")
        
        # Filter data
        filtered_paths = []
        filtered_labels = []
        
        for path, label in zip(all_paths, all_labels):
            if label in top_classes:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        all_paths = filtered_paths
        all_labels = filtered_labels
        
        # print(f"\nFiltered dataset: {len(all_paths)} images")
        # print("\nFiltered distribution:")
        # for dx, count in pd.Series(all_labels).value_counts().items():
        #     pct = (count / len(all_labels)) * 100
        #     print(f"  {dx}: {count} ({pct:.2f}%)")
    
    # Create class mapping
    unique_diseases = sorted(list(set(all_labels)))
    disease_classes = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    # print(f"\n{'='*60}")
    # print(f"Final: {len(unique_diseases)} classes")
    # print(f"{'='*60}")
    # for disease, idx in disease_classes.items():
    #     count = all_labels.count(disease)
    #     pct = (count / len(all_labels)) * 100
    #     print(f"  [{idx}] {disease}: {count} ({pct:.2f}%)")
    
    # # Split data
    # print(f"\n{'='*60}")
    # print("Creating splits...")
    # print(f"{'='*60}")
    
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
    
    # print(f"  Train: {len(train_paths)} ({train_split*100:.0f}%)")
    # print(f"  Val:   {len(val_paths)} ({val_split*100:.0f}%)")
    # print(f"  Test:  {len(test_paths)} ({test_split*100:.0f}%)")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_paths, train_labels, disease_classes, train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, disease_classes, val_test_transform)
    test_dataset = SkinLesionDataset(test_paths, test_labels, disease_classes, val_test_transform)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # print(f"\n{'='*60}")
    # print("✓ Data loaders ready!")
    # print(f"{'='*60}")
    
    return train_loader, val_loader, test_loader, disease_classes
