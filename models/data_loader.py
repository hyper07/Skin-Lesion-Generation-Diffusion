"""
Data Loader for HAM10000 + ISIC Skin Lesion Datasets
"""
import os
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from dataset import SkinLesionDataset

# Load environment variables
load_dotenv()


def load_data_loaders_from_notebook(image_size=256):
    """
    Load data loaders using the same logic as the notebook
    This replicates the data loading from 'Train Test Val (Top 3).ipynb'
    """
    DATA_DIR = os.getenv('DATA_DIR')
    if DATA_DIR is None:
        raise ValueError("DATA_DIR not found in .env file. Please create .env with DATA_DIR=<path>")

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

    # ISIC diagnosis mapping
    isic_diagnosis_map = {
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

    # Load HAM10000
    ham_metadata_path = os.path.join(DATA_DIR, "dataverse_files/HAM10000_metadata.csv")
    ham_images_dir = os.path.join(DATA_DIR, "dataverse_files/HAM10000_images")

    ham_df = pd.read_csv(ham_metadata_path)
    ham_paths = []
    ham_labels = []

    for _, row in ham_df.iterrows():
        image_id = row.get('image_id', row.get('lesion_id', row.iloc[0]))
        dx_code = row.get('dx', row.get('diagnosis', row.iloc[1]))
        diagnosis = ham_dx_map.get(dx_code, dx_code)

        img_path = os.path.join(ham_images_dir, f"{image_id}.jpg")

        if os.path.exists(img_path):
            ham_paths.append(img_path)
            ham_labels.append(diagnosis)

    # Load ISIC
    isic_metadata_path = os.path.join(DATA_DIR, "dataverse_files/ISIC_metadata.csv")
    isic_images = os.path.join(DATA_DIR, "dataverse_files/ISIC_images")

    isic_df = pd.read_csv(isic_metadata_path)
    isic_paths = []
    isic_labels = []

    for _, row in isic_df.iterrows():
        image_id = row['isic_id']
        raw_diagnosis = row['diagnosis_3']

        if pd.isna(raw_diagnosis):
            continue

        diagnosis = isic_diagnosis_map.get(raw_diagnosis, raw_diagnosis)
        img_path = os.path.join(isic_images, f"{image_id}.jpg")

        if os.path.exists(img_path):
            isic_paths.append(img_path)
            isic_labels.append(diagnosis)

    # Combine
    all_paths = ham_paths + isic_paths
    all_labels = ham_labels + isic_labels

    # Stratified sampling: 400 samples max for Nevus, all samples for other classes
    nevus_limit = 400
    sampled_paths = []
    sampled_labels = []

    # Group by class
    from collections import defaultdict
    class_data = defaultdict(list)
    for path, label in zip(all_paths, all_labels):
        class_data[label].append(path)

    # Sample from each class
    for class_name, paths in class_data.items():
        if class_name == 'Nevus':
            # Limit Nevus to 400 samples
            import random
            random.seed(42)
            sampled_class_paths = random.sample(paths, min(nevus_limit, len(paths)))
        else:
            # Use all samples for other classes
            sampled_class_paths = paths

        sampled_paths.extend(sampled_class_paths)
        sampled_labels.extend([class_name] * len(sampled_class_paths))

    total_samples = len(sampled_paths)
    print(f"Stratified sampling completed:")
    print(f"  Nevus: limited to {nevus_limit} samples")
    print(f"  Other classes: all available samples")
    print(f"  Total samples: {total_samples}")

    # All possible classes from HAM10000 and ISIC mappings
    all_possible_classes = [
        'Actinic keratosis',
        'Basal cell carcinoma',
        'Benign keratosis (HAM)',
        'Dermatofibroma',
        'Melanoma (BCN)',
        'Melanoma (HAM)',
        'Melanoma metastasis',
        'Nevus',
        'Scar',
        'Seborrheic keratosis',
        'Solar lentigo',
        'Squamous cell carcinoma',
        'Vascular lesion'
    ]

    # Create class mapping from all possible diseases
    disease_classes = {disease: idx for idx, disease in enumerate(all_possible_classes)}

    print(f"Classes: {all_possible_classes}")
    print(f"Class distribution in sample:")
    label_counts_sample = pd.Series(sampled_labels).value_counts()
    for class_name, count in label_counts_sample.items():
        print(f"  {class_name}: {count}")

    # Use sampled data
    filtered_paths = sampled_paths
    filtered_labels = sampled_labels

    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filtered_paths, filtered_labels,
        test_size=0.3,
        stratify=filtered_labels,
        random_state=42
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    # Transforms (256x256 images, normalized to [-1, 1])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Create datasets
    train_dataset = SkinLesionDataset(train_paths, train_labels, disease_classes, train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, disease_classes, val_test_transform)
    test_dataset = SkinLesionDataset(test_paths, test_labels, disease_classes, val_test_transform)

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Classes: {disease_classes}")
    
    return train_dataset, val_dataset, test_dataset, disease_classes
