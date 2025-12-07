"""
Dataset class for skin lesion images
"""

from torch.utils.data import Dataset
from PIL import Image


class SkinLesionDataset(Dataset):
    """Dataset for skin lesion images with class labels"""
    
    def __init__(self, image_paths, labels, disease_classes, transform=None, rotation_angles=None):
        """
        Args:
            image_paths: List of image file paths (or tuples of (path, rotation_angle))
            labels: List of label names (strings)
            disease_classes: Dictionary mapping disease names to class indices
            transform: Optional transform to apply to images
            rotation_angles: Optional list of rotation angles (0, 90, 180, 270) for each image
        """
        self.image_paths = image_paths
        self.labels = labels
        self.disease_classes = disease_classes
        self.transform = transform
        self.rotation_angles = rotation_angles if rotation_angles is not None else [0] * len(image_paths)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rotation_angle = self.rotation_angles[idx] if idx < len(self.rotation_angles) else 0
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply rotation if specified
            if rotation_angle != 0:
                image = image.rotate(rotation_angle, expand=False)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            from torchvision import transforms
            image = Image.new('RGB', (256, 256), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label_name = self.labels[idx]
        label_idx = self.disease_classes[label_name]
        
        return image, label_idx, label_name

