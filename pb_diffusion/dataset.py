from torch.utils.data import Dataset
from PIL import Image

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, disease_classes, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.disease_classes = disease_classes
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_name = self.labels[idx]
        label_idx = self.disease_classes[label_name]
        
        return image, label_idx, label_name