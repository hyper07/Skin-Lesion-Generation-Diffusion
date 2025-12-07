"""
CNN Classifier for Skin Disease Classification with Class Imbalance Handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pb_diffusion.data_loader import create_data_loaders
import os
from dotenv import load_dotenv

class SkinDiseaseClassifier(nn.Module):
    """ResNet-50 based classifier for skin diseases"""
    
    def __init__(self, num_classes=13, pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def compute_class_weights(train_loader, num_classes, device):
    """Compute inverse frequency weights for class imbalance"""
    class_counts = Counter()
    
    print("Computing class weights...")
    for _, labels, _ in tqdm(train_loader, desc="Analyzing class distribution"):
        class_counts.update(labels.cpu().numpy())
    
    total_samples = sum(class_counts.values())
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    
    print("\nClass distribution:")
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)
        # Inverse frequency weighting
        weight = total_samples / (num_classes * count)
        class_weights[cls_idx] = weight
        print(f"  Class {cls_idx}: {count} samples, weight: {weight:.4f}")
    
    # Normalize weights (optional, helps with learning rate stability)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set with precision, recall, F1"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    
    # Compute precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'per_class': {
            'precision': precision_per_class * 100,
            'recall': recall_per_class * 100,
            'f1': f1_per_class * 100,
            'support': support_per_class
        }
    }
    
    return metrics, all_preds, all_labels


def train_classifier(
    train_loader,
    val_loader,
    test_loader,
    disease_classes,
    num_epochs=50,
    learning_rate=1e-4,
    device='mps',
    checkpoint_dir='checkpoints/cnn_evaluator',
    early_stopping_patience=5,
    use_class_weights=True
):
    """Train the classifier with comprehensive metrics tracking and class imbalance handling"""
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/cnn_evaluator_{timestamp}")
    train_val_dir = log_dir / "train_val"
    test_dir = log_dir / "test"
    train_val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(disease_classes)
    model = SkinDiseaseClassifier(num_classes=num_classes).to(device)
    
    # Compute class weights if enabled
    if use_class_weights:
        class_weights = compute_class_weights(train_loader, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n✓ Using weighted loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\n✓ Using standard cross-entropy loss")
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    print(f"\nTraining CNN Evaluator with {num_classes} classes")
    print(f"Device: {device}")
    print(f"Timestamp: {timestamp}")
    print(f"Logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate with full metrics
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Schedule learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.2f}% | Val Recall: {val_metrics['recall']:.2f}% | Val F1: {val_metrics['f1']:.2f}%")
        
        # Save per-epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'train': {'loss': train_loss, 'accuracy': train_acc},
            'val': {
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1']
            }
        }
        
        with open(train_val_dir / f"epoch_{epoch+1:03d}.json", 'w') as f:
            json.dump(epoch_results, f, indent=2)
        
        # Save best model (based on validation loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'disease_classes': disease_classes,
                'num_classes': num_classes,
                'timestamp': timestamp,
                'class_weights': class_weights.cpu() if use_class_weights else None
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")
            print(f"✓ Saved best model (Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save full training history
    with open(train_val_dir / "full_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("Loading best model for final test evaluation...")
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {test_metrics['precision']:.2f}%")
    print(f"Test Recall:    {test_metrics['recall']:.2f}%")
    print(f"Test F1:        {test_metrics['f1']:.2f}%")
    
    # Per-class test metrics
    print(f"\n{'='*60}")
    print("PER-CLASS TEST METRICS")
    print(f"{'='*60}")
    print(f"{'Class':<35} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<8}")
    print("-" * 70)
    
    per_class_results = []
    for cls_idx in range(num_classes):
        class_name = class_names[cls_idx]
        prec = test_metrics['per_class']['precision'][cls_idx]
        rec = test_metrics['per_class']['recall'][cls_idx]
        f1 = test_metrics['per_class']['f1'][cls_idx]
        support = int(test_metrics['per_class']['support'][cls_idx])
        
        print(f"{class_name:<35} {prec:>6.2f}% {rec:>6.2f}% {f1:>6.2f}% {support:>8d}")
        
        per_class_results.append({
            'class_id': cls_idx,
            'class_name': class_name,
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'support': support
        })
    
    # Save test results
    test_results = {
        'timestamp': timestamp,
        'overall': {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        },
        'per_class': per_class_results,
        'disease_classes': disease_classes,
        'used_class_weights': use_class_weights
    }
    
    with open(test_dir / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Also save classification report
    report = classification_report(test_labels, test_preds, 
                                   target_names=[class_names[i] for i in range(num_classes)], 
                                   digits=4)
    with open(test_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    print(f"\n✓ Test results saved to {test_dir}")
    print(f"✓ Training history saved to {train_val_dir}")
    print(f"✓ Best model saved to {checkpoint_dir}/best_model.pt")
    
    return model, test_results, timestamp


def evaluate_generated_images(model, generated_loader, device, disease_classes):
    """Evaluate classifier on generated images"""
    print("\nEvaluating on Generated Images...")
    
    # Use unweighted loss for evaluation
    criterion = nn.CrossEntropyLoss()
    metrics, preds, labels = evaluate(model, generated_loader, criterion, device)
    
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    print(f"\nGenerated Images Results:")
    print(f"Overall Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Overall Precision: {metrics['precision']:.2f}%")
    print(f"Overall Recall:    {metrics['recall']:.2f}%")
    print(f"Overall F1:        {metrics['f1']:.2f}%")
    print(f"Loss:              {metrics['loss']:.4f}")
    
    print(f"\nPer-Class Metrics on Generated:")
    print(f"{'Class':<35} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 60)
    for cls_idx in range(len(disease_classes)):
        class_name = class_names[cls_idx]
        prec = metrics['per_class']['precision'][cls_idx]
        rec = metrics['per_class']['recall'][cls_idx]
        f1 = metrics['per_class']['f1'][cls_idx]
        print(f"{class_name:<35} {prec:>6.2f}% {rec:>6.2f}% {f1:>6.2f}%")
    
    return {
        'overall': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'loss': metrics['loss']
        },
        'per_class': {
            class_names[i]: {
                'precision': float(metrics['per_class']['precision'][i]),
                'recall': float(metrics['per_class']['recall'][i]),
                'f1': float(metrics['per_class']['f1'][i])
            }
            for i in range(len(disease_classes))
        }
    }


if __name__ == "__main__":
    
    
    load_dotenv()
    DATA_DIR = os.getenv('DATA_DIR')
    
    # Load data
    train_loader, val_loader, test_loader, disease_classes = create_data_loaders(
        ham_metadata_path=os.path.join(DATA_DIR, "HAM10000_metadata"),
        ham_img_part1=os.path.join(DATA_DIR, "HAM10000_images"),
        ham_img_part2=os.path.join(DATA_DIR, "HAM10000_images"),
        bcn_metadata_path=os.path.join(DATA_DIR, "ISIC_metadata.csv"),
        bcn_img_dir=os.path.join(DATA_DIR, "ISIC_images"),
        batch_size=32,
        img_size=224,  # ResNet50 standard input
        top_n_classes=None  # Use all 13 classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Train with class imbalance handling
    model, results, timestamp = train_classifier(
        train_loader, val_loader, test_loader,
        disease_classes,
        num_epochs=1,
        learning_rate=1e-4,
        device=device,
        checkpoint_dir='checkpoints/cnn_evaluator',
        early_stopping_patience=5,
        use_class_weights=True  # Enable class imbalance handling
    )