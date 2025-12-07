"""
Comprehensive Evaluation Script for Generated Skin Disease Images
Auto-evaluates all models with latest timestamps using test set from data loader
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import csv
from tqdm import tqdm
import numpy as np
from datetime import datetime
from collections import defaultdict

# For image quality metrics
from torchvision.models import inception_v3
from scipy import linalg
import lpips

from cnn_classifier import SkinDiseaseClassifier, evaluate

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pb_diffusion.data_loader import create_data_loaders
from dotenv import load_dotenv


class ImageDataset(Dataset):
    """Dataset for loading generated images from directory structure"""
    
    def __init__(self, root_dir, disease_classes, transform=None):
        self.transform = transform
        self.disease_classes = disease_classes
        self.images = []
        self.labels = []
        
        root_path = Path(root_dir)
        
        for folder in root_path.iterdir():
            if not folder.is_dir():
                continue
            
            class_idx = self._match_class(folder.name, disease_classes)
            if class_idx is None:
                print(f"Warning: Could not match folder '{folder.name}' to any class")
                continue
            
            for img_path in folder.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
        
        # Limit to 200 images per class
        class_images = defaultdict(list)
        for img, label in zip(self.images, self.labels):
            class_images[label].append(img)
        
        self.images = []
        self.labels = []
        for label, imgs in class_images.items():
            selected = imgs[:200]
            self.images.extend(selected)
            self.labels.extend([label] * len(selected))
        
        print(f"Loaded {len(self.images)} images from {root_dir} (limited to 200 per class)")
    
    def _match_class(self, folder_name, disease_classes):
        """Match folder name to class index"""
        folder_lower = folder_name.lower().strip()
        
        # Extract class from "class_0_Actinic keratosis" format
        if folder_lower.startswith("class_"):
            try:
                parts = folder_name.split("_", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
            except:
                pass
        
        # Try exact match
        for disease_name, idx in disease_classes.items():
            if disease_name.lower() == folder_lower:
                return idx
        
        # Try partial match
        for disease_name, idx in disease_classes.items():
            disease_lower = disease_name.lower()
            if folder_lower in disease_lower or disease_lower in folder_lower:
                return idx
        
        return None
    
    def get_images_by_class(self):
        """Return dict mapping class_idx -> list of image paths"""
        class_images = defaultdict(list)
        for img_path, label in zip(self.images, self.labels):
            class_images[label].append(img_path)
        return class_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_test_images_by_class(test_dataset):
    """Extract images from test dataset organized by class"""
    class_images = defaultdict(list)
    
    print("Organizing test images by class...")
    for idx in range(len(test_dataset)):
        img_path = test_dataset.image_paths[idx]
        label = test_dataset.disease_classes[test_dataset.labels[idx]]
        class_images[label].append(img_path)
    
    return class_images


def find_latest_timestamp(base_dir, model_name):
    """Find latest timestamp folder for a model"""
    model_dir = Path(base_dir) / model_name
    if not model_dir.exists():
        return None
    
    timestamps = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.count('_') >= 1]
    if not timestamps:
        return None
    
    timestamps.sort(reverse=True)
    return timestamps[0]


def find_all_models(base_dir):
    """Find all model directories with timestamps"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    models = []
    default_models = ['cgan', 'conditional_diffusion', 'prebuilt_diffusion', 'prebuilt_gan']
    
    for model_name in default_models:
        model_dir = base_path / model_name
        if model_dir.exists():
            timestamp = find_latest_timestamp(base_dir, model_name)
            if timestamp:
                models.append((model_name, timestamp))
    
    return models


def load_classifier(checkpoint_path, device):
    """Load trained classifier"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    disease_classes = checkpoint['disease_classes']
    num_classes = checkpoint['num_classes']
    
    model = SkinDiseaseClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, disease_classes


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def evaluate_classification(model, loader, device, disease_classes):
    """Evaluate classification metrics"""
    print("\n" + "="*70)
    print("CLASSIFICATION METRICS")
    print("="*70)
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics, preds, labels = evaluate(model, loader, criterion, device)
    
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    overall = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'loss': float(metrics['loss'])
    }
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {overall['accuracy']:.2f}%")
    print(f"  Precision: {overall['precision']:.2f}%")
    print(f"  Recall:    {overall['recall']:.2f}%")
    print(f"  F1 Score:  {overall['f1']:.2f}%")
    
    per_class = {}
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<35} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<8}")
    print("-" * 70)
    
    for cls_idx in range(len(disease_classes)):
        class_name = class_names[cls_idx]
        per_class[class_name] = {
            'precision': float(metrics['per_class']['precision'][cls_idx]),
            'recall': float(metrics['per_class']['recall'][cls_idx]),
            'f1': float(metrics['per_class']['f1'][cls_idx]),
            'support': int(metrics['per_class']['support'][cls_idx])
        }
        
        p = per_class[class_name]
        print(f"{class_name:<35} {p['precision']:>6.2f}% {p['recall']:>6.2f}% {p['f1']:>6.2f}% {p['support']:>8d}")
    
    return {'overall': overall, 'per_class': per_class}


# ============================================================================
# FID
# ============================================================================

class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        self.model = inception.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def extract_features(images, feature_extractor, device, batch_size=32):
    features = []
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting features", leave=False):
        batch = images[i:i+batch_size].to(device)
        feat = feature_extractor(batch)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_images, gen_images, device):
    feature_extractor = InceptionFeatureExtractor(device)
    
    real_features = extract_features(real_images, feature_extractor, device)
    gen_features = extract_features(gen_images, feature_extractor, device)
    
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return float(fid)


def evaluate_fid(test_images_by_class, gen_dataset, device, disease_classes):
    print("\n" + "="*70)
    print("FID (Fréchet Inception Distance)")
    print("="*70)
    
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gen_by_class = gen_dataset.get_images_by_class()
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    per_class_fid = {}
    all_real_images = []
    all_gen_images = []
    
    print("\nComputing per-class FID:")
    for cls_idx in range(len(disease_classes)):
        class_name = class_names[cls_idx]
        
        real_paths = test_images_by_class.get(cls_idx, [])
        gen_paths = gen_by_class.get(cls_idx, [])
        
        if len(real_paths) == 0 or len(gen_paths) == 0:
            print(f"  {class_name}: SKIPPED (insufficient images)")
            continue
        
        real_imgs = torch.stack([inception_transform(Image.open(p).convert('RGB')) for p in real_paths])
        gen_imgs = torch.stack([inception_transform(Image.open(p).convert('RGB')) for p in gen_paths])
        
        all_real_images.append(real_imgs)
        all_gen_images.append(gen_imgs)
        
        fid = compute_fid(real_imgs, gen_imgs, device)
        per_class_fid[class_name] = fid
        print(f"  {class_name}: {fid:.2f}")
    
    print("\nComputing overall FID...")
    all_real = torch.cat(all_real_images, dim=0)
    all_gen = torch.cat(all_gen_images, dim=0)
    overall_fid = compute_fid(all_real, all_gen, device)
    print(f"  Overall FID: {overall_fid:.2f}")
    
    return {'overall': overall_fid, 'per_class': per_class_fid}


# ============================================================================
# IS
# ============================================================================

def compute_inception_score(images, device, splits=10):
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    
    preds = []
    for i in tqdm(range(0, len(images), 32), desc="Computing IS", leave=False):
        batch = images[i:i+32].to(device)
        with torch.no_grad():
            pred = torch.nn.functional.softmax(inception(batch), dim=1)
        preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))
    
    return float(np.mean(split_scores)), float(np.std(split_scores))


def evaluate_is(gen_dataset, device, disease_classes):
    print("\n" + "="*70)
    print("IS (Inception Score)")
    print("="*70)
    
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gen_by_class = gen_dataset.get_images_by_class()
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    per_class_is = {}
    all_gen_images = []
    
    print("\nComputing per-class IS:")
    for cls_idx in range(len(disease_classes)):
        class_name = class_names[cls_idx]
        gen_paths = gen_by_class.get(cls_idx, [])
        
        if len(gen_paths) < 10:
            print(f"  {class_name}: SKIPPED (need at least 10 images)")
            continue
        
        gen_imgs = torch.stack([inception_transform(Image.open(p).convert('RGB')) for p in gen_paths])
        all_gen_images.append(gen_imgs)
        
        is_mean, is_std = compute_inception_score(gen_imgs, device)
        per_class_is[class_name] = {'mean': is_mean, 'std': is_std}
        print(f"  {class_name}: {is_mean:.2f} ± {is_std:.2f}")
    
    print("\nComputing overall IS...")
    if all_gen_images:
        all_gen = torch.cat(all_gen_images, dim=0)
        overall_mean, overall_std = compute_inception_score(all_gen, device)
        print(f"  Overall IS: {overall_mean:.2f} ± {overall_std:.2f}")
    else:
        overall_mean, overall_std = 0.0, 0.0
        print(f"  Overall IS: N/A (no images)")
    
    return {
        'overall': {'mean': overall_mean, 'std': overall_std},
        'per_class': per_class_is
    }


# ============================================================================
# LPIPS
# ============================================================================

def compute_lpips_for_class(real_paths, gen_paths, lpips_model, device, transform):
    real_count = len(real_paths)
    gen_count = len(gen_paths)
    num_pairs = min(real_count, gen_count, 200)
    
    if num_pairs == 0:
        return None, 0
    
    real_sample = np.random.choice(real_paths, num_pairs, replace=False)
    gen_sample = np.random.choice(gen_paths, num_pairs, replace=False)
    
    distances = []
    for real_path, gen_path in tqdm(zip(real_sample, gen_sample), 
                                     total=num_pairs, 
                                     desc="Computing LPIPS", 
                                     leave=False):
        real_img = transform(Image.open(real_path).convert('RGB')).unsqueeze(0).to(device)
        gen_img = transform(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            dist = lpips_model(real_img, gen_img)
        distances.append(dist.item())
    
    return float(np.mean(distances)), num_pairs


def evaluate_lpips(test_images_by_class, gen_dataset, device, disease_classes):
    print("\n" + "="*70)
    print("LPIPS (Learned Perceptual Image Patch Similarity)")
    print("="*70)
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    lpips_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    gen_by_class = gen_dataset.get_images_by_class()
    class_names = {idx: name for name, idx in disease_classes.items()}
    
    per_class_lpips = {}
    pair_counts = {}
    all_distances = []
    
    print("\nComputing per-class LPIPS:")
    print(f"{'Class':<35} {'Real':<8} {'Gen':<8} {'Pairs':<8} {'LPIPS':<10}")
    print("-" * 70)
    
    for cls_idx in range(len(disease_classes)):
        class_name = class_names[cls_idx]
        
        real_paths = test_images_by_class.get(cls_idx, [])
        gen_paths = gen_by_class.get(cls_idx, [])
        
        real_count = len(real_paths)
        gen_count = len(gen_paths)
        
        if real_count == 0 or gen_count == 0:
            print(f"{class_name:<35} {real_count:<8} {gen_count:<8} {'SKIP':<8} {'N/A':<10}")
            continue
        
        lpips_score, num_pairs = compute_lpips_for_class(
            real_paths, gen_paths, lpips_model, device, lpips_transform
        )
        
        per_class_lpips[class_name] = lpips_score
        pair_counts[class_name] = num_pairs
        all_distances.extend([lpips_score] * num_pairs)
        
        print(f"{class_name:<35} {real_count:<8} {gen_count:<8} {num_pairs:<8} {lpips_score:<10.4f}")
    
    overall_lpips = float(np.mean(all_distances)) if all_distances else 0.0
    print(f"\n  Overall LPIPS: {overall_lpips:.4f}")
    
    return {
        'overall': overall_lpips,
        'per_class': per_class_lpips,
        'pair_counts': pair_counts
    }


# ============================================================================
# EVALUATE SINGLE MODEL
# ============================================================================

def evaluate_model(model_name, timestamp, classifier_model, disease_classes, 
                   test_images_by_class, generated_base_dir, device, 
                   batch_size=32, img_size=224):
    """Evaluate a single model"""
    
    print("\n" + "="*80)
    print(f"EVALUATING: {model_name} ({timestamp})")
    print("="*80)
    
    gen_dir = Path(generated_base_dir) / model_name / timestamp
    
    if not gen_dir.exists():
        print(f"❌ Directory not found: {gen_dir}")
        return None
    
    # Create output directory
    output_dir = Path(f"logs/eval/{model_name}/{timestamp}")
    classification_dir = output_dir / "classification"
    quality_dir = output_dir / "image_quality"
    fid_dir = quality_dir / "fid"
    is_dir = quality_dir / "is"
    lpips_dir = quality_dir / "lpips"
    
    for d in [classification_dir, fid_dir, is_dir, lpips_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Setup transforms
    classifier_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load generated dataset
    gen_dataset = ImageDataset(gen_dir, disease_classes, classifier_transform)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    classification_results = evaluate_classification(classifier_model, gen_loader, device, disease_classes)
    fid_results = evaluate_fid(test_images_by_class, gen_dataset, device, disease_classes)
    is_results = evaluate_is(gen_dataset, device, disease_classes)
    lpips_results = evaluate_lpips(test_images_by_class, gen_dataset, device, disease_classes)
    
    # Save results
    with open(classification_dir / "overall_metrics.json", 'w') as f:
        json.dump(classification_results['overall'], f, indent=2)
    with open(classification_dir / "per_class_metrics.json", 'w') as f:
        json.dump(classification_results['per_class'], f, indent=2)
    
    with open(fid_dir / "overall.json", 'w') as f:
        json.dump({'fid': fid_results['overall']}, f, indent=2)
    with open(fid_dir / "per_class.json", 'w') as f:
        json.dump(fid_results['per_class'], f, indent=2)
    
    with open(is_dir / "overall.json", 'w') as f:
        json.dump(is_results['overall'], f, indent=2)
    with open(is_dir / "per_class.json", 'w') as f:
        json.dump(is_results['per_class'], f, indent=2)
    
    with open(lpips_dir / "overall.json", 'w') as f:
        json.dump({'lpips': lpips_results['overall']}, f, indent=2)
    with open(lpips_dir / "per_class.json", 'w') as f:
        json.dump(lpips_results['per_class'], f, indent=2)
    with open(lpips_dir / "pair_counts.json", 'w') as f:
        json.dump(lpips_results['pair_counts'], f, indent=2)
    
    # Summary
    summary = {
        'model': model_name,
        'timestamp': timestamp,
        'classification': classification_results['overall'],
        'fid': fid_results['overall'],
        'is': is_results['overall'],
        'lpips': lpips_results['overall']
    }
    
    with open(quality_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation of Generated Images")
    parser.add_argument("--classifier", type=str, default="checkpoints/cnn_evaluator/best_model.pt",
                       help="Path to trained classifier checkpoint")
    parser.add_argument("--generated_base_dir", type=str, default="output",
                       help="Base directory containing model outputs")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                       help="Specific models to evaluate (default: all found)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    
    # Load environment and data
    load_dotenv()
    DATA_DIR = os.getenv('DATA_DIR')
    
    print("\nLoading test dataset...")
    _, _, test_loader, disease_classes = create_data_loaders(
        ham_metadata_path=os.path.join(DATA_DIR, "HAM10000_metadata"),
        ham_img_part1=os.path.join(DATA_DIR, "HAM10000_images"),
        ham_img_part2=os.path.join(DATA_DIR, "HAM10000_images"),
        bcn_metadata_path=os.path.join(DATA_DIR, "ISIC_metadata.csv"),
        bcn_img_dir=os.path.join(DATA_DIR, "ISIC_images"),
        batch_size=args.batch_size,
        img_size=args.img_size,
        top_n_classes=None
    )
    
    # Extract test images by class
    test_images_by_class = get_test_images_by_class(test_loader.dataset)
    
    # Load classifier
    print(f"\nLoading classifier from {args.classifier}")
    classifier_model, _ = load_classifier(args.classifier, device)
    
    # Find models to evaluate
    print(f"\nLoading models from {args.generated_base_dir}")
    if args.models:
        models_to_eval = []
        for model_name in args.models:
            timestamp = find_latest_timestamp(args.generated_base_dir, model_name)
            if timestamp:
                models_to_eval.append((model_name, timestamp))
            else:
                print(f"⚠️  No timestamp found for {model_name}")
    else:
        models_to_eval = find_all_models(args.generated_base_dir)
    
    if not models_to_eval:
        print("❌ No models found to evaluate")
        return
    
    print(f"\nFound {len(models_to_eval)} model(s) to evaluate:")
    for model_name, timestamp in models_to_eval:
        print(f"  - {model_name} ({timestamp})")
    
    # Evaluate all models
    all_summaries = []
    for model_name, timestamp in models_to_eval:
        summary = evaluate_model(
            model_name, timestamp, 
            classifier_model, disease_classes,
            test_images_by_class,
            args.generated_base_dir, device, 
            args.batch_size, args.img_size
        )
        if summary:
            all_summaries.append(summary)
    
    # Create comparison table
    if len(all_summaries) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<25} {'F1':<10} {'FID':<10} {'IS':<15} {'LPIPS':<10}")
        print("-" * 80)
        for s in all_summaries:
            print(f"{s['model']:<25} {s['classification']['f1']:>7.2f}%  {s['fid']:>8.2f}  "
                  f"{s['is']['mean']:>6.2f}±{s['is']['std']:.2f}  {s['lpips']:>8.4f}")
    
    # Save summary to JSON
    summary_file = Path("../app-streamlit/data/summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Summary saved to {summary_file}")
    
    # Save to CSV
    csv_file = Path("../app-streamlit/data/summary.csv")
    if all_summaries:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'timestamp', 'f1', 'fid', 'is_mean', 'is_std', 'lpips'])
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({
                    'model': s['model'],
                    'timestamp': s['timestamp'],
                    'f1': s['classification']['f1'],
                    'fid': s['fid'],
                    'is_mean': s['is']['mean'],
                    'is_std': s['is']['std'],
                    'lpips': s['lpips']
                })
        print(f"✓ CSV summary saved to {csv_file}")
    
    print("\n✓ All evaluations complete")


if __name__ == "__main__":
    main()