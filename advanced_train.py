"""
Advanced Training Script with State-of-the-Art Techniques
Features: Cosine Annealing, SWA, Label Smoothing, Mixup/CutMix, Early Stopping, K-Fold CV
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
import sys
from typing import Dict, List, Tuple, Optional

from background_features import BackgroundFeatureExtractor
from advanced_classifier import (
    AdvancedDeepfakeClassifier, 
    FocalLoss, 
    LabelSmoothingCrossEntropy
)
from augmentations import Mixup, CutMix, RandomAugmentMixer, TestTimeAugmentation
from metrics import MetricsCalculator, EarlyStopping, compute_optimal_threshold
from config import (
    DATASET_DIR, MODEL_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE, TRAIN_VAL_SPLIT,
    get_device, get_device_info
)


class AdvancedDeepfakeDataset(Dataset):
    """Enhanced dataset with augmentation support"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 augment: bool = False):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.augment = augment
        
        # Feature augmentations
        if augment:
            self.noise_std = 0.02
            self.scale_range = (0.95, 1.05)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]
        
        # Apply augmentation during training
        if self.augment:
            # Gaussian noise
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
            
            # Random scaling
            scale = torch.empty(1).uniform_(*self.scale_range)
            features = features * scale
        
        return features, target


def extract_all_features(data_dir: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Extract features from all images in dataset"""
    data_dir = Path(data_dir)
    feature_extractor = BackgroundFeatureExtractor(use_texture_features=True)
    
    features = []
    targets = []
    
    # Get image paths
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    
    all_images = []
    
    if real_dir.exists():
        for ext in image_extensions:
            all_images.extend([(str(p), 0) for p in real_dir.glob(ext)])
    
    if fake_dir.exists():
        for ext in image_extensions:
            all_images.extend([(str(p), 1) for p in fake_dir.glob(ext)])
    
    print(f"Found {len(all_images)} images")
    
    for image_path, label in tqdm(all_images, desc="Extracting features"):
        try:
            feat = feature_extractor.extract_unified_signature(image_path)
            features.append(feat)
            targets.append(label)
        except Exception as e:
            print(f"Warning: Error processing {image_path}: {e}")
            continue
    
    features = np.array(features)
    targets = np.array(targets)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, targets, scaler


def train_epoch_advanced(model: nn.Module, 
                         dataloader: DataLoader,
                         criterion: nn.Module,
                         optimizer: optim.Optimizer,
                         device: torch.device,
                         scaler: Optional[GradScaler] = None,
                         use_amp: bool = False,
                         augmenter: Optional[RandomAugmentMixer] = None) -> Tuple[float, float]:
    """Train for one epoch with advanced techniques"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, targets in tqdm(dataloader, desc="  Training", leave=False):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply Mixup/CutMix
        use_mixing = augmenter is not None and np.random.random() < 0.5
        
        if use_mixing:
            mixed_features, targets_a, targets_b, lam = augmenter(features, targets)
            features = mixed_features
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(features)
                if use_mixing:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            if use_mixing:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate_advanced(model: nn.Module,
                      dataloader: DataLoader,
                      criterion: nn.Module,
                      device: torch.device,
                      metrics_calc: MetricsCalculator,
                      use_amp: bool = False) -> Tuple[float, float, Dict]:
    """Validate with comprehensive metrics"""
    model.eval()
    total_loss = 0.0
    
    metrics_calc.reset()
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="  Validating", leave=False):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Update metrics
            metrics_calc.update(
                targets.cpu().numpy(),
                preds.cpu().numpy(),
                probs[:, 1].cpu().numpy()
            )
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics = metrics_calc.compute_all_metrics()
    accuracy = metrics['accuracy'] * 100
    
    return avg_loss, accuracy, metrics


def train_fold(fold: int,
               train_features: np.ndarray,
               train_targets: np.ndarray,
               val_features: np.ndarray,
               val_targets: np.ndarray,
               device: torch.device,
               config: Dict) -> Tuple[nn.Module, Dict]:
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = AdvancedDeepfakeDataset(train_features, train_targets, augment=True)
    val_dataset = AdvancedDeepfakeDataset(val_features, val_targets, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Initialize model
    input_dim = train_features.shape[1]
    model = AdvancedDeepfakeClassifier(
        input_dim=input_dim,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_transformer_blocks=config['num_transformer_blocks'],
        num_residual_blocks=config['num_residual_blocks'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config['learning_rate'] * 0.1)
    swa_start = int(config['epochs'] * 0.75)
    
    # Augmenters
    augmenter = RandomAugmentMixer(mixup_alpha=0.4, cutmix_alpha=1.0)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=0.001,
        mode='max'
    )
    
    # Metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Mixed precision
    use_amp = config['use_amp'] and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    best_val_acc = 0.0
    best_metrics = {}
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp, augmenter=augmenter
        )
        
        # Validate
        val_loss, val_acc, val_metrics = validate_advanced(
            model, val_loader, criterion, device, metrics_calc, use_amp=use_amp
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Print progress
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
              f"LR: {lr:.2e}", end="")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()
            print(" ✓ [BEST]")
        else:
            print()
        
        # Early stopping
        if early_stopping(val_acc, model, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            early_stopping.restore_best_weights(model)
            break
        
        # GPU memory management
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Update batch normalization for SWA model
    if epoch >= swa_start:
        print("Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model = swa_model.module
    
    return model, {
        'best_val_acc': best_val_acc,
        'best_metrics': best_metrics,
        'history': history
    }


def main():
    """Main training function with K-Fold CV"""
    print("=" * 60)
    print("Advanced Deepfake Detection - World-Class Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'embed_dim': 256,
        'num_heads': 4,
        'num_transformer_blocks': 2,
        'num_residual_blocks': 3,
        'dropout': 0.3,
        'label_smoothing': 0.1,
        'patience': 15,
        'n_folds': 5,
        'use_amp': True,
        'use_cv': True  # Set to False for simple train/val split
    }
    
    # Get device
    device = get_device()
    device_info = get_device_info()
    print(f"\n🖥️  Device: {device_info['device']} ({device_info['gpu_name']})")
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Extract features
    print("\n📊 Extracting features from dataset...")
    features, targets, scaler = extract_all_features(DATASET_DIR)
    
    print(f"✓ Extracted {len(features)} samples")
    print(f"✓ Feature dimension: {features.shape[1]}")
    print(f"✓ Class distribution: Real={np.sum(targets==0)}, Fake={np.sum(targets==1)}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # K-Fold Cross Validation or Simple Split
    if config['use_cv'] and len(features) >= 20:
        print(f"\n🔄 Running {config['n_folds']}-Fold Cross Validation...")
        
        kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
        
        fold_results = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features, targets)):
            train_features, val_features = features[train_idx], features[val_idx]
            train_targets, val_targets = targets[train_idx], targets[val_idx]
            
            model, results = train_fold(
                fold, train_features, train_targets, 
                val_features, val_targets, device, config
            )
            
            fold_results.append(results)
            fold_models.append(model)
            
            # Save fold model
            fold_path = os.path.join(MODEL_DIR, f"model_fold_{fold+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold,
                'val_accuracy': results['best_val_acc'],
                'metrics': results['best_metrics'],
                'input_dim': features.shape[1],
                'config': config
            }, fold_path)
        
        # Compute CV statistics
        cv_accs = [r['best_val_acc'] for r in fold_results]
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results")
        print(f"{'='*60}")
        print(f"Fold Accuracies: {[f'{a:.2f}%' for a in cv_accs]}")
        print(f"Mean Accuracy: {np.mean(cv_accs):.2f}% ± {np.std(cv_accs):.2f}%")
        
        # Save best fold as main model
        best_fold = np.argmax(cv_accs)
        best_model = fold_models[best_fold]
        
    else:
        print("\n🔀 Using train/validation split...")
        
        train_features, val_features, train_targets, val_targets = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        best_model, results = train_fold(
            0, train_features, train_targets,
            val_features, val_targets, device, config
        )
        
        fold_results = [results]
    
    # Save best model
    best_path = os.path.join(MODEL_DIR, "best_model.pth")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_dim': features.shape[1],
        'config': config,
        'cv_results': [r['best_val_acc'] for r in fold_results] if config['use_cv'] else None
    }, best_path)
    
    print(f"\n✓ Best model saved to {best_path}")
    
    # Save training config and results
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'device': str(device),
            'feature_dim': int(features.shape[1]),
            'num_samples': int(len(features)),
            'cv_accuracies': [float(r['best_val_acc']) for r in fold_results],
            'mean_accuracy': float(np.mean([r['best_val_acc'] for r in fold_results])),
            'std_accuracy': float(np.std([r['best_val_acc'] for r in fold_results])),
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print("Run inference with: python infer.py --image your_image.jpg")


if __name__ == "__main__":
    main()
