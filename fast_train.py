"""
High-Performance Training Script with Parallel Processing
Optimized for: Speed, Multiprocessing, and Fast Iteration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, Tuple, Optional
import time

# --- Custom Imports ---
# Ensure these files are in the same directory
from background_features import BackgroundFeatureExtractor
from advanced_classifier import AdvancedDeepfakeClassifier, LabelSmoothingCrossEntropy
from augmentations import RandomAugmentMixer
from metrics import MetricsCalculator, EarlyStopping
from config import (
    DATASET_DIR, MODEL_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE, TRAIN_VAL_SPLIT,
    get_device
)

# --- Hardware Optimization ---
# Enable fast matrix multiplication on NVIDIA GPUs (Ampere and newer)
torch.set_float32_matmul_precision('high')

class FastDeepfakeDataset(Dataset):
    """Optimized dataset that keeps data ready in memory"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, augment: bool = False):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).long()
        self.augment = augment
        self.noise_std = 0.02
        self.scale_range = (0.95, 1.05)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # fast direct access
        features = self.features[idx]
        target = self.targets[idx]
        
        # Lightweight Augmentations (CPU)
        if self.augment:
            # 50% chance of noise
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(features) * self.noise_std
                features = features + noise
            
            # 50% chance of scaling
            if torch.rand(1) < 0.5:
                scale = torch.empty(1).uniform_(*self.scale_range)
                features = features * scale
        
        return features, target

# --- Parallel Processing Worker ---
def _process_single_image(args):
    """
    Worker function that runs on a separate CPU core.
    Extracts features for one image.
    """
    path, label = args
    try:
        # We re-initialize the extractor here to ensure thread safety
        # and avoid complex pickling issues with multiprocessing
        extractor = BackgroundFeatureExtractor(use_texture_features=True)
        feat = extractor.extract_unified_signature(path)
        return feat, label
    except Exception as e:
        # Return None on failure so the main loop can skip it
        return None

def extract_features_parallel(data_dir: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts features using ALL CPU cores.
    Drastically reduces time from hours to minutes.
    """
    data_dir = Path(data_dir)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    
    # 1. Gather all image paths first
    tasks = []
    print("📂 Scanning dataset directories...")
    for label, sub_dir in enumerate(["real", "fake"]):
        dir_path = data_dir / sub_dir
        if dir_path.exists():
            for ext in image_extensions:
                tasks.extend([(str(p), label) for p in dir_path.glob(ext)])
    
    print(f"✅ Found {len(tasks)} images.")
    
    # 2. Run Parallel Extraction
    # Use 75% of available cores to keep the system responsive, or all if on a server
    max_workers = max(1, os.cpu_count() - 2) 
    print(f"🚀 Starting extraction using {max_workers} CPU cores...")
    
    features = []
    targets = []
    
    start_time = time.time()
    
    # ProcessPoolExecutor manages the worker processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map returns results in order
        results = list(tqdm(
            executor.map(_process_single_image, tasks), 
            total=len(tasks), 
            desc="Extracting Features",
            unit="img"
        ))
    
    # 3. Filter valid results
    for res in results:
        if res is not None:
            features.append(res[0])
            targets.append(res[1])
            
    elapsed = time.time() - start_time
    print(f"✅ Extraction complete in {elapsed/60:.2f} minutes.")
            
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    
    # Normalize
    print("Scale & Normalize...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, targets, scaler

def train_epoch_fast(model, dataloader, criterion, optimizer, device, scaler, augmenter):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Minimal progress bar overhead
    pbar = tqdm(dataloader, desc="Train", leave=False, mininterval=1.0)
    
    for features, targets in pbar:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixup / CutMix
        use_mixing = False
        if augmenter is not None and np.random.random() < 0.5:
             features, targets_a, targets_b, lam = augmenter(features, targets)
             use_mixing = True
        
        optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
        
        # Automatic Mixed Precision (AMP)
        with autocast('cuda', enabled=True):
            outputs = model(features)
            if use_mixing:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Detach metrics to avoid VRAM buildup
        total_loss += loss.detach() 
        _, predicted = torch.max(outputs.detach(), 1)
        total += targets.size(0)
        correct += (predicted == targets).sum()
    
    return total_loss.item() / len(dataloader), (correct.float() / total * 100).item()

@torch.no_grad()
def validate_fast(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, targets in dataloader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=True):
            outputs = model(features)
            loss = criterion(outputs, targets)
            
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    return total_loss / len(dataloader), (correct / total * 100)

def main():
    print("=" * 60)
    print("⚡ FAST DEEPFAKE TRAINING (Parallel + Cached) ⚡")
    print("=" * 60)
    
    # 1. Device Setup
    device = get_device()
    print(f"Device: {device}")
    
    # Enable cudnn benchmark for consistent input sizes (very fast)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 2. Config
    config = {
        'batch_size': 32,          # Good balance for 3050 GPU
        'epochs': 30,              # Enough for convergence with SWA
        'learning_rate': 2e-3,     # Slightly higher LR for AdamW
        'weight_decay': 0.01,
        'embed_dim': 256,
        'num_heads': 4,
        'num_transformer_blocks': 2,
        'num_residual_blocks': 2,
        'dropout': 0.2,
        'patience': 8
    }
    
    # 3. Data Loading with CACHING
    cache_path = os.path.join(MODEL_DIR, "features_cache_v2.pkl")
    
    if os.path.exists(cache_path):
        print(f"\n📦 Found cached features at: {cache_path}")
        print("   Loading directly (Skipping extraction)...")
        with open(cache_path, 'rb') as f:
            features, targets, scaler = pickle.load(f)
        print("   ✅ Loaded!")
    else:
        print("\n🔍 No cache found. Starting parallel extraction...")
        features, targets, scaler = extract_features_parallel(DATASET_DIR)
        
        # Save cache
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((features, targets, scaler), f)
        print(f"   💾 Saved features to {cache_path}")

    # 4. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.2, stratify=targets, random_state=42
    )
    
    train_dataset = FastDeepfakeDataset(X_train, y_train, augment=True)
    val_dataset = FastDeepfakeDataset(X_val, y_val, augment=False)
    
    # 5. Dataloaders
    # num_workers=2 is usually safe for Windows. If it crashes, set to 0.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # 6. Model Init
    print("\n🧠 Initializing Model...")
    model = AdvancedDeepfakeClassifier(
        input_dim=features.shape[1],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_transformer_blocks=config['num_transformer_blocks'],
        num_residual_blocks=config['num_residual_blocks'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)
    
    # 7. Compile (Optional Speedup)
    try:
        print("   Compiling model graph (torch.compile)...")
        model = torch.compile(model) 
    except Exception as e:
        print(f"   Warning: Could not compile model. Running in standard mode. ({e})")

    # 8. Optimization Setup
    # Use fused AdamW if on CUDA for extra speed
    use_fused = (device.type == 'cuda')
    try:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'], fused=use_fused)
    except:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'])
        
    scaler = GradScaler()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    augmenter = RandomAugmentMixer(mixup_alpha=0.4, cutmix_alpha=1.0)
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'], 
        steps_per_epoch=len(train_loader), epochs=config['epochs']
    )
    
    # SWA (Stochastic Weight Averaging)
    swa_model = AveragedModel(model)
    swa_start = int(config['epochs'] * 0.75)

    # 9. Training Loop
    print(f"\n🏃 Starting Training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch_fast(
            model, train_loader, criterion, optimizer, device, scaler, augmenter
        )
        val_loss, val_acc = validate_fast(model, val_loader, criterion, device)
        
        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        else:
            scheduler.step()
            
        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if early_stopping(val_acc, model, epoch):
            print("🛑 Early stopping triggered!")
            break

    # 10. Finalize
    if epoch >= swa_start:
        print("🔄 Updating Batch Norm for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        final_model = swa_model.module
    else:
        final_model = model

    save_path = os.path.join(MODEL_DIR, "best_model_fast.pth")
    torch.save(final_model.state_dict(), save_path)
    print(f"\n✅ Training Complete. Model saved to: {save_path}")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()