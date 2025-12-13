"""
Training script for background authenticity classifier
Fully automatic and foolproof training pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import sys

from background_features import BackgroundFeatureExtractor
from classifier import BackgroundAuthenticityClassifier
from pipeline import DeepfakeDetectionPipeline
from config import *
from auto_download import ensure_dataset_exists


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection"""
    
    def __init__(self, data_dir: str, label_file: str = None, scaler: StandardScaler = None):
        self.data_dir = Path(data_dir)
        self.scaler = scaler
        
        # Auto-detect labels from directory structure
        self.labels = self._auto_detect_labels()
        
        if len(self.labels) == 0:
            raise ValueError(f"No images found in {data_dir}. Expected structure: {data_dir}/real/ and {data_dir}/fake/")
        
        # Extract features for all images
        self.feature_extractor = BackgroundFeatureExtractor()
        self.features = []
        self.targets = []
        
        print("Extracting features from images...")
        for image_path, label in tqdm(self.labels.items(), desc="Processing images"):
            try:
                features = self.feature_extractor.extract_unified_signature(image_path)
                self.features.append(features)
                self.targets.append(label)
            except Exception as e:
                print(f"\nWarning: Error processing {image_path}: {e}")
                print("Skipping this image...")
                continue
        
        if len(self.features) == 0:
            raise ValueError("No valid images could be processed. Please check your image files.")
        
        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        
        # Normalize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features = self.scaler.transform(self.features)
        
        print(f"\n✓ Loaded {len(self.features)} samples")
        class_counts = np.bincount(self.targets)
        print(f"✓ Class distribution: Real={class_counts[0]}, Fake={class_counts[1] if len(class_counts) > 1 else 0}")
    
    def _auto_detect_labels(self) -> dict:
        """Auto-detect labels from directory structure"""
        labels = {}
        
        # Expected structure: dataset/real/ and dataset/fake/
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"
        
        # Also check for data/train/real and data/train/fake (legacy)
        if not real_dir.exists() and not fake_dir.exists():
            real_dir = self.data_dir / "train" / "real"
            fake_dir = self.data_dir / "train" / "fake"
        
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        
        if real_dir.exists():
            for ext in image_extensions:
                for img_path in real_dir.glob(ext):
                    labels[str(img_path)] = 0  # Real = 0
        
        if fake_dir.exists():
            for ext in image_extensions:
                for img_path in fake_dir.glob(ext):
                    labels[str(img_path)] = 1  # Fake = 1
        
        return labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.targets[idx]])[0]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for features, targets in tqdm(dataloader, desc="  Training", leave=False):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="  Validating", leave=False):
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def check_dataset_exists():
    """Check if dataset directory exists and has required structure"""
    dataset_path = Path(DATASET_DIR)
    
    if not dataset_path.exists():
        print("=" * 60)
        print("ERROR: Dataset directory not found!")
        print("=" * 60)
        print(f"\nExpected directory structure:")
        print(f"  {DATASET_DIR}/")
        print(f"    ├── real/    (put real camera images here)")
        print(f"    └── fake/    (put AI-generated images here)")
        print(f"\nSupported image formats: .jpg, .jpeg, .png")
        print(f"\nPlease create the dataset directory and add your images.")
        print("=" * 60)
        return False
    
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"
    
    if not real_dir.exists() and not fake_dir.exists():
        print("=" * 60)
        print("ERROR: Dataset structure incorrect!")
        print("=" * 60)
        print(f"\nFound: {dataset_path}")
        print(f"Expected: {dataset_path}/real/ and {dataset_path}/fake/")
        print(f"\nPlease organize your images as:")
        print(f"  {DATASET_DIR}/real/  (real camera images)")
        print(f"  {DATASET_DIR}/fake/  (AI-generated images)")
        print("=" * 60)
        return False
    
    # Count images
    real_count = 0
    fake_count = 0
    
    if real_dir.exists():
        real_count = len(list(real_dir.glob("*.jpg")) + 
                        list(real_dir.glob("*.jpeg")) + 
                        list(real_dir.glob("*.png")) +
                        list(real_dir.glob("*.JPG")) + 
                        list(real_dir.glob("*.JPEG")) + 
                        list(real_dir.glob("*.PNG")))
    
    if fake_dir.exists():
        fake_count = len(list(fake_dir.glob("*.jpg")) + 
                        list(fake_dir.glob("*.jpeg")) + 
                        list(fake_dir.glob("*.png")) +
                        list(fake_dir.glob("*.JPG")) + 
                        list(fake_dir.glob("*.JPEG")) + 
                        list(fake_dir.glob("*.PNG")))
    
    if real_count == 0 and fake_count == 0:
        print("=" * 60)
        print("ERROR: No images found in dataset!")
        print("=" * 60)
        print(f"\nFound directories but no images in:")
        print(f"  {real_dir} ({real_count} images)")
        print(f"  {fake_dir} ({fake_count} images)")
        print(f"\nPlease add image files (.jpg, .jpeg, .png) to these directories.")
        print("=" * 60)
        return False
    
    print(f"\n✓ Dataset found: {real_count} real images, {fake_count} fake images")
    return True


def main():
    """Main training function - fully automatic"""
    print("=" * 60)
    print("Deepfake Detection - Training Pipeline")
    print("=" * 60)
    
    # Automatically download images if dataset is empty
    print("\n🔍 Checking dataset...")
    dataset_ready = ensure_dataset_exists(DATASET_DIR, min_images=20)
    
    if not dataset_ready:
        print("\n❌ ERROR: Dataset is insufficient for training.")
        print("   Please ensure you have at least some images in dataset/real/ and dataset/fake/")
        sys.exit(1)
    
    # Verify dataset structure (after auto-download)
    if not check_dataset_exists():
        sys.exit(1)
    
    # Setup device (force CPU for foolproof training)
    device = torch.device("cpu")
    print(f"\n✓ Using device: {device}")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load dataset from dataset/ directory
    print(f"\nLoading dataset from: {DATASET_DIR}/")
    try:
        full_dataset = DeepfakeDataset(DATASET_DIR)
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        sys.exit(1)
    
    # Split into train and validation
    print("\nSplitting dataset into train/validation...")
    if len(full_dataset.features) < 10:
        print("WARNING: Very small dataset (< 10 samples). Results may be poor.")
    
    # Use train_test_split to split features and targets
    train_features, val_features, train_targets, val_targets = train_test_split(
        full_dataset.features,
        full_dataset.targets,
        test_size=1 - TRAIN_VAL_SPLIT,
        random_state=42,
        stratify=full_dataset.targets if len(np.unique(full_dataset.targets)) > 1 else None
    )
    
    print(f"✓ Training samples: {len(train_features)}")
    print(f"✓ Validation samples: {len(val_features)}")
    
    # Create datasets for train and validation
    class SplitDataset(Dataset):
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.targets[idx]])[0]
    
    train_dataset = SplitDataset(train_features, train_targets)
    val_dataset = SplitDataset(val_features, val_targets)
    
    # Save scaler for inference
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(full_dataset.scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # Initialize model
    input_dim = train_features.shape[1]
    print(f"\n✓ Initializing model (input_dim={input_dim})...")
    model = BackgroundAuthenticityClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n{'=' * 60}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'=' * 60}\n")
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%", end="")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "best_model.pth")
            
            # Save model directly (simpler and more reliable)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'input_dim': input_dim,
            }
            torch.save(checkpoint, model_path)
            print(f" ✓ [BEST - Saved]")
        else:
            print()
        
        # Periodic saves
        if (epoch + 1) % SAVE_INTERVAL == 0:
            model_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'input_dim': input_dim,
            }
            torch.save(checkpoint, model_path)
    
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"{'=' * 60}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Model saved to: {os.path.join(MODEL_DIR, 'best_model.pth')}")
    
    # Save training history
    history = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_accs': [float(x) for x in train_accs],
        'val_accs': [float(x) for x in val_accs],
        'best_val_acc': float(best_val_acc)
    }
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    print(f"\n{'=' * 60}")
    print("You can now run inference with:")
    print(f"  python infer.py --image your_image.jpg")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
