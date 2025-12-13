"""
Training script for background authenticity classifier
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
import pickle

from background_features import BackgroundFeatureExtractor
from classifier import BackgroundAuthenticityClassifier
from pipeline import DeepfakeDetectionPipeline
from config import *


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection"""
    
    def __init__(self, data_dir: str, label_file: str = None, scaler: StandardScaler = None):
        self.data_dir = Path(data_dir)
        self.scaler = scaler
        
        # Load labels
        if label_file and Path(label_file).exists():
            with open(label_file, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = self._auto_detect_labels()
        
        # Extract features
        self.feature_extractor = BackgroundFeatureExtractor()
        self.features = []
        self.targets = []
        
        print("Extracting features from images...")
        for image_path, label in tqdm(self.labels.items()):
            try:
                features = self.feature_extractor.extract_unified_signature(image_path)
                self.features.append(features)
                self.targets.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        
        # Normalize
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features = self.scaler.transform(self.features)
        
        print(f"Loaded {len(self.features)} samples")
        print(f"Class distribution: {np.bincount(self.targets)}")
    
    def _auto_detect_labels(self) -> dict:
        """Auto-detect labels from directory structure"""
        labels = {}

        real_dir = self.data_dir / "REAL"
        fake_dir = self.data_dir / "FAKE"

        # Supported image extensions (case-insensitive)
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"]

        if real_dir.exists():
            for ext in extensions:
                for img_path in real_dir.glob(ext):
                    labels[str(img_path)] = 0  # REAL

        if fake_dir.exists():
            for ext in extensions:
                for img_path in fake_dir.glob(ext):
                    labels[str(img_path)] = 1  # FAKE

        # Fail fast if no images found
        if len(labels) == 0:
            raise ValueError(
                f"No images found in dataset directory: {self.data_dir}\n"
                f"Expected structure: {self.data_dir}/REAL/ and {self.data_dir}/FAKE/\n"
                f"Supported extensions: .jpg, .jpeg, .png, .webp (case-insensitive)"
            )

        return labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.targets[idx]])[0]
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for features, targets in tqdm(dataloader, desc="Training"):
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Validating"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total


def main():
    device = torch.device(DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading training dataset...")
    train_dataset = DeepfakeDataset(TRAIN_DIR)
    
    print("Loading validation dataset...")
    val_dataset = DeepfakeDataset(TEST_DIR, scaler=train_dataset.scaler)
    
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(train_dataset.scaler, f)
    
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
    
    model = BackgroundAuthenticityClassifier(
        input_dim=train_dataset.features.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pipeline = DeepfakeDetectionPipeline(device=device)
            pipeline.model = model
            pipeline.save_model(
                os.path.join(MODEL_DIR, "best_model.pth"),
                epoch, train_loss, val_acc
            )
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
