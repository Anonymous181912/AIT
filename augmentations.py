"""
Advanced Data Augmentation Module
Features: Mixup, CutMix, Random Erasing, Test-Time Augmentation (TTA)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Callable
import random


class Mixup:
    """
    Mixup augmentation: Linearly interpolates pairs of samples and labels.
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        
    def __call__(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.
        
        Args:
            features: Input features (B, D)
            targets: One-hot or class labels (B,)
            
        Returns:
            mixed_features, targets_a, targets_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        batch_size = features.size(0)
        index = torch.randperm(batch_size, device=features.device)
        
        mixed_features = lam * features + (1 - lam) * features[index]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_features, targets_a, targets_b, lam


class CutMix:
    """
    CutMix augmentation: Cuts and pastes feature regions.
    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    
    Adapted for 1D feature vectors.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.
        
        Args:
            features: Input features (B, D)
            targets: Class labels (B,)
            
        Returns:
            mixed_features, targets_a, targets_b, lambda
        """
        batch_size, feature_dim = features.shape
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # For 1D features, cut a contiguous region
        cut_ratio = 1 - lam
        cut_size = int(feature_dim * cut_ratio)
        
        if cut_size > 0:
            cut_start = random.randint(0, feature_dim - cut_size)
            cut_end = cut_start + cut_size
            
            index = torch.randperm(batch_size, device=features.device)
            
            mixed_features = features.clone()
            mixed_features[:, cut_start:cut_end] = features[index, cut_start:cut_end]
            
            # Adjust lambda to reflect actual cut ratio
            lam = 1 - (cut_size / feature_dim)
            
            targets_a, targets_b = targets, targets[index]
        else:
            mixed_features = features
            targets_a = targets_b = targets
            lam = 1.0
        
        return mixed_features, targets_a, targets_b, lam


class FeatureDropout:
    """
    Feature-level dropout: Randomly drops feature dimensions.
    Helps with robustness to missing features.
    """
    
    def __init__(self, drop_prob: float = 0.1, feature_groups: Optional[List[Tuple[int, int]]] = None):
        self.drop_prob = drop_prob
        self.feature_groups = feature_groups  # [(start, end), ...] for group dropout
        
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
            
        if self.feature_groups:
            # Group-wise dropout
            features = features.clone()
            for start, end in self.feature_groups:
                if random.random() < self.drop_prob:
                    features[:, start:end] = 0
        else:
            # Random element dropout
            mask = torch.rand_like(features) > self.drop_prob
            features = features * mask
        
        return features
    
    @property
    def training(self):
        return True  # Override in actual usage


class GaussianNoise:
    """Add Gaussian noise to features for regularization"""
    
    def __init__(self, std: float = 0.1):
        self.std = std
        
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(features) * self.std
            return features + noise
        return features
    
    @property
    def training(self):
        return True


class FeatureScaling:
    """Random feature scaling for augmentation"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
        
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            scale = torch.empty(features.size(1), device=features.device).uniform_(*self.scale_range)
            return features * scale
        return features
    
    @property
    def training(self):
        return True


class ComposeAugmentations:
    """Compose multiple augmentations"""
    
    def __init__(self, augmentations: List[Callable], prob: float = 0.5):
        self.augmentations = augmentations
        self.prob = prob
        
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            if random.random() < self.prob:
                features = aug(features)
        return features


class TestTimeAugmentation:
    """
    Test-Time Augmentation: Apply multiple augmentations at inference
    and average the predictions.
    """
    
    def __init__(self, 
                 n_augments: int = 5,
                 noise_std: float = 0.05,
                 scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.n_augments = n_augments
        self.noise = GaussianNoise(std=noise_std)
        self.scaling = FeatureScaling(scale_range=scale_range)
        
    def augment_features(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Generate augmented versions of features"""
        augmented = [features]  # Include original
        
        for _ in range(self.n_augments - 1):
            aug_features = features.clone()
            
            # Apply noise
            aug_features = aug_features + torch.randn_like(aug_features) * 0.05
            
            # Apply scaling
            scale = torch.empty(features.size(-1), device=features.device).uniform_(0.95, 1.05)
            aug_features = aug_features * scale
            
            augmented.append(aug_features)
        
        return augmented
    
    def predict_with_tta(self, model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with TTA.
        
        Args:
            model: The model to use
            features: Input features (B, D)
            
        Returns:
            Averaged predictions (B, num_classes)
        """
        model.eval()
        augmented = self.augment_features(features)
        
        all_probs = []
        with torch.no_grad():
            for aug_features in augmented:
                logits = model(aug_features)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Average predictions
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs


def mixup_criterion(criterion: Callable, pred: torch.Tensor, 
                   targets_a: torch.Tensor, targets_b: torch.Tensor, 
                   lam: float) -> torch.Tensor:
    """Compute loss for mixup/cutmix samples"""
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


class RandomAugmentMixer:
    """
    Randomly choose between Mixup and CutMix for each batch.
    """
    
    def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0, 
                 mixup_prob: float = 0.5, cutmix_prob: float = 0.5):
        self.mixup = Mixup(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha)
        self.mixup_prob = mixup_prob / (mixup_prob + cutmix_prob)
        
    def __call__(self, features: torch.Tensor, targets: torch.Tensor):
        """Apply either Mixup or CutMix randomly"""
        if random.random() < self.mixup_prob:
            return self.mixup(features, targets)
        else:
            return self.cutmix(features, targets)


if __name__ == "__main__":
    # Test augmentations
    features = torch.randn(8, 79)
    targets = torch.randint(0, 2, (8,))
    
    # Test Mixup
    mixup = Mixup(alpha=0.4)
    mixed, t_a, t_b, lam = mixup(features, targets)
    print(f"Mixup lambda: {lam:.3f}")
    
    # Test CutMix
    cutmix = CutMix(alpha=1.0)
    cut, t_a, t_b, lam = cutmix(features, targets)
    print(f"CutMix lambda: {lam:.3f}")
    
    # Test TTA
    tta = TestTimeAugmentation(n_augments=5)
    augmented = tta.augment_features(features)
    print(f"TTA generated {len(augmented)} versions")
