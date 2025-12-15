"""
Face-based Deepfake Classifier
CNN-based classifier using EfficientNet backbone for face deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from pathlib import Path
import warnings


class FaceDeepfakeClassifier(nn.Module):
    """
    Face-based deepfake classifier using EfficientNet backbone.
    Classifies face regions as Real or AI-Generated.
    """
    
    def __init__(self, 
                 backbone: str = "efficientnet_b0",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Initialize face classifier.
        
        Args:
            backbone: Model architecture (efficientnet_b0, efficientnet_b1, etc.)
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Try to load EfficientNet from timm
        try:
            import timm
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
            self.feature_dim = self.backbone.num_features
            print(f"✓ Face classifier initialized: {backbone} (features={self.feature_dim})")
        except ImportError:
            warnings.warn("timm not available. Install with: pip install timm")
            # Fallback to simple CNN
            self.backbone = self._create_simple_backbone()
            self.feature_dim = 512
            print("✓ Face classifier initialized: Simple CNN (timm not available)")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _create_simple_backbone(self) -> nn.Module:
        """Create a simple CNN backbone when timm is not available"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


class FaceClassifierInference:
    """
    Inference wrapper for face deepfake classification.
    Handles preprocessing and provides easy-to-use interface.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize face classifier for inference.
        
        Args:
            model_path: Path to trained model weights
            device: "auto", "cuda", "cpu", or "mps"
            input_size: Input image size
        """
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        
        # Initialize model
        self.model = FaceDeepfakeClassifier()
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing parameters (ImageNet normalization)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"✓ Face classifier weights loaded from {model_path}")
    
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for inference.
        
        Args:
            face_image: RGB face image as numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert to float and normalize
        face_float = face_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        face_normalized = (face_float - self.mean) / self.std
        
        # Convert to tensor (HWC -> CHW)
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        return face_tensor.unsqueeze(0)
    
    def predict(self, face_image: np.ndarray) -> Dict:
        """
        Predict if face is real or AI-generated.
        
        Args:
            face_image: RGB face image as numpy array
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Preprocess
        face_tensor = self.preprocess(face_image).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0, prediction].item()
        
        # Map prediction to label
        label = "AI-Generated" if prediction == 1 else "Real"
        
        return {
            'prediction': label,
            'prediction_idx': prediction,
            'confidence': confidence,
            'real_probability': probs[0, 0].item(),
            'ai_generated_probability': probs[0, 1].item()
        }
    
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face features for ensemble classification.
        
        Args:
            face_image: RGB face image
            
        Returns:
            Feature vector
        """
        # Preprocess
        face_tensor = self.preprocess(face_image).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.backbone(face_tensor)
        
        return features.cpu().numpy().flatten()
    
    def get_feature_dimensions(self) -> int:
        """Get dimension of face features"""
        return self.model.feature_dim
