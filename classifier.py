"""
Deepfake classification model
Neural network for classifying images as real or AI-generated based on background features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Optional


class BackgroundAuthenticityClassifier(nn.Module):
    """
    Neural network classifier for background authenticity
    Takes background features and classifies as real or AI-generated
    """
    
    def __init__(self, 
                 input_dim: int = 50,  # Will be set based on feature dimensions
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        super(BackgroundAuthenticityClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input feature tensor of shape (batch_size, input_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Process features
        features = self.feature_processor(x)
        
        # Classify
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
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EnsembleClassifier:
    """
    Ensemble classifier combining background and face-based detection
    """
    
    def __init__(self, 
                 background_model: BackgroundAuthenticityClassifier,
                 face_model: Optional[nn.Module] = None,
                 background_weight: float = 0.6,
                 face_weight: float = 0.4):
        self.background_model = background_model
        self.face_model = face_model
        self.background_weight = background_weight
        self.face_weight = face_weight
    
    def predict(self, background_features: torch.Tensor, 
                face_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make ensemble prediction
        """
        # Background prediction
        bg_probs = F.softmax(self.background_model(background_features), dim=1)
        
        if self.face_model is not None and face_features is not None:
            # Face prediction
            face_probs = F.softmax(self.face_model(face_features), dim=1)
            
            # Weighted ensemble
            ensemble_probs = (self.background_weight * bg_probs + 
                            self.face_weight * face_probs)
        else:
            ensemble_probs = bg_probs
        
        predictions = torch.argmax(ensemble_probs, dim=1)
        return predictions, ensemble_probs

