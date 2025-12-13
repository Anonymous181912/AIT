"""
Main pipeline for deepfake detection
Orchestrates feature extraction, classification, and inference
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

from background_features import BackgroundFeatureExtractor
from classifier import BackgroundAuthenticityClassifier, EnsembleClassifier
from preprocessing import ImagePreprocessor


class DeepfakeDetectionPipeline:
    """
    End-to-end pipeline for deepfake detection using background authenticity
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cpu",
                 use_face_detection: bool = False):
        self.device = torch.device(device)
        self.use_face_detection = use_face_detection
        
        # Initialize feature extractor
        self.feature_extractor = BackgroundFeatureExtractor()
        
        # Get feature dimensions
        feature_dims = self.feature_extractor.get_feature_dimensions()
        self.input_dim = feature_dims['total']
        
        # Initialize model
        self.model = BackgroundAuthenticityClassifier(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_classes=2,
            dropout_rate=0.3
        ).to(self.device)
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Model needs to be trained first.")
    
    def extract_features(self, image_path: str, 
                        face_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Extract background features from image"""
        signature = self.feature_extractor.extract_unified_signature(
            image_path, face_bbox
        )
        return signature
    
    def predict(self, image_path: str, 
               face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, any]:
        """
        Predict if image is real or AI-generated
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Extract features
        features = self.extract_features(image_path, face_bbox)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0, prediction].item()
        
        # Map prediction to label
        label = "AI-Generated" if prediction == 1 else "Real Camera Image"
        
        return {
            'prediction': label,
            'confidence': confidence,
            'real_probability': probs[0, 0].item(),
            'ai_generated_probability': probs[0, 1].item(),
            'features_used': {
                'frequency': True,
                'noise': True,
                'metadata': True
            }
        }
    
    def predict_batch(self, image_paths: list, 
                     face_bboxes: Optional[list] = None) -> list:
        """Predict for multiple images"""
        results = []
        for i, image_path in enumerate(image_paths):
            face_bbox = face_bboxes[i] if face_bboxes else None
            result = self.predict(image_path, face_bbox)
            result['image_path'] = image_path
            results.append(result)
        return results
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str, epoch: int = 0, 
                   train_loss: float = 0.0, val_accuracy: float = 0.0):
        """Save trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'input_dim': self.input_dim,
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
    
    def explain_prediction(self, image_path: str, 
                          face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Provide explanation for prediction by showing feature contributions
        """
        # Extract all features
        features = self.feature_extractor.extract_all_features(image_path, face_bbox)
        
        # Get feature dimensions
        dims = self.feature_extractor.get_feature_dimensions()
        
        # Convert to tensor
        full_signature = self.feature_extractor.extract_unified_signature(image_path, face_bbox)
        features_tensor = torch.from_numpy(full_signature).float().unsqueeze(0).to(self.device)
        
        # Feature importance (gradient-based)
        features_tensor.requires_grad = True
        self.model.eval()
        logits = self.model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Compute gradients for AI-generated class
        target_class = 1  # AI-generated
        probs[0, target_class].backward()
        gradients = features_tensor.grad[0].cpu().numpy()
        
        # Split gradients by feature type
        freq_end = dims['frequency']
        noise_end = freq_end + dims['noise']
        
        freq_importance = np.abs(gradients[:freq_end]).mean()
        noise_importance = np.abs(gradients[freq_end:noise_end]).mean()
        metadata_importance = np.abs(gradients[noise_end:]).mean()
        
        explanation = {
            'prediction': "AI-Generated" if torch.argmax(probs, dim=1).item() == 1 else "Real",
            'confidence': probs[0, torch.argmax(probs, dim=1).item()].item(),
            'feature_contributions': {
                'frequency_features': float(freq_importance),
                'noise_features': float(noise_importance),
                'metadata_features': float(metadata_importance),
            },
            'feature_statistics': {
                'frequency_mean': float(features['frequency'].mean()),
                'noise_mean': float(features['noise'].mean()),
                'metadata_mean': float(features['metadata'].mean()),
            }
        }
        
        return explanation

