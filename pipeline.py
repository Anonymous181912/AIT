"""
Main pipeline for deepfake detection
Orchestrates feature extraction, classification, and inference
Enhanced with GPU support and face integration
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import json
import pickle

from background_features import BackgroundFeatureExtractor
from classifier import BackgroundAuthenticityClassifier, EnsembleClassifier
from preprocessing import ImagePreprocessor
from config import get_device, MODEL_DIR


class DeepfakeDetectionPipeline:
    """
    End-to-end pipeline for deepfake detection using background authenticity.
    Enhanced with GPU support for faster inference.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 device: str = "auto",
                 use_face_detection: bool = False,
                 use_texture_features: bool = True):
        """
        Initialize detection pipeline.
        
        Args:
            model_path: Path to trained model weights
            scaler_path: Path to feature scaler
            device: "auto", "cuda", "cpu", or "mps"
            use_face_detection: Enable face detection integration
            use_texture_features: Enable texture features for robustness
        """
        # Set device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.use_face_detection = use_face_detection
        
        # Initialize feature extractor with texture support
        self.feature_extractor = BackgroundFeatureExtractor(
            use_texture_features=use_texture_features
        )
        
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
        
        # Load feature scaler
        self.scaler = None
        if scaler_path is None:
            scaler_path = Path(MODEL_DIR) / "scaler.pkl"
        if Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded from {scaler_path}")
        
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
        
        # Apply scaler if available
        if self.scaler is not None:
            signature = self.scaler.transform(signature.reshape(1, -1)).flatten()
        
        return signature
    
    def predict(self, image_path: str, 
               face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, any]:
        """
        Predict if image is real or AI-generated.
        
        Args:
            image_path: Path to image
            face_bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Extract features
        features = self.extract_features(image_path, face_bbox)
        
        # Convert to tensor and move to device
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
        
        # Get feature availability info
        feature_availability = self.feature_extractor.get_feature_availability()
        
        return {
            'prediction': label,
            'confidence': confidence,
            'real_probability': probs[0, 0].item(),
            'ai_generated_probability': probs[0, 1].item(),
            'features_used': feature_availability,
            'device': str(self.device)
        }
    
    def predict_batch(self, image_paths: list, 
                     face_bboxes: Optional[list] = None) -> list:
        """Predict for multiple images with GPU acceleration"""
        results = []
        
        # Extract all features first
        all_features = []
        valid_paths = []
        
        for i, image_path in enumerate(image_paths):
            try:
                face_bbox = face_bboxes[i] if face_bboxes else None
                features = self.extract_features(image_path, face_bbox)
                all_features.append(features)
                valid_paths.append(image_path)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'prediction': 'Error'
                })
        
        if all_features:
            # Batch inference on GPU
            features_tensor = torch.from_numpy(np.array(all_features)).float().to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
            
            for i, (path, pred, prob) in enumerate(zip(valid_paths, predictions, probs)):
                pred_idx = pred.item()
                results.append({
                    'image_path': path,
                    'prediction': "AI-Generated" if pred_idx == 1 else "Real Camera Image",
                    'confidence': prob[pred_idx].item(),
                    'real_probability': prob[0].item(),
                    'ai_generated_probability': prob[1].item()
                })
        
        return results
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Check if input dimensions match
            saved_input_dim = checkpoint.get('input_dim', self.input_dim)
            if saved_input_dim != self.input_dim:
                print(f"⚠ Warning: Model was trained with input_dim={saved_input_dim}, "
                      f"but current features have {self.input_dim} dimensions.")
                print("  Reinitializing model with saved dimensions...")
                self.model = BackgroundAuthenticityClassifier(
                    input_dim=saved_input_dim,
                    hidden_dim=256,
                    num_classes=2,
                    dropout_rate=0.3
                ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Model loaded from {model_path} (device: {self.device})")
    
    def save_model(self, model_path: str, epoch: int = 0, 
                   train_loss: float = 0.0, val_accuracy: float = 0.0):
        """Save trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'input_dim': self.input_dim,
            'device': str(self.device)
        }
        torch.save(checkpoint, model_path)
        print(f"✓ Model saved to {model_path}")
    
    def explain_prediction(self, image_path: str, 
                          face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Provide explanation for prediction by showing feature contributions.
        """
        # Extract all features
        features = self.feature_extractor.extract_all_features(image_path, face_bbox)
        
        # Get feature dimensions
        dims = self.feature_extractor.get_feature_dimensions()
        
        # Convert to tensor
        full_signature = self.feature_extractor.extract_unified_signature(image_path, face_bbox)
        
        # Apply scaler if available
        if self.scaler is not None:
            full_signature = self.scaler.transform(full_signature.reshape(1, -1)).flatten()
        
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
        metadata_end = noise_end + dims.get('metadata', 11)
        
        freq_importance = np.abs(gradients[:freq_end]).mean()
        noise_importance = np.abs(gradients[freq_end:noise_end]).mean()
        metadata_importance = np.abs(gradients[noise_end:metadata_end]).mean()
        
        # Texture importance (if available)
        texture_importance = 0.0
        if dims.get('texture', 0) > 0 and metadata_end < len(gradients):
            texture_importance = np.abs(gradients[metadata_end:]).mean()
        
        # Get feature availability
        availability = self.feature_extractor.get_feature_availability()
        
        explanation = {
            'prediction': "AI-Generated" if torch.argmax(probs, dim=1).item() == 1 else "Real",
            'confidence': probs[0, torch.argmax(probs, dim=1).item()].item(),
            'feature_contributions': {
                'frequency_features': float(freq_importance),
                'noise_features': float(noise_importance),
                'metadata_features': float(metadata_importance),
                'texture_features': float(texture_importance),
            },
            'feature_availability': availability,
            'feature_statistics': {
                'frequency_mean': float(features['frequency'].mean()),
                'noise_mean': float(features['noise'].mean()),
                'metadata_mean': float(features['metadata'].mean()),
            },
            'device': str(self.device)
        }
        
        if 'texture' in features:
            explanation['feature_statistics']['texture_mean'] = float(features['texture'].mean())
        
        return explanation
