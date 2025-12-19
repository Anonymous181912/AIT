"""
Face detection and integration module
Combines background authenticity analysis with face-based detection
Enhanced with real deep learning models
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import warnings

from face_detector import AdvancedFaceDetector
from face_classifier import FaceClassifierInference


class HybridDeepfakeDetector:
    """
    Combines background authenticity analysis with face-based detection.
    Uses weighted ensemble of both approaches for robust detection.
    """
    
    def __init__(self, 
                 background_pipeline,
                 face_model_path: Optional[str] = None,
                 face_detector_backend: str = "mtcnn",
                 background_weight: float = 0.6,
                 face_weight: float = 0.4,
                 device: str = "auto"):
        """
        Initialize hybrid detector.
        
        Args:
            background_pipeline: Trained background detection pipeline
            face_model_path: Path to trained face classifier (optional)
            face_detector_backend: "mtcnn", "mediapipe", or "opencv"
            background_weight: Weight for background analysis (0-1)
            face_weight: Weight for face analysis (0-1)
            device: "auto", "cuda", "cpu", or "mps"
        """
        self.background_pipeline = background_pipeline
        self.background_weight = background_weight
        self.face_weight = face_weight
        
        # Initialize face detector
        try:
            self.face_detector = AdvancedFaceDetector(
                backend=face_detector_backend,
                min_confidence=0.9
            )
        except Exception as e:
            warnings.warn(f"Face detector initialization failed: {e}")
            self.face_detector = None
        
        # Initialize face classifier
        self.face_classifier = None
        if face_model_path or True:  # Always try to initialize
            try:
                self.face_classifier = FaceClassifierInference(
                    model_path=face_model_path,
                    device=device
                )
            except Exception as e:
                warnings.warn(f"Face classifier initialization failed: {e}")
    
    def detect(self, image_path: str) -> Dict:
        """
        Hybrid detection combining background and face analysis.
        
        Args:
            image_path: Path to image
            
        Returns:
            Detection result with combined prediction
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_result = None
        face_detection = None
        face_bbox = None
        
        if self.face_detector is not None:
            face_detection = self.face_detector.get_largest_face(image_rgb)
            if face_detection:
                face_bbox = face_detection['bbox']
        
        # Background analysis (pass face bbox if detected)
        bg_result = self.background_pipeline.predict(image_path, face_bbox)
        
        # Face analysis (if face detected and classifier available)
        if face_detection and self.face_classifier:
            try:
                # Extract and align face
                face_image = self.face_detector.align_face(
                    image_rgb, 
                    face_detection,
                    output_size=(224, 224)
                )
                
                if face_image is not None:
                    # Run face classifier
                    face_result = self.face_classifier.predict(face_image)
                    face_result['face_detected'] = True
                    face_result['detection_confidence'] = face_detection['confidence']
                    face_result['has_landmarks'] = face_detection['landmarks'] is not None
            except Exception as e:
                warnings.warn(f"Face classification failed: {e}")
                face_result = {
                    'face_detected': True,
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'error': str(e)
                }
        elif face_detection:
            face_result = {
                'face_detected': True,
                'prediction': 'Unknown (no classifier)',
                'confidence': 0.0,
            }
        else:
            face_result = {'face_detected': False}
        
        # Combine predictions
        bg_score = bg_result['ai_generated_probability']
        
        if face_result and face_result.get('face_detected') and 'ai_generated_probability' in face_result:
            # Weighted combination
            face_score = face_result['ai_generated_probability']
            
            # Dynamic weight adjustment based on confidence
            bg_conf = bg_result['confidence']
            face_conf = face_result.get('confidence', 0.5)
            
            # Normalize weights by confidence
            total_conf = bg_conf + face_conf
            if total_conf > 0:
                effective_bg_weight = self.background_weight * bg_conf / total_conf
                effective_face_weight = self.face_weight * face_conf / total_conf
                # Renormalize
                weight_sum = effective_bg_weight + effective_face_weight
                if weight_sum > 0:
                    effective_bg_weight /= weight_sum
                    effective_face_weight /= weight_sum
                else:
                    effective_bg_weight = self.background_weight
                    effective_face_weight = self.face_weight
            else:
                effective_bg_weight = self.background_weight
                effective_face_weight = self.face_weight
            
            combined_score = effective_bg_weight * bg_score + effective_face_weight * face_score
            final_prediction = "AI-Generated" if combined_score > 0.5 else "Real Camera Image"
            
            combination_method = "weighted_ensemble"
        else:
            # Only background analysis available
            combined_score = bg_score
            final_prediction = bg_result['prediction']
            effective_bg_weight = 1.0
            effective_face_weight = 0.0
            combination_method = "background_only"
        
        return {
            'prediction': final_prediction,
            'confidence': abs(combined_score - 0.5) * 2,  # Normalize to [0, 1]
            'combined_score': combined_score,
            'combination_method': combination_method,
            'weights_used': {
                'background': effective_bg_weight if 'effective_bg_weight' in dir() else 1.0,
                'face': effective_face_weight if 'effective_face_weight' in dir() else 0.0
            },
            'background_analysis': bg_result,
            'face_analysis': face_result
        }
    
    def detect_batch(self, image_paths: List[str]) -> List[Dict]:
        """Detect on multiple images"""
        results = []
        for path in image_paths:
            try:
                result = self.detect(path)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': path,
                    'error': str(e),
                    'prediction': 'Error'
                })
        return results
    
    def explain(self, image_path: str) -> Dict:
        """
        Provide detailed explanation of detection.
        
        Args:
            image_path: Path to image
            
        Returns:
            Detailed explanation with feature contributions
        """
        # Get basic detection
        result = self.detect(image_path)
        
        # Add background explanation
        try:
            bg_explanation = self.background_pipeline.explain_prediction(image_path)
            result['background_explanation'] = bg_explanation
        except Exception as e:
            result['background_explanation'] = {'error': str(e)}
        
        # Add summary
        result['explanation_summary'] = self._generate_summary(result)
        
        return result
    
    def _generate_summary(self, result: Dict) -> str:
        """Generate human-readable explanation summary"""
        lines = []
        
        # Overall prediction
        lines.append(f"Prediction: {result['prediction']}")
        lines.append(f"Confidence: {result['confidence']:.2%}")
        lines.append(f"Method: {result['combination_method']}")
        lines.append("")
        
        # Background analysis
        bg = result.get('background_analysis', {})
        lines.append("Background Analysis:")
        lines.append(f"  - Real probability: {bg.get('real_probability', 0):.2%}")
        lines.append(f"  - AI probability: {bg.get('ai_generated_probability', 0):.2%}")
        lines.append("")
        
        # Face analysis
        face = result.get('face_analysis', {})
        if face.get('face_detected'):
            lines.append("Face Analysis:")
            lines.append(f"  - Face detected: Yes")
            lines.append(f"  - Prediction: {face.get('prediction', 'N/A')}")
            if 'ai_generated_probability' in face:
                lines.append(f"  - AI probability: {face['ai_generated_probability']:.2%}")
        else:
            lines.append("Face Analysis:")
            lines.append("  - Face detected: No")
            lines.append("  - Using background analysis only")
        
        return "\n".join(lines)


# Legacy compatibility
class FaceDetector:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        self._detector = AdvancedFaceDetector(backend="opencv")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes"""
        results = self._detector.detect_faces(image)
        return [r['bbox'] for r in results]
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get largest face bounding box"""
        result = self._detector.get_largest_face(image)
        return result['bbox'] if result else None


class FaceFeatureExtractor:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        try:
            self._classifier = FaceClassifierInference()
        except:
            self._classifier = None
    
    def extract_face_features(self, image: np.ndarray, 
                             face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract features from face region"""
        if self._classifier is None:
            # Fallback to simple features
            return self._extract_simple_features(image, face_bbox)
        
        x, y, w, h = face_bbox
        face_region = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (224, 224))
        
        return self._classifier.extract_features(face_resized)
    
    def _extract_simple_features(self, image: np.ndarray, 
                                  face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Simple features as fallback"""
        x, y, w, h = face_bbox
        face_region = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (224, 224))
        
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY) if len(face_resized.shape) == 3 else face_resized
        
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        features.extend(hist.flatten()[:32])
        
        # Gradient features
        grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(grad_mag))
        features.append(np.std(grad_mag))
        
        return np.array(features, dtype=np.float32)
