"""
Advanced Face Detection Module
Supports multiple backends: MTCNN, MediaPipe, and OpenCV Haar Cascade
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import warnings


class AdvancedFaceDetector:
    """
    Advanced face detector with multiple backend support.
    Falls back gracefully if a backend is unavailable.
    
    Backends:
    - MTCNN: Most accurate, best for deepfake detection
    - MediaPipe: Fast and production-ready
    - OpenCV Haar Cascade: Lightweight fallback
    """
    
    def __init__(self, backend: str = "mtcnn", min_confidence: float = 0.9):
        """
        Initialize face detector.
        
        Args:
            backend: "mtcnn", "mediapipe", or "opencv"
            min_confidence: Minimum detection confidence
        """
        self.backend = backend
        self.min_confidence = min_confidence
        self.detector = None
        
        # Try to initialize the requested backend
        self._initialize_backend(backend)
    
    def _initialize_backend(self, backend: str):
        """Initialize the specified detection backend"""
        if backend == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.detector = MTCNN(
                    keep_all=True,
                    device=device,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7]
                )
                self.backend = "mtcnn"
                print(f"✓ Face detector initialized: MTCNN on {device}")
            except ImportError as e:
                warnings.warn(f"MTCNN not available ({e}). Trying MediaPipe...")
                self._initialize_backend("mediapipe")
            except Exception as e:
                warnings.warn(f"MTCNN initialization failed ({e}). Trying MediaPipe...")
                self._initialize_backend("mediapipe")
        
        elif backend == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_face = mp.solutions.face_detection
                self.detector = self.mp_face.FaceDetection(
                    model_selection=1,  # Full-range model
                    min_detection_confidence=self.min_confidence
                )
                self.backend = "mediapipe"
                print("✓ Face detector initialized: MediaPipe")
            except ImportError as e:
                warnings.warn(f"MediaPipe not available ({e}). Using OpenCV...")
                self._initialize_backend("opencv")
            except Exception as e:
                warnings.warn(f"MediaPipe initialization failed ({e}). Using OpenCV...")
                self._initialize_backend("opencv")
        
        elif backend == "opencv":
            try:
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.backend = "opencv"
                print("✓ Face detector initialized: OpenCV Haar Cascade")
            except Exception as e:
                warnings.warn(f"OpenCV face detector failed: {e}")
                self.detector = None
                self.backend = "none"
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of face detections, each with:
            - bbox: (x, y, w, h)
            - confidence: detection confidence
            - landmarks: facial landmarks (if available)
        """
        if self.detector is None:
            return []
        
        if self.backend == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.backend == "mediapipe":
            return self._detect_mediapipe(image)
        elif self.backend == "opencv":
            return self._detect_opencv(image)
        else:
            return []
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN"""
        import torch
        
        # MTCNN expects RGB image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
        
        results = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= self.min_confidence:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_landmarks = None
                    if landmarks is not None and i < len(landmarks):
                        face_landmarks = landmarks[i].tolist() if landmarks[i] is not None else None
                    
                    results.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h)
                        'confidence': float(prob),
                        'landmarks': face_landmarks
                    })
        
        return results
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        # MediaPipe expects RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = image.shape[:2]
        
        results_mp = self.detector.process(image)
        
        results = []
        if results_mp.detections:
            for detection in results_mp.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounds are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extract landmarks
                landmarks = []
                for kp in detection.location_data.relative_keypoints:
                    landmarks.append([kp.x * w, kp.y * h])
                
                results.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0] if detection.score else 0.0,
                    'landmarks': landmarks if landmarks else None
                })
        
        return results
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar cascade doesn't provide confidence
                'landmarks': None
            })
        
        return results
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Dict]:
        """Get the largest detected face"""
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Return face with largest area
        return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
    
    def extract_face_region(self, image: np.ndarray, 
                           face: Dict, 
                           margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face region from image with margin.
        
        Args:
            image: Source image
            face: Face detection result
            margin: Margin around face as fraction of face size
            
        Returns:
            Cropped face region
        """
        x, y, w, h = face['bbox']
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def align_face(self, image: np.ndarray, face: Dict, 
                   output_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Align face using landmarks (if available).
        
        Args:
            image: Source image
            face: Face detection result with landmarks
            output_size: Output face size
            
        Returns:
            Aligned face image
        """
        landmarks = face.get('landmarks')
        
        if landmarks is None or len(landmarks) < 2:
            # Just crop without alignment
            face_img = self.extract_face_region(image, face)
            if face_img is not None:
                return cv2.resize(face_img, output_size)
            return None
        
        # Use eye landmarks for alignment (if available)
        # MTCNN landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        try:
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            
            # Calculate angle
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Calculate center between eyes
            eye_center = ((left_eye[0] + right_eye[0]) / 2, 
                         (left_eye[1] + right_eye[1]) / 2)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            
            # Rotate image
            h, w = image.shape[:2]
            rotated = cv2.warpAffine(image, M, (w, h))
            
            # Extract face from rotated image
            face_img = self.extract_face_region(rotated, face)
            if face_img is not None:
                return cv2.resize(face_img, output_size)
        except Exception:
            pass
        
        # Fallback: simple crop
        face_img = self.extract_face_region(image, face)
        if face_img is not None:
            return cv2.resize(face_img, output_size)
        return None
