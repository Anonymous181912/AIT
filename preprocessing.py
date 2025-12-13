"""
Image preprocessing module for deepfake detection
Handles image loading, normalization, and preparation for feature extraction
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocesses images for background feature extraction"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path, preserving original format"""
        # Load with OpenCV to preserve metadata and color channels
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def preprocess_for_features(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for background feature extraction
        Maintains original characteristics important for authenticity detection
        """
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use INTER_AREA for downsampling to preserve quality
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size if needed
        if new_h != self.target_size[0] or new_w != self.target_size[1]:
            pad_h = self.target_size[0] - new_h
            pad_w = self.target_size[1] - new_w
            resized = cv2.copyMakeBorder(
                resized, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        return resized
    
    def extract_background_region(self, image: np.ndarray, 
                                  face_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Extract background region by masking out face area
        If no face bbox provided, uses entire image
        """
        if face_bbox is None:
            return image
        
        x, y, w, h = face_bbox
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Create mask to exclude face region
        mask[y:y+h, x:x+w] = 0
        
        # Apply mask
        background = image.copy()
        background[mask == 0] = [0, 0, 0]  # Black out face region
        
        return background
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor"""
        # Convert HWC to CHW and normalize to [0, 1]
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return tensor
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

