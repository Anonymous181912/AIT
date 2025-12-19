"""
Texture Feature Extraction Module
Robust texture-based features that work without metadata
For metadata-independent deepfake detection
"""
import numpy as np
import cv2
from typing import Dict, Tuple
from scipy import ndimage
from scipy.stats import entropy


class TextureFeatureExtractor:
    """
    Extracts texture-based features that don't rely on metadata.
    Provides robust detection when EXIF data is unavailable.
    """
    
    def __init__(self, 
                 lbp_radius: int = 3,
                 lbp_points: int = 24,
                 glcm_distances: list = None,
                 gabor_frequencies: list = None):
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.glcm_distances = glcm_distances or [1, 2, 4]
        self.gabor_frequencies = gabor_frequencies or [0.1, 0.2, 0.3, 0.4]
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) features.
        LBP captures local texture patterns that differ between real and AI-generated images.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute LBP
        lbp = self._compute_lbp(gray, self.lbp_radius, self.lbp_points)
        
        # Compute histogram of LBP
        n_bins = self.lbp_points + 2  # Uniform LBP patterns
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Extract statistical features from LBP
        features = [
            np.mean(lbp),           # Mean LBP value
            np.std(lbp),            # Standard deviation
            entropy(hist + 1e-10),  # Entropy of LBP histogram
            np.max(hist),           # Max histogram bin (uniformity)
            np.sum(hist[:self.lbp_points]),  # Uniform pattern ratio
        ]
        
        # Add first few histogram bins as features
        features.extend(hist[:8].tolist())
        
        return np.array(features, dtype=np.float32)
    
    def _compute_lbp(self, gray: np.ndarray, radius: int, points: int) -> np.ndarray:
        """Compute Local Binary Pattern"""
        rows, cols = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray[i, j]
                code = 0
                for p in range(points):
                    angle = 2 * np.pi * p / points
                    x = int(round(j + radius * np.cos(angle)))
                    y = int(round(i - radius * np.sin(angle)))
                    if gray[y, x] >= center:
                        code |= (1 << p)
                lbp[i, j] = code
        
        return lbp
    
    def extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) features.
        GLCM captures texture homogeneity, contrast, and correlation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Quantize to reduce GLCM size
        levels = 32
        gray_quantized = (gray / 256 * levels).astype(np.uint8)
        
        features = []
        
        # Compute GLCM for different distances and angles
        for distance in self.glcm_distances:
            # Compute GLCM for 0 degrees (horizontal)
            glcm = self._compute_glcm(gray_quantized, distance, 0, levels)
            
            # Extract Haralick features
            contrast = self._glcm_contrast(glcm)
            homogeneity = self._glcm_homogeneity(glcm)
            energy = self._glcm_energy(glcm)
            correlation = self._glcm_correlation(glcm)
            
            features.extend([contrast, homogeneity, energy, correlation])
        
        return np.array(features, dtype=np.float32)
    
    def _compute_glcm(self, gray: np.ndarray, distance: int, angle: int, levels: int) -> np.ndarray:
        """Compute Gray-Level Co-occurrence Matrix"""
        glcm = np.zeros((levels, levels), dtype=np.float32)
        
        rows, cols = gray.shape
        
        if angle == 0:  # Horizontal
            dx, dy = distance, 0
        elif angle == 45:
            dx, dy = distance, -distance
        elif angle == 90:  # Vertical
            dx, dy = 0, distance
        else:  # 135 degrees
            dx, dy = -distance, -distance
        
        for i in range(rows):
            for j in range(cols):
                ni, nj = i + dy, j + dx
                if 0 <= ni < rows and 0 <= nj < cols:
                    glcm[gray[i, j], gray[ni, nj]] += 1
        
        # Normalize
        glcm = glcm / (glcm.sum() + 1e-10)
        
        return glcm
    
    def _glcm_contrast(self, glcm: np.ndarray) -> float:
        """Compute GLCM contrast"""
        levels = glcm.shape[0]
        i, j = np.ogrid[:levels, :levels]
        return float(np.sum(glcm * (i - j) ** 2))
    
    def _glcm_homogeneity(self, glcm: np.ndarray) -> float:
        """Compute GLCM homogeneity (inverse difference moment)"""
        levels = glcm.shape[0]
        i, j = np.ogrid[:levels, :levels]
        return float(np.sum(glcm / (1 + np.abs(i - j))))
    
    def _glcm_energy(self, glcm: np.ndarray) -> float:
        """Compute GLCM energy (angular second moment)"""
        return float(np.sum(glcm ** 2))
    
    def _glcm_correlation(self, glcm: np.ndarray) -> float:
        """Compute GLCM correlation"""
        levels = glcm.shape[0]
        i, j = np.ogrid[:levels, :levels]
        
        mu_i = np.sum(i * np.sum(glcm, axis=1, keepdims=True))
        mu_j = np.sum(j * np.sum(glcm, axis=0, keepdims=True))
        
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * np.sum(glcm, axis=1, keepdims=True)))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * np.sum(glcm, axis=0, keepdims=True)))
        
        if sigma_i == 0 or sigma_j == 0:
            return 0.0
        
        return float(np.sum(glcm * (i - mu_i) * (j - mu_j)) / (sigma_i * sigma_j + 1e-10))
    
    def extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gabor filter features.
        Gabor filters capture multi-scale texture information.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32) / 255.0
        
        features = []
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for frequency in self.gabor_frequencies:
            for theta in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(31, 31),
                    sigma=4.0,
                    theta=theta,
                    lambd=1.0/frequency,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                # Apply Gabor filter
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                
                # Extract statistics
                features.append(np.mean(np.abs(filtered)))
                features.append(np.std(filtered))
        
        return np.array(features, dtype=np.float32)
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edge-based texture features.
        Edge patterns differ between real and AI-generated images.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Edge histogram (8 bins for direction)
        dir_bins = 8
        dir_hist, _ = np.histogram(direction.ravel(), bins=dir_bins, range=(-np.pi, np.pi), density=True)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        features = [
            np.mean(magnitude),      # Mean gradient magnitude
            np.std(magnitude),       # Gradient variance
            edge_density,            # Edge density
            entropy(dir_hist + 1e-10),  # Direction entropy
        ]
        
        # Add direction histogram
        features.extend(dir_hist.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def extract_texture_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract complete texture signature combining all methods.
        Returns ~32 dimensional feature vector.
        """
        try:
            lbp_features = self.extract_lbp_features(image)
        except Exception:
            lbp_features = np.zeros(13, dtype=np.float32)
        
        try:
            glcm_features = self.extract_glcm_features(image)
        except Exception:
            glcm_features = np.zeros(12, dtype=np.float32)
        
        try:
            edge_features = self.extract_edge_features(image)
        except Exception:
            edge_features = np.zeros(12, dtype=np.float32)
        
        # Note: Gabor features are computationally expensive, 
        # so we use a subset for efficiency
        # gabor_features = self.extract_gabor_features(image)
        
        signature = np.concatenate([
            lbp_features,    # 13 features
            glcm_features,   # 12 features
            edge_features,   # 12 features
        ])
        
        return signature
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each texture feature type"""
        return {
            'lbp': 13,
            'glcm': 12,
            'edge': 12,
            'total': 37
        }
