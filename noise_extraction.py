"""
Noise residual extraction module
Extracts camera sensor noise patterns and compression artifacts
Real cameras leave unique noise signatures that AI-generated images lack
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import wiener
from typing import Dict, Tuple


class NoiseExtractor:
    """Extracts noise patterns and residuals from images"""
    
    def __init__(self, patch_size: int = 8):
        self.patch_size = patch_size
    
    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise residual using high-pass filtering
        Real camera images contain sensor noise that AI-generated images lack
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Convert to float for processing
        gray_float = gray.astype(np.float32)
        
        # Apply denoising to estimate clean image
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        denoised_float = denoised.astype(np.float32)
        
        # Noise residual = original - denoised
        noise_residual = gray_float - denoised_float
        
        return noise_residual
    
    def compute_noise_statistics(self, noise_residual: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of noise residual"""
        stats = {
            'mean': float(np.mean(noise_residual)),
            'std': float(np.std(noise_residual)),
            'variance': float(np.var(noise_residual)),
            'skewness': self._compute_skewness(noise_residual),
            'kurtosis': self._compute_kurtosis(noise_residual),
            'energy': float(np.sum(noise_residual ** 2)),
        }
        
        # Spatial correlation (real noise has spatial structure)
        h, w = noise_residual.shape
        if h > 1 and w > 1:
            # Horizontal correlation
            h_corr = np.corrcoef(noise_residual[:-1, :].flatten(), 
                                noise_residual[1:, :].flatten())[0, 1]
            # Vertical correlation
            v_corr = np.corrcoef(noise_residual[:, :-1].flatten(), 
                                noise_residual[:, 1:].flatten())[0, 1]
            
            stats['horizontal_correlation'] = float(h_corr) if not np.isnan(h_corr) else 0.0
            stats['vertical_correlation'] = float(v_corr) if not np.isnan(v_corr) else 0.0
        else:
            stats['horizontal_correlation'] = 0.0
            stats['vertical_correlation'] = 0.0
        
        return stats
    
    def extract_sensor_noise_pattern(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract camera sensor noise pattern
        Real cameras have consistent noise patterns across images
        """
        noise_residual = self.extract_noise_residual(image)
        
        # Compute noise pattern in frequency domain
        fft_noise = np.fft.fft2(noise_residual)
        fft_shifted = np.fft.fftshift(fft_noise)
        magnitude = np.abs(fft_shifted)
        
        # Extract pattern characteristics
        pattern = {
            'noise_residual': noise_residual,
            'noise_fft_magnitude': magnitude,
            'noise_energy': np.sum(magnitude ** 2),
        }
        
        # Local noise variance (patch-based)
        h, w = noise_residual.shape
        patch_variances = []
        for i in range(0, h - self.patch_size + 1, self.patch_size):
            for j in range(0, w - self.patch_size + 1, self.patch_size):
                patch = noise_residual[i:i+self.patch_size, j:j+self.patch_size]
                patch_variances.append(np.var(patch))
        
        pattern['patch_variances'] = np.array(patch_variances)
        pattern['variance_mean'] = np.mean(patch_variances) if patch_variances else 0.0
        pattern['variance_std'] = np.std(patch_variances) if patch_variances else 0.0
        
        return pattern
    
    def detect_compression_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect JPEG/compression artifacts
        Real images often have compression artifacts, AI-generated may lack them
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Convert to float
        gray_float = gray.astype(np.float32)
        
        # Apply DCT to detect block artifacts (JPEG-like)
        from scipy.fftpack import dct
        
        h, w = gray_float.shape
        block_size = 8
        
        # Compute block boundaries
        block_artifacts = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray_float[i:i+block_size, j:j+block_size]
                
                # Compute DCT
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                
                # Block boundary discontinuity (indicator of compression)
                if i + block_size < h:
                    boundary_diff = np.mean(np.abs(
                        gray_float[i+block_size-1, j:j+block_size] - 
                        gray_float[i+block_size, j:j+block_size]
                    ))
                    block_artifacts.append(boundary_diff)
        
        artifacts = {
            'mean_block_discontinuity': float(np.mean(block_artifacts)) if block_artifacts else 0.0,
            'std_block_discontinuity': float(np.std(block_artifacts)) if block_artifacts else 0.0,
        }
        
        # High-frequency content (compression removes high frequencies)
        from scipy.fft import fft2, fftshift
        f_transform = fft2(gray_float)
        f_shifted = fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        
        # High-frequency energy
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = min(center_x, center_y)
        high_freq_mask = r > max_radius * 0.8
        
        artifacts['high_freq_energy'] = float(np.sum(magnitude[high_freq_mask]))
        
        return artifacts
    
    def extract_noise_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract compact noise signature for classification
        Combines noise statistics and pattern features
        """
        noise_residual = self.extract_noise_residual(image)
        noise_stats = self.compute_noise_statistics(noise_residual)
        noise_pattern = self.extract_sensor_noise_pattern(image)
        compression_artifacts = self.detect_compression_artifacts(image)
        
        signature = []
        
        # Noise statistics
        signature.append(noise_stats['mean'])
        signature.append(noise_stats['std'])
        signature.append(noise_stats['variance'])
        signature.append(noise_stats['skewness'])
        signature.append(noise_stats['kurtosis'])
        signature.append(noise_stats['energy'])
        signature.append(noise_stats['horizontal_correlation'])
        signature.append(noise_stats['vertical_correlation'])
        
        # Noise pattern features
        signature.append(noise_pattern['noise_energy'])
        signature.append(noise_pattern['variance_mean'])
        signature.append(noise_pattern['variance_std'])
        
        # Compression artifacts
        signature.append(compression_artifacts['mean_block_discontinuity'])
        signature.append(compression_artifacts['std_block_discontinuity'])
        signature.append(compression_artifacts['high_freq_energy'])
        
        return np.array(signature, dtype=np.float32)
    
    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = data.size
        skew = (1/n) * np.sum(((data - mean) / std) ** 3)
        return float(skew)
    
    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = data.size
        kurt = (1/n) * np.sum(((data - mean) / std) ** 4) - 3.0
        return float(kurt)

