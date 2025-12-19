"""
Frequency domain analysis module
Extracts features from FFT and DCT transforms to detect synthetic generation patterns
"""
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.fftpack import dct
from typing import Tuple, Dict


class FrequencyAnalyzer:
    """Analyzes frequency domain characteristics of images"""
    
    def __init__(self, fft_window_size: int = 32, dct_block_size: int = 8):
        self.fft_window_size = fft_window_size
        self.dct_block_size = dct_block_size
    
    def compute_fft_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute FFT-based features
        Real camera images have natural frequency distributions,
        while AI-generated images often show artificial patterns
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute 2D FFT
        f_transform = fft2(gray.astype(np.float32))
        f_shifted = fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        phase = np.angle(f_shifted)
        
        # Extract frequency domain features
        features = {
            'magnitude_spectrum': magnitude,
            'phase_spectrum': phase,
            'log_magnitude': np.log1p(magnitude),  # Log scale for better visualization
        }
        
        # Compute radial frequency distribution
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Radial frequency profile
        radial_profile = []
        max_radius = min(center_x, center_y)
        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile.append(np.mean(magnitude[mask]))
        
        features['radial_profile'] = np.array(radial_profile)
        
        # High-frequency energy (indicator of compression/artifacts)
        high_freq_mask = r > max_radius * 0.7
        features['high_freq_energy'] = np.sum(magnitude[high_freq_mask])
        
        # Low-frequency energy
        low_freq_mask = r < max_radius * 0.3
        features['low_freq_energy'] = np.sum(magnitude[low_freq_mask])
        
        # Frequency distribution statistics
        features['freq_mean'] = np.mean(magnitude)
        features['freq_std'] = np.std(magnitude)
        features['freq_skew'] = self._compute_skewness(magnitude)
        features['freq_kurtosis'] = self._compute_kurtosis(magnitude)
        
        return features
    
    def compute_dct_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute DCT-based features
        DCT reveals compression artifacts and block-level patterns
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        features = {}
        
        # Compute DCT for entire image
        dct_full = dct(dct(gray.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
        
        features['dct_full'] = dct_full
        features['dct_energy'] = np.sum(np.abs(dct_full))
        
        # Block-based DCT (JPEG-like analysis)
        block_features = []
        for i in range(0, h - self.dct_block_size + 1, self.dct_block_size):
            for j in range(0, w - self.dct_block_size + 1, self.dct_block_size):
                block = gray[i:i+self.dct_block_size, j:j+self.dct_block_size]
                dct_block = dct(dct(block.astype(np.float32), axis=0, norm='ortho'), 
                               axis=1, norm='ortho')
                
                # Extract block-level statistics
                block_features.append({
                    'dc_coefficient': dct_block[0, 0],  # DC component
                    'ac_energy': np.sum(np.abs(dct_block[1:, 1:])),  # AC components
                    'block_variance': np.var(dct_block),
                })
        
        features['block_statistics'] = block_features
        
        # DCT coefficient distribution
        features['dct_mean'] = np.mean(np.abs(dct_full))
        features['dct_std'] = np.std(dct_full)
        
        # High-frequency DCT coefficients (compression artifacts)
        hf_dct = dct_full[h//4:, w//4:]
        features['high_freq_dct_energy'] = np.sum(np.abs(hf_dct))
        
        return features
    
    def extract_frequency_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract compact frequency signature for classification
        Combines key FFT and DCT features into a feature vector
        """
        fft_features = self.compute_fft_features(image)
        dct_features = self.compute_dct_features(image)
        
        signature = []
        
        # FFT-based features
        signature.append(fft_features['freq_mean'])
        signature.append(fft_features['freq_std'])
        signature.append(fft_features['freq_skew'])
        signature.append(fft_features['freq_kurtosis'])
        signature.append(fft_features['high_freq_energy'])
        signature.append(fft_features['low_freq_energy'])
        
        # Radial profile statistics
        if len(fft_features['radial_profile']) > 0:
            signature.append(np.mean(fft_features['radial_profile']))
            signature.append(np.std(fft_features['radial_profile']))
        else:
            signature.extend([0, 0])
        
        # DCT-based features
        signature.append(dct_features['dct_mean'])
        signature.append(dct_features['dct_std'])
        signature.append(dct_features['dct_energy'])
        signature.append(dct_features['high_freq_dct_energy'])
        
        # Block statistics
        if dct_features['block_statistics']:
            dc_coeffs = [b['dc_coefficient'] for b in dct_features['block_statistics']]
            ac_energies = [b['ac_energy'] for b in dct_features['block_statistics']]
            block_vars = [b['block_variance'] for b in dct_features['block_statistics']]
            
            signature.append(np.mean(dc_coeffs))
            signature.append(np.std(dc_coeffs))
            signature.append(np.mean(ac_energies))
            signature.append(np.std(ac_energies))
            signature.append(np.mean(block_vars))
        else:
            signature.extend([0, 0, 0, 0, 0])
        
        return np.array(signature, dtype=np.float32)
    
    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = data.size
        skew = (1/n) * np.sum(((data - mean) / std) ** 3)
        return float(skew)
    
    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = data.size
        kurt = (1/n) * np.sum(((data - mean) / std) ** 4) - 3.0
        return float(kurt)

