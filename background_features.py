"""
Background feature extraction module
Combines all background-level features into a unified signature
"""
import numpy as np
from typing import Dict, Optional
from frequency_analysis import FrequencyAnalyzer
from noise_extraction import NoiseExtractor
from metadata_inspector import MetadataInspector
from preprocessing import ImagePreprocessor


class BackgroundFeatureExtractor:
    """
    Main class for extracting background-level authenticity features
    Combines frequency analysis, noise patterns, and metadata
    """
    
    def __init__(self, 
                 fft_window_size: int = 32,
                 dct_block_size: int = 8,
                 noise_patch_size: int = 8):
        self.frequency_analyzer = FrequencyAnalyzer(
            fft_window_size=fft_window_size,
            dct_block_size=dct_block_size
        )
        self.noise_extractor = NoiseExtractor(patch_size=noise_patch_size)
        self.metadata_inspector = MetadataInspector()
        self.preprocessor = ImagePreprocessor()
    
    def extract_all_features(self, image_path: str, 
                            face_bbox: Optional[tuple] = None) -> Dict[str, np.ndarray]:
        """
        Extract all background features from an image
        """
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        processed_image = self.preprocessor.preprocess_for_features(image)
        
        # Extract background region (mask out face if provided)
        background = self.preprocessor.extract_background_region(processed_image, face_bbox)
        
        # Extract features
        features = {}
        
        # Frequency domain features
        features['frequency'] = self.frequency_analyzer.extract_frequency_signature(background)
        
        # Noise pattern features
        features['noise'] = self.noise_extractor.extract_noise_signature(background)
        
        # Metadata features
        try:
            features['metadata'] = self.metadata_inspector.extract_metadata_signature(image_path)
        except Exception as e:
            # If metadata extraction fails, use zero vector
            features['metadata'] = np.zeros(11, dtype=np.float32)
        
        return features
    
    def extract_unified_signature(self, image_path: str, 
                                 face_bbox: Optional[tuple] = None) -> np.ndarray:
        """
        Extract unified signature combining all background features
        Returns a single feature vector for classification
        """
        features = self.extract_all_features(image_path, face_bbox)
        
        # Concatenate all feature vectors
        signature = np.concatenate([
            features['frequency'],
            features['noise'],
            features['metadata']
        ])
        
        return signature
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature type"""
        # Create dummy features to get dimensions
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        freq_sig = self.frequency_analyzer.extract_frequency_signature(dummy_image)
        noise_sig = self.noise_extractor.extract_noise_signature(dummy_image)
        metadata_sig = np.zeros(11, dtype=np.float32)  # Known metadata dimension
        
        return {
            'frequency': len(freq_sig),
            'noise': len(noise_sig),
            'metadata': len(metadata_sig),
            'total': len(freq_sig) + len(noise_sig) + len(metadata_sig)
        }

