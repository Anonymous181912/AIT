"""
Background feature extraction module
Combines all background-level features into a unified signature
Enhanced with texture features for metadata-independent detection
"""
import numpy as np
from typing import Dict, Optional
from frequency_analysis import FrequencyAnalyzer
from noise_extraction import NoiseExtractor
from metadata_inspector import MetadataInspector
from preprocessing import ImagePreprocessor

# Import texture features (new)
try:
    from texture_features import TextureFeatureExtractor
    TEXTURE_AVAILABLE = True
except ImportError:
    TEXTURE_AVAILABLE = False


class BackgroundFeatureExtractor:
    """
    Main class for extracting background-level authenticity features.
    Combines frequency analysis, noise patterns, metadata, and texture features.
    
    Enhanced to work without metadata (graceful degradation).
    """
    
    def __init__(self, 
                 fft_window_size: int = 32,
                 dct_block_size: int = 8,
                 noise_patch_size: int = 8,
                 use_texture_features: bool = True,
                 require_metadata: bool = False):
        """
        Initialize feature extractor.
        
        Args:
            fft_window_size: Window size for FFT analysis
            dct_block_size: Block size for DCT analysis
            noise_patch_size: Patch size for noise analysis
            use_texture_features: Enable texture features (recommended)
            require_metadata: If False, gracefully degrade when metadata missing
        """
        self.frequency_analyzer = FrequencyAnalyzer(
            fft_window_size=fft_window_size,
            dct_block_size=dct_block_size
        )
        self.noise_extractor = NoiseExtractor(patch_size=noise_patch_size)
        self.metadata_inspector = MetadataInspector()
        self.preprocessor = ImagePreprocessor()
        
        # Texture features (new)
        self.use_texture_features = use_texture_features and TEXTURE_AVAILABLE
        if self.use_texture_features:
            self.texture_extractor = TextureFeatureExtractor()
        else:
            self.texture_extractor = None
        
        self.require_metadata = require_metadata
        
        # Track feature availability for adaptive weighting
        self._feature_availability = {
            'frequency': True,
            'noise': True,
            'metadata': True,
            'texture': self.use_texture_features
        }
    
    def extract_all_features(self, image_path: str, 
                            face_bbox: Optional[tuple] = None) -> Dict[str, np.ndarray]:
        """
        Extract all background features from an image.
        
        Args:
            image_path: Path to image
            face_bbox: Optional face bounding box to mask out
            
        Returns:
            Dictionary of feature arrays by type
        """
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        processed_image = self.preprocessor.preprocess_for_features(image)
        
        # Extract background region (mask out face if provided)
        background = self.preprocessor.extract_background_region(processed_image, face_bbox)
        
        # Extract features
        features = {}
        availability = {}
        
        # Frequency domain features (always available)
        try:
            features['frequency'] = self.frequency_analyzer.extract_frequency_signature(background)
            availability['frequency'] = True
        except Exception as e:
            features['frequency'] = np.zeros(17, dtype=np.float32)
            availability['frequency'] = False
        
        # Noise pattern features (always available)
        try:
            features['noise'] = self.noise_extractor.extract_noise_signature(background)
            availability['noise'] = True
        except Exception as e:
            features['noise'] = np.zeros(14, dtype=np.float32)
            availability['noise'] = False
        
        # Metadata features (may be unavailable)
        try:
            features['metadata'] = self.metadata_inspector.extract_metadata_signature(image_path)
            availability['metadata'] = True
            
            # Check if metadata is actually present
            if np.sum(np.abs(features['metadata'])) < 0.1:
                availability['metadata'] = False
        except Exception as e:
            # Graceful degradation when metadata is missing
            features['metadata'] = np.zeros(11, dtype=np.float32)
            availability['metadata'] = False
        
        # Texture features (new - provides robustness without metadata)
        if self.use_texture_features and self.texture_extractor:
            try:
                features['texture'] = self.texture_extractor.extract_texture_signature(background)
                availability['texture'] = True
            except Exception as e:
                features['texture'] = np.zeros(37, dtype=np.float32)
                availability['texture'] = False
        
        # Store availability
        self._feature_availability = availability
        features['_availability'] = availability
        
        return features
    
    def extract_unified_signature(self, image_path: str, 
                                 face_bbox: Optional[tuple] = None) -> np.ndarray:
        """
        Extract unified signature combining all background features.
        Returns a single feature vector for classification.
        
        Args:
            image_path: Path to image
            face_bbox: Optional face bounding box
            
        Returns:
            Unified feature vector
        """
        features = self.extract_all_features(image_path, face_bbox)
        
        # Build signature based on available features
        signature_parts = [
            features['frequency'],  # 17 features
            features['noise'],      # 14 features
            features['metadata'],   # 11 features
        ]
        
        # Add texture features if enabled
        if self.use_texture_features and 'texture' in features:
            signature_parts.append(features['texture'])  # 37 features
        
        # Concatenate all feature vectors
        signature = np.concatenate(signature_parts)
        
        # Handle any NaN or Inf values
        signature = np.nan_to_num(signature, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return signature
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature type"""
        # Create dummy features to get dimensions
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        freq_sig = self.frequency_analyzer.extract_frequency_signature(dummy_image)
        noise_sig = self.noise_extractor.extract_noise_signature(dummy_image)
        metadata_sig = np.zeros(11, dtype=np.float32)  # Known metadata dimension
        
        dims = {
            'frequency': len(freq_sig),
            'noise': len(noise_sig),
            'metadata': len(metadata_sig),
        }
        
        if self.use_texture_features and self.texture_extractor:
            texture_dims = self.texture_extractor.get_feature_dimensions()
            dims['texture'] = texture_dims['total']
        else:
            dims['texture'] = 0
        
        dims['total'] = sum(dims.values())
        
        return dims
    
    def get_feature_availability(self) -> Dict[str, bool]:
        """Get availability of each feature type from last extraction"""
        return self._feature_availability.copy()
    
    def is_metadata_available(self) -> bool:
        """Check if metadata was available in last extraction"""
        return self._feature_availability.get('metadata', False)
