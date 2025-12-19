import torch
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from advanced_classifier import AdvancedDeepfakeClassifier
except ImportError:
    try:
        from models.advanced_classifier import AdvancedDeepfakeClassifier
    except ImportError:
        print("CRITICAL ERROR: Could not find 'advanced_classifier.py'.")
        sys.exit(1)

from background_features import BackgroundFeatureExtractor
from config import get_device, MODEL_DIR

class DeepfakeDetectionPipeline:
    def __init__(self, device: str = "auto"):
        self.device = get_device() if device == "auto" else torch.device(device)
        self.feature_extractor = BackgroundFeatureExtractor()
        
        # Get dimensions
        feature_dims = self.feature_extractor.get_feature_dimensions()
        self.input_dim = feature_dims['total']
        
        # Init Model
        self.model = AdvancedDeepfakeClassifier(input_dim=self.input_dim, num_classes=2, dropout=0.3).to(self.device)
        
        # Load Scaler & Weights
        self.scaler = self._load_resource("scaler.pkl", is_pickle=True)
        self._load_resource("models/best_model.pth", is_model=True)

    def _load_resource(self, filename, is_pickle=False, is_model=False):
        paths = [filename, Path("models") / filename, Path(MODEL_DIR) / filename]
        for p in paths:
            if Path(p).exists():
                if is_pickle:
                    with open(p, 'rb') as f: return pickle.load(f)
                if is_model:
                    checkpoint = torch.load(p, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                    self.model.eval()
                    print(f"✓ Weights loaded: {p}")
        return None

    def predict(self, image_path: str) -> Dict[str, any]:
        signature = self.feature_extractor.extract_unified_signature(image_path)
        if self.scaler:
            signature = self.scaler.transform(signature.reshape(1, -1)).flatten()
        
        tensor = torch.from_numpy(signature).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        return {
            'prediction': "AI-Generated" if pred == 1 else "Real Camera Image",
            'ai_prob': probs[0, 1].item(),
            'device': str(self.device)
        }