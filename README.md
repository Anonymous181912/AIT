# Advanced Deepfake Detection System

## Background-Level Authenticity Verification

This project implements a novel deepfake detection system that goes beyond traditional facial analysis by focusing on **background-level authenticity verification**. The system analyzes camera sensor noise patterns, compression artifacts, frequency domain characteristics, and metadata signatures to distinguish between real camera-captured images and AI-generated images.

## Core Innovation

Unlike conventional deepfake detectors that focus on facial features (eye blinking, texture, landmarks), this system introduces a **background authenticity layer** that examines:

1. **Camera Sensor Noise Patterns**: Real cameras leave unique noise signatures that AI-generated images lack
2. **Frequency Domain Analysis**: FFT and DCT transforms reveal synthetic generation patterns
3. **Compression Artifacts**: Real images often have natural compression traces
4. **Metadata Signatures**: EXIF data completeness and anomalies indicate authenticity

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Image Input                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │   Preprocessing     │
        │  (Background Extract)│
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼────┐    ┌────▼────┐
│ FFT/  │    │ Noise  │    │Metadata │
│ DCT   │    │Pattern │    │Analysis │
└───┬───┘    └────┬────┘    └────┬────┘
    │             │              │
    └─────────────┼──────────────┘
                  │
          ┌───────▼───────┐
          │  Feature      │
          │  Signature    │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  Classifier   │
          │  (Neural Net) │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  Prediction   │
          │ Real vs AI    │
          └───────────────┘
```

## Features

### 1. Frequency Domain Analysis (`frequency_analysis.py`)
- **FFT-based features**: Magnitude spectrum, phase spectrum, radial frequency profiles
- **DCT-based features**: Block-level DCT analysis for compression artifact detection
- **High/low frequency energy**: Distinguishes natural vs synthetic frequency distributions

### 2. Noise Pattern Extraction (`noise_extraction.py`)
- **Noise residual extraction**: High-pass filtering to isolate sensor noise
- **Spatial correlation analysis**: Real noise has spatial structure
- **Compression artifact detection**: JPEG block boundary analysis

### 3. Metadata Inspection (`metadata_inspector.py`)
- **EXIF completeness**: Real images have rich metadata
- **Anomaly detection**: Identifies suspicious software signatures
- **Camera information**: Presence of make/model indicates authenticity

### 4. Background Feature Extraction (`background_features.py`)
- **Unified signature generation**: Combines all background features
- **Modular design**: Each feature type can work independently
- **Face region masking**: Focuses analysis on background regions

### 5. Classification Model (`classifier.py`)
- **Neural network architecture**: Multi-layer perceptron with batch normalization
- **Ensemble support**: Can combine with face-based detectors
- **Interpretability**: Feature importance analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepp

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── real/          # Real camera images
│   └── fake/          # AI-generated images
└── test/
    ├── real/
    └── fake/
```

### 2. Train the Model

```bash
python train.py
```

The training script will:
- Extract features from all training images
- Train a neural network classifier
- Save the best model to `models/best_model.pth`
- Generate training history

### 3. Run Inference

**Single image:**
```bash
python infer.py --image path/to/image.jpg
```

**With explanation:**
```bash
python infer.py --image path/to/image.jpg --explain
```

**Batch processing:**
```bash
python infer.py --batch path/to/directory/
```

### 4. Evaluate Model

```bash
python evaluate.py --test_dir data/test/
```

This will generate:
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- ROC curve
- Visualization plots

## Technical Details

### Feature Dimensions

- **Frequency features**: ~20 dimensions (FFT + DCT statistics)
- **Noise features**: ~14 dimensions (noise statistics + patterns)
- **Metadata features**: ~11 dimensions (EXIF completeness + anomalies)
- **Total signature**: ~45 dimensions

### Model Architecture

```
Input (45 dims) 
  → Linear(256) + BatchNorm + ReLU + Dropout
  → Linear(128) + BatchNorm + ReLU + Dropout  
  → Linear(64) + BatchNorm + ReLU + Dropout
  → Linear(2) [Real, AI-Generated]
```

### Why This Approach is Novel

1. **Background Focus**: Most detectors analyze faces; we analyze backgrounds
2. **Multi-Modal**: Combines frequency, noise, and metadata analysis
3. **Camera Signature**: Detects real camera capture characteristics
4. **Robust to Face Manipulation**: Works even when faces are well-generated

## Integration with Face Detection

The system can optionally integrate with face-based detectors:

```python
from face_integration import HybridDeepfakeDetector
from pipeline import DeepfakeDetectionPipeline

# Initialize
bg_pipeline = DeepfakeDetectionPipeline(model_path="models/best_model.pth")
hybrid = HybridDeepfakeDetector(bg_pipeline)

# Detect
result = hybrid.detect("image.jpg")
```

## Research Applications

This system is suitable for:
- **Hackathons**: Novel approach with clear innovation
- **Research**: Background authenticity is underexplored
- **Real-world deployment**: Robust to various image types
- **Forensics**: Metadata and noise analysis for authenticity verification

## Limitations & Future Work

1. **Face Model Integration**: Currently uses placeholder for face features
2. **Video Support**: Currently image-only (can be extended)
3. **Real-time Processing**: Optimization needed for production
4. **Adversarial Robustness**: Defense against adversarial attacks

## Citation

If you use this system in your research, please cite:

```bibtex
@software{deepfake_detection_2024,
  title={Background-Level Authenticity Verification for Deepfake Detection},
  author={Tanmay},
  year={2024},
  url={https://github.com/Anonymous181912/AIT}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

For questions or collaborations, please open an issue on GitHub.

