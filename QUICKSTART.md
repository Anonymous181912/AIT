# Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Data Preparation

2. **Organize your dataset:**
```
data/
├── train/
│   ├── real/          # Real camera images (.jpg, .png, .jpeg)
│   └── fake/          # AI-generated images
└── test/
    ├── real/
    └── fake/
```

## Training

3. **Train the model:**
```bash
python train.py
```

This will:
- Extract background features from all images
- Train a neural network classifier
- Save the best model to `models/best_model.pth`
- Generate training history

**Expected output:**
```
Loading training dataset...
Extracting features from images...
Loaded 1000 samples
Class distribution: [500 500]

Starting training...
Epoch 1/50
Training: 100%|████████| 32/32 [00:30<00:00]
Validating: 100%|████████| 8/8 [00:05<00:00]
Train Loss: 0.6234, Train Acc: 65.00%
Val Loss: 0.5891, Val Acc: 68.75%
Saved best model with validation accuracy: 68.75%
```

## Inference

4. **Test on a single image:**
```bash
python infer.py --image path/to/your/image.jpg
```

**Output:**
```
==================================================
PREDICTION RESULT
==================================================
Image: path/to/your/image.jpg
Prediction: Real Camera Image
Confidence: 0.9234
Real Probability: 0.9234
AI-Generated Probability: 0.0766
```

5. **Get detailed explanation:**
```bash
python infer.py --image path/to/your/image.jpg --explain
```

6. **Batch processing:**
```bash
python infer.py --batch path/to/directory/
```

## Evaluation

7. **Evaluate on test set:**
```bash
python evaluate.py --test_dir data/test/
```

**Output:**
```
============================================================
EVALUATION RESULTS
============================================================

Overall Metrics:
  Accuracy:  0.8750
  Precision: 0.8762
  Recall:    0.8750
  F1-Score:  0.8750
  ROC-AUC:   0.9234

Per-Class Metrics:
  Real Images:
    Precision: 0.8900
    Recall:    0.8600
    F1-Score:  0.8747
  AI-Generated Images:
    Precision: 0.8625
    Recall:    0.8900
    F1-Score:  0.8760
```

## Understanding the System

### How Background Authenticity Works

1. **Frequency Analysis**: 
   - Real images have natural frequency distributions
   - AI-generated images show artificial patterns in FFT/DCT

2. **Noise Patterns**:
   - Real cameras leave sensor noise signatures
   - AI-generated images lack consistent noise structure

3. **Metadata**:
   - Real images have complete EXIF data (camera, settings, datetime)
   - AI-generated images often lack or have suspicious metadata

### Feature Extraction Process

```
Image → Preprocessing → Background Extraction
  ↓
Frequency Analysis (FFT/DCT) → 20 features
Noise Extraction → 14 features  
Metadata Inspection → 11 features
  ↓
Unified Signature (45 features) → Classifier → Prediction
```

## Troubleshooting

### "Model file not found"
- Train the model first: `python train.py`
- Or specify model path: `--model path/to/model.pth`

### "Image file not found"
- Check the image path is correct
- Supported formats: .jpg, .jpeg, .png

### "CUDA out of memory"
- Use CPU: `--device cpu`
- Reduce batch size in `config.py`

### "No module named 'X'"
- Install missing dependencies: `pip install -r requirements.txt`

## Next Steps

1. **Experiment with different architectures** in `classifier.py`
2. **Tune hyperparameters** in `config.py`
3. **Add face detection integration** using `face_integration.py`
4. **Extend to video** by processing frames
5. **Deploy as API** using Flask/FastAPI

## Example Code

See `example_usage.py` for more detailed examples:

```python
from pipeline import DeepfakeDetectionPipeline

# Initialize
pipeline = DeepfakeDetectionPipeline(
    model_path="models/best_model.pth",
    device="cpu"
)

# Predict
result = pipeline.predict("image.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Performance Tips

1. **Use GPU** for faster training: Set `DEVICE = "cuda"` in `config.py`
2. **Batch processing**: Process multiple images together
3. **Feature caching**: Save extracted features to avoid recomputation
4. **Model optimization**: Use quantization for deployment

## Research Applications

This system is novel because it:
- Focuses on **background** rather than faces
- Uses **multi-modal** analysis (frequency + noise + metadata)
- Detects **camera signatures** that AI can't replicate
- Works even when **faces are well-generated**

Perfect for:
- Hackathons (novel approach)
- Research papers (background authenticity)
- Real-world deployment (robust detection)
- Forensics (metadata analysis)

