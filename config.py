"""
Configuration file for Deepfake Detection System
"""
import os

# Paths
DATASET_DIR = "dataset"  # Main dataset directory: dataset/real/ and dataset/fake/
DATA_DIR = "data"  # Legacy support
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Image processing
IMAGE_SIZE = (256, 256)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Feature extraction
NOISE_PATCH_SIZE = 8
FFT_WINDOW_SIZE = 32
DCT_BLOCK_SIZE = 8
COMPRESSION_QUALITY_THRESHOLD = 85

# Model parameters
HIDDEN_DIM = 256
NUM_CLASSES = 2  # Real vs AI-generated
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Training
DEVICE = "cpu"  # Default to CPU for foolproof training
NUM_WORKERS = 0  # Set to 0 for Windows compatibility and simplicity
SAVE_INTERVAL = 5
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation if no separate test set

# Feature weights for ensemble
BACKGROUND_WEIGHT = 0.6
FACE_WEIGHT = 0.4

