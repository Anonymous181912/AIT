"""
Configuration file for Deepfake Detection System
"""
import os

# Paths
DATA_DIR = "data"
# Paths
TRAIN_DIR = "train"
TEST_DIR  = "test"
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
DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
NUM_WORKERS = 4
SAVE_INTERVAL = 5

# Feature weights for ensemble
BACKGROUND_WEIGHT = 0.6
FACE_WEIGHT = 0.4

