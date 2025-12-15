"""
Configuration file for Deepfake Detection System
Enhanced with GPU support and intelligent device selection
"""
import os
import torch


def get_device():
    """
    Intelligent device selection with fallback chain:
    CUDA GPU -> Apple MPS -> CPU
    """
    if torch.cuda.is_available():
        # Check for multiple GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Multiple GPUs detected ({num_gpus}). Using GPU 0.")
        return torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info():
    """Get detailed device information for logging"""
    device = get_device()
    info = {"device": str(device), "type": device.type}
    
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        info["cuda_version"] = torch.version.cuda
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon"
    else:
        info["gpu_name"] = "N/A (CPU)"
    
    return info


# Paths
DATASET_DIR = "dataset"  # Main dataset directory: dataset/real/ and dataset/fake/
DATA_DIR = "data"  # Legacy support
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Image processing
IMAGE_SIZE = (256, 256)
FACE_SIZE = (224, 224)  # For face classifier (EfficientNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Feature extraction
NOISE_PATCH_SIZE = 8
FFT_WINDOW_SIZE = 32
DCT_BLOCK_SIZE = 8
COMPRESSION_QUALITY_THRESHOLD = 85

# Texture features (new)
LBP_RADIUS = 3
LBP_POINTS = 24
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, 45, 90, 135]
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_ORIENTATIONS = [0, 45, 90, 135]

# Model parameters
HIDDEN_DIM = 256
NUM_CLASSES = 2  # Real vs AI-generated
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50

# GPU Training settings
DEVICE = get_device()  # Intelligent device selection
USE_MIXED_PRECISION = True  # Enable AMP for faster training on GPU
GRADIENT_ACCUMULATION_STEPS = 1  # Increase if GPU memory is limited
MAX_GRAD_NORM = 1.0  # Gradient clipping for stability

# DataLoader settings
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0  # Parallel data loading for GPU
PIN_MEMORY = DEVICE.type == "cuda"  # Pin memory for faster GPU transfer

# Training
SAVE_INTERVAL = 5
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation if no separate test set

# Feature weights for ensemble
BACKGROUND_WEIGHT = 0.6
FACE_WEIGHT = 0.4

# Face detection settings
FACE_DETECTOR_BACKEND = "mtcnn"  # Options: "mtcnn", "mediapipe", "opencv"
FACE_DETECTION_CONFIDENCE = 0.9
FACE_LANDMARKS = True

# Feature availability
USE_TEXTURE_FEATURES = True  # Enable texture features for metadata-independent detection
REQUIRE_METADATA = False  # If False, gracefully degrade when metadata is missing

