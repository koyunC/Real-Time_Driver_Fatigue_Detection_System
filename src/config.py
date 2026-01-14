import torch
import os
import cv2

# VGG-18 Model Configuration
MODEL_PATH = os.path.join("models", "resnet18-e_20-d_10k.pth")
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['close', 'open']
NUM_CLASSES = 2

# Eye Extraction Configuration
EYE_VERTICAL_OFFSET = 35
EYE_HORIZONTAL_OFFSET = 35
EYE_DETECTOR_MODEL_PATH = os.path.join("models", "detector.tflite")

# System Configuration
HISTORY_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAMMA_VALUE = 1.2

# Camera Configuration
CAP_PROP_FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
CAP_PROP_FRAME_WIDTH = 640
CAP_PROP_FRAME_HEIGHT = 480
CAP_PROP_FPS = 30
