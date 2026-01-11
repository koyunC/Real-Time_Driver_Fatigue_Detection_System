import cv2
import numpy as np
from src import config

def apply_clahe(image: np.ndarray) -> np.ndarray:
    # Applies Contrast Limited Adaptive Histogram Equalization
    
    # Convert to gray scale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    equalized = clahe.apply(gray)
    
    # Convert back to BGR if input was BGR
    if len(image.shape) == 3:
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return equalized

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # Adjusts the gamma of the image
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def normalize_image(image: np.ndarray) -> np.ndarray:
    # Normalizes the image pixel values to 0-255 range
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Full preprocessing pipeline for a frame
    image = normalize_image(image)
    image = apply_clahe(image)
    image = adjust_gamma(image, config.GAMMA_VALUE)
    return image
