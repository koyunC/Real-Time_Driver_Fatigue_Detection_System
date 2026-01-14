import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple
from src import config
from src.preprocessing import preprocess_image

class EyeStateClassifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, left_eye_img: np.ndarray, right_eye_img: np.ndarray) -> int:
        """
        Predicts the number of open eyes (0, 1, or 2).
        Returns:
            int: Number of open eyes.
        """
        if left_eye_img is None or right_eye_img is None:
            return 0 
        
        left_eye_proc = preprocess_image(left_eye_img)
        right_eye_proc = preprocess_image(right_eye_img)

        left_tensor = self.transform(Image.fromarray(left_eye_proc)).unsqueeze(0).to(self.device)
        right_tensor = self.transform(Image.fromarray(right_eye_proc)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs_left = self.model(left_tensor)
            outputs_right = self.model(right_tensor)
            _, predicted_left = torch.max(outputs_left, 1)
            _, predicted_right = torch.max(outputs_right, 1)

        return predicted_left.item() + predicted_right.item()
