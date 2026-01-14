import torch
import torch.nn as nn
from torchvision import models
import logging

def load_model(model_path: str, num_classes: int = 2, device: torch.device = "cuda") -> nn.Module:
    logging.info(f"Loading model from {model_path} to {device}")
    
    model = models.resnet18(pretrained=False) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    model.to(device)
    model.eval()
    return model
