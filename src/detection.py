import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
from src import config

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetector = mp.tasks.vision.FaceDetector
        self.FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        options = self.FaceDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=config.EYE_DETECTOR_MODEL_PATH),
            min_detection_confidence=min_detection_confidence,
            running_mode=self.VisionRunningMode.IMAGE)
        
        try:
            self.detector = self.FaceDetector.create_from_options(options)
            self.use_legacy = False
        except Exception as e:
            print(f"Failed to initialize MediaPipe FaceDetector: {e}")
            raise

    def process_frame(self, frame: np.ndarray):
        # Convert the image to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.detector.detect(mp_image)
    
    def extract_eyes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Tuple[int, int]]]]:
        """
        Detects face and extracts eye regions.
        Returns: (left_eye_img, right_eye_img, eyes_coordinates)
        eyes_coordinates is a list of tuples (x1, x2) where x1 and x2 are the left and right bounds of the eye region
        """
        results = self.process_frame(frame)
        height, width, _ = frame.shape
        
        if not results.detections:
            return None, None, None

        detection = results.detections[0]
        bbox = detection.bounding_box
        
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)
        
        screen_left_eye_x = int(x + w * 0.3) 
        screen_right_eye_x = int(x + w * 0.7)
        eye_y = int(y + h * 0.15)
        
        eyes_coor = [
            (screen_left_eye_x - config.EYE_HORIZONTAL_OFFSET, screen_left_eye_x + config.EYE_HORIZONTAL_OFFSET),
            (screen_right_eye_x - config.EYE_HORIZONTAL_OFFSET, screen_right_eye_x + config.EYE_HORIZONTAL_OFFSET),
            (eye_y - config.EYE_VERTICAL_OFFSET, eye_y + config.EYE_VERTICAL_OFFSET)
        ]
        
        # Boundary checks
        for i in range(2):
           if eyes_coor[i][0] < 0: eyes_coor[i] = (0, eyes_coor[i][1])
           if eyes_coor[i][1] > width: eyes_coor[i] = (eyes_coor[i][0], width)
        
        if eyes_coor[2][0] < 0: eyes_coor[2] = (0, eyes_coor[2][1])
        if eyes_coor[2][1] > height: eyes_coor[2] = (eyes_coor[2][0], height)

        # Extract images
        subject_right_eye_img = frame[eyes_coor[2][0]:eyes_coor[2][1], eyes_coor[0][0]:eyes_coor[0][1]]
        subject_left_eye_img = frame[eyes_coor[2][0]:eyes_coor[2][1], eyes_coor[1][0]:eyes_coor[1][1]]
        
        return subject_left_eye_img, subject_right_eye_img, eyes_coor
