import cv2
import time
import argparse
import logging
import sys
from collections import deque
from src import config
from src.camera import ThreadedCamera
from src.detection import FaceDetector
from src.model import load_model
from src.inference import EyeStateClassifier
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run_detection(camera_idx, device_name, model_path, frame_callback=None):
    # Initialize components
    try:
        logging.info("Initializing system...")
        
        # Device
        device = torch.device(device_name)
        logging.info(f"Using device: {device}")

        # Model
        model = load_model(model_path, num_classes=config.NUM_CLASSES, device=device)
        classifier = EyeStateClassifier(model, device)
        
        # Detector
        detector = FaceDetector()
        
        # Camera
        camera = ThreadedCamera(src=camera_idx, width=config.CAP_PROP_FRAME_WIDTH, height=config.CAP_PROP_FRAME_HEIGHT, fps=config.CAP_PROP_FPS)
        camera.start()
        
        # State tracking
        eyes_states_history = deque(maxlen=config.HISTORY_LEN)
        # 1 for safe (open), 0 for danger (closed)
        current_state = 1 
        
        prev_time = time.time()
        frame_count = 0
        headless = False
        
        logging.info("System started. Press 'q' to quit (if window is open).")
        
        while True:
            available, frame = camera.read()
            if not available:
                time.sleep(0.01)
                continue
            
            frame_count += 1

            # Process frame
            # 1. Detect eyes
            left_img, right_img, eyes_coor = detector.extract_eyes(frame)
            
            num_opened_eyes = 0
            
            # 2. Predict
            if left_img is not None and right_img is not None:
                # If no eyes detected, defaults to 0 (closed) which implies danger

                # Draw boxes around eyes
                # eyes_coor: [left_eye_x_range, right_eye_x_range, y_range]
                # Recall: 0 -> person's right (screen left), 1 -> person's left (screen right)
                
                # Screen Left Eye (Subject Right)
                cv2.rectangle(frame, 
                              (eyes_coor[0][0], eyes_coor[2][0]), 
                              (eyes_coor[0][1], eyes_coor[2][1]), 
                              (255, 0, 0), 2)
                
                # Screen Right Eye (Subject Left)
                cv2.rectangle(frame, 
                              (eyes_coor[1][0], eyes_coor[2][0]), 
                              (eyes_coor[1][1], eyes_coor[2][1]), 
                              (255, 0, 0), 2)

                num_opened_eyes = classifier.predict(left_img, right_img)

            # 3. Update State
            is_open = 1 if num_opened_eyes > 0 else 0
            eyes_states_history.append(is_open)
            
            # Danger logic: if history is full and ALL are closed (0)
            if len(eyes_states_history) == config.HISTORY_LEN:
                # Check if all recent frames are closed
                if sum(eyes_states_history) == 0:
                    current_state = 0 # DANGER
                else:
                    current_state = 1 # SAFE
            else:
                # Not enough history, assume safe
                current_state = 1

            # 4. Visualization
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
            prev_time = curr_time
            
            # Draw Status
            if current_state == 0:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Safe", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Open Eyes: {num_opened_eyes}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if headless and frame_count % 30 == 0:
                 logging.info(f"Frame {frame_count} - FPS: {fps:.1f} - State: {'SAFE' if current_state == 1 else 'DANGER'} - Eyes Open: {num_opened_eyes}")

            if frame_callback:
                # Use external callback (e.g. for Web)
                if not frame_callback(frame):
                    break
            else:
                # Default local display
                try:
                    cv2.imshow("Driver Fatigue Detection", frame)
                    # Exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    # Fallback for headless environments
                    if not headless:
                        logging.warning(f"Display failed (running headless?): {e}")
                        headless = True
                    pass
                
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if 'camera' in locals() and hasattr(camera, 'stop'):
            camera.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logging.info("System cleanup complete.")

def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default=config.DEVICE.type, help="Device to use (cpu/cuda)")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Path to model weights")
    args = parser.parse_args()

    run_detection(args.camera, args.device, args.model)

if __name__ == "__main__":
    main()
