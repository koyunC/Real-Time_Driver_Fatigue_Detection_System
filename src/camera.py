import cv2
import threading
import time
import logging
import numpy as np

class ThreadedCamera:
    """
    Reads frames from the camera in a separate thread to avoid blocking the main processing loop.
    This improves FPS by ensuring the processing pipeline always has access to the most recent frame.
    """
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.mock_mode = False
        
        # Try 1: Explicit Path with V4L2
        if isinstance(self.src, int):
            dev_path = f"/dev/video{self.src}"
            logging.info(f"Attempting to open camera at {dev_path} with CAP_V4L2...")
            self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)

        # Try 2: Index with V4L2
        if not self.cap.isOpened() and isinstance(self.src, int):
            logging.warning(f"Failed to open {dev_path}. Trying index {self.src} with CAP_V4L2...")
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)

        # Try 3: Index with Default Backend
        if not self.cap.isOpened() and isinstance(self.src, int):
            logging.warning(f"Failed with V4L2. Trying index {self.src} with default backend...")
            self.cap = cv2.VideoCapture(self.src)

        if self.cap.isOpened():
            logging.info(f"Camera opened successfully: {self.src}")
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.grabbed, self.frame = self.cap.read()
        else:
            logging.warning(f"Cannot open camera source {src}. Switching to MOCK MODE.")
            self.mock_mode = True
            self.grabbed = True
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(self.frame, "NO CAMERA", (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.started = False
        self.read_lock = threading.Lock()
        self.stop_signal = False

    def start(self):
        if self.started:
            logging.warning("ThreadedCamera already started.")
            return self
        self.started = True
        self.stop_signal = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"Camera thread started. Mock mode: {self.mock_mode}")
        return self

    def update(self):
        while self.started:
            if self.stop_signal:
                break
                
            if self.mock_mode:
                # Simulate frame capture rate
                time.sleep(1.0 / self.fps)
                # Create a simple dummy animation (e.g. noise or moving bar) to prove it's alive
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Add timestamp to show changes
                cv2.putText(frame, f"MOCK CAMERA {time.time():.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Add random noise circle
                cx = int(self.width/2 + 100 * np.sin(time.time()))
                cy = int(self.height/2 + 100 * np.cos(time.time()))
                cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
                
                with self.read_lock:
                    self.grabbed = True
                    self.frame = frame
            else:
                grabbed, frame = self.cap.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    
                if not grabbed:
                    logging.warning("Failed to grab frame. Exiting camera thread.")
                    self.stop_signal = True

    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        self.stop_signal = True
        if hasattr(self, 'thread'):
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        logging.info("Camera thread stopped and resource released.")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
