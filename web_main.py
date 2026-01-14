import cv2
import time
import argparse
import logging
import sys
import threading
from flask import Flask, Response, render_template_string
import numpy as np

from src import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)

# Global variables for the latest frame and lock
output_frame = None
lock = threading.Lock()

from main import run_detection

def detect_fatigue(args):
    global output_frame, lock
    
    def frame_callback(frame):
        global output_frame, lock
        with lock:
            output_frame = frame.copy()
        # Return True to continue loop
        return True

    # Call the reusable function from main.py
    run_detection(args.camera, args.device, args.model, frame_callback=frame_callback)

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template_string("""
        <html>
          <head>
            <title>Driver Fatigue Detection</title>
            <style>
              body { background-color: #333; color: white; font-family: sans-serif; text-align: center; }
              img { border: 5px solid #555; border-radius: 10px; margin-top: 20px; max-width: 100%; }
            </style>
          </head>
          <body>
            <h1>Driver Fatigue Detection System</h1>
            <img src="/video_feed">
          </body>
        </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Fatigue Detection System (Web)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default=config.DEVICE.type, help="Device to use (cpu/cuda)")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Path to model weights")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    
    args = parser.parse_args()
    
    t = threading.Thread(target=detect_fatigue, args=(args,))
    t.daemon = True
    t.start()
    
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
