# Real-Time_Driver_Fatigue_Detection_System

This repository implements a high-performance driver fatigue detection system optimized for the **NVIDIA Jetson Nano**. It combines real-time facial landmark tracking with deep learning classification to ensure road safety.

## Technical highlights

- **Multi-threaded Pipeline**: Decouples camera stream capture from inference loop using [camera.py](src/camera.py), preventing frame-dropping and blocking during model inference.

- **Hybrid Detection**: Utilizes **MediaPipe** to track facial landmarks and dynamically extract high-precision ROI for the eyes in varying lighting environments.

- **Edge-optimized Inference**: After comparing **ResNet-18** and **ResNet-34**, we chose **ResNet-18** for the system implementation to achieve a balance of 99% accuracy and low latency.

## Project Structure
Modular code is stored in the [`src`](src) directory.

```
src/
├── __init__.py         
├── camera.py           # Camera handling & threading
├── config.py           # Global configurations
├── detection.py        # Face & eye ROI detection
├── inference.py        # Integrated inference pipeline class
├── model.py            # Model architecture definition
└── preprocessing.py    # Image pre-processing utilities
```


## Model Training & Evaluation

- Dataset: Balanced subset (10k images) of the [MRL Eye dataset](https://mrl.cs.vsb.cz/eyedataset.html)

- Comparison: Similar accuracy around 99%, but lower latency on **ResNet-18**.

- Legacy research: Initial experiment and training logs are preserved in [notebooks](notebooks) directory.

## Quick Start

### Note: The current Docker environment is designed for refactor validation and core logic testing.

Run `./run_web.sh` to start the system with a web interface. (or `./run_simulation.sh` for CLI only mode)

