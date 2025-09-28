# Real-Time_Driver_Fatigue_Detection_System
This repository contains a real-time driver fatigue detection system implemented in **Python/PyTorch** and optimized for the **Jetson Nano**. The project includes both model training and system implementation.

In the training part, we compare the performance of **ResNet-18** and **ResNet-34** on a subset(N=10k) of the [MRL Eye dataset](https://mrl.cs.vsb.cz/eyedataset.html).

For the system implementation part, we use a pipeline that consists of face detection, eyes regions extraction, image pre-processing, and eyes state classification. The system is optimized to run in real-time on the Jetson Nano.

## Demo Video

[**Youtube link**](https://youtu.be/qRJu7e0x36s)

**Note**: The demo video is recorded on a machine with an *Intel Core i7-10710U CPU* and *NVIDIA GeForce GTX 1650 with Max-Q Design*. The performance on the Jetson Nano can only achieve 2-3 FPS.

## Model Training

The training code is located in the [`model_training`](model_training) directory. It includes scripts for dataset preparation, data preprocessing, model training, and evaluation.

### Dataset Preparation

- [`gen_subset.py`](model_training/gen_subset.py): Generate a balanced subset of the MRL Eye dataset with given subset size. In this project, the subset size is set to 10k.

### Training and Evaluation

The training and evaluation scripts are located in the jupyter notebooks: [`resNet18_training.ipynb`](model_training/resNet18_training.ipynb) and [`resNet34_training.ipynb`](model_training/resNet34_training.ipynb). The training process was done on a machine with a *NVIDIA GeForce GTX 1650 with Max-Q Design*.

Both training use the following hyperparameters:

- Optimizer: SGD (lr=0.001, momentum=0.9)
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Number of Epochs: 20

They both achieve over 99% accuracy on the validation set, considering the limited performance we can get from Jetson Nano, we chose ResNet-18 for the system implementation.

## System Implementation

The system implementation code is located in the [`implementation/CompactVersion.ipynb`](implementation/CompactVersion.ipynb).

### Pipeline

The pipeline consists of four main components:

1. **Image pre-processing**: Capturing video frames from the webcam and applying normalization, CLAHE, and gamma adjustment.
2. **Face Detection**: Using MediaPipe's Face Detection module to detect eyes areas and calculate the center coordinates of the eyes.
3. **Eye Region Extraction**: Extracting the eye regions in a 70px*70px square based on the calculated coordinates.
4. **Eye State Classification**: Using the trained ResNet-18 model to classify the eye state as open or closed.
5. **Drowsiness Alert**: Triggering an alert if the eyes are detected as closed for continuously 3 frames in a 5 frame window.

## Other files and directories

- [`model`](model): Contains the trained ResNet-18, ResNet-34, and mediapipe model weights.
- [`requirements.txt`](requirements.txt): Lists the required Python packages and their versions.
