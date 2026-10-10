# Real-Time Object Detection System Using YOLOv4 and YOLOv11

## Overview

This project implements a real-time object detection system using two versions of the YOLO (You Only Look Once) object detection framework: YOLOv4 and YOLOv11.

The system processes live video from a smartphone camera using an IP Webcam stream and detects objects in each frame. It displays the detected objects with bounding boxes, class labels, and confidence scores.

The YOLOv4 implementation uses OpenCV's DNN module to load and run the pretrained YOLOv4 model. The YOLOv11 implementation uses the Ultralytics YOLO API.

This repository contains both implementations for learning, experimentation, and comparison of object detection approaches.

## Features

- Real-time object detection from an IP Webcam video stream
- Object detection using YOLOv4 and YOLOv11
- Bounding-box visualization around detected objects
- Object class labels and confidence scores
- OpenCV-based video capture and display
- Frame skipping and threaded camera reading in the optimized YOLOv4 implementation
- FPS and inference-time display in the optimized YOLOv4 implementation
- Separate implementations for two YOLO model versions

## Technologies Used

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- YOLOv4
- YOLO11
- Computer Vision
- Deep Learning

## Project Structure

```text
Object-Detection-System/
│
├── v4/
│   ├── object.py
│   ├── object2.py
│   ├── coco.names
│   ├── yolov4.cfg
│   └── requirements.txt
│
├── v11/
│   ├── object_new.py
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

**Model weights:** The YOLOv4 weights file and any locally downloaded YOLO11 model weights are not included in the repository.

## Requirements

- Python 3
- pip
- A webcam or smartphone running an IP Webcam application
- A local network connection accessible to both the computer and smartphone
- YOLOv4 configuration, class names, and pretrained weights for the YOLOv4 implementation
- Internet access for installing dependencies and downloading the YOLO11 model when needed

## Installation and Setup

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Subhransu-git/Object-Detection-System.git
cd Object-Detection-System
```

### 2. Set up YOLOv4

Open Command Prompt in the repository root and run:

```bat
cd v4
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `v4/requirements.txt` file should contain:

```text
opencv-python
numpy
```

### 3. Download YOLOv4 model files

The YOLOv4 implementation requires these files in the `v4` directory:

- `yolov4.cfg` — model configuration
- `yolov4.weights` — pretrained model weights
- `coco.names` — object class labels

The configuration and class-name files are included in this repository. The weights file must be obtained separately from a trusted source.
https://sourceforge.net/projects/darknet-yolo.mirror/files/yolov4/yolov4.weights/download

Ensure the downloaded weights file is named `yolov4.weights` and is placed in the `v4` directory.

### 4. Set up YOLOv11

Return to the repository root:

```bat
cd ..
cd v11
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `v11/requirements.txt` file should contain:

```text
ultralytics
opencv-python
```

The YOLOv11 implementation loads the model using:

```python
model = YOLO("yolo11n.pt")
```

Ultralytics will normally download the pretrained YOLO11 Nano model automatically if the weights are not already available locally and the download is accessible.

## Configure the Smartphone Camera

Both implementations use an IP Webcam video stream.

### Steps

1. Install an IP Webcam application on your smartphone.
2. Connect your smartphone and computer to the same local network.
3. Start the video server in the smartphone application.
4. Find the IP address and video-stream URL displayed by the application.
5. Update the URL in the relevant Python script.

For `v4/object.py` and `v11/object_new.py`, replace the existing video URL with your own:

```python
url = "http://YOUR_PHONE_IP:8080/video"
```

For `v4/object2.py`, update the `camera_url` variable instead:

```python
camera_url = "http://YOUR_PHONE_IP:8080/video"
```

Replace `YOUR_PHONE_IP` and adjust the port or path if your IP Webcam application uses different settings.

**Note:** The example URL is a placeholder. The computer must be able to access the configured video stream.

## How to Run the Project

### Run YOLOv4

Activate the YOLOv4 virtual environment from the repository root:

```bat
v4\.venv\Scripts\activate
python v4\object.py
```

Alternatively, run the optimized implementation:

```bat
python v4\object2.py
```

The optimized implementation uses a separate camera-reading thread, skips frames to reduce inference workload, and displays FPS and YOLO inference time.

### Run YOLOv11

Activate the YOLOv11 virtual environment:

```bat
v11\.venv\Scripts\activate
python v11\object_new.py
```

The application displays the live video stream with object detections drawn by the Ultralytics YOLO API.

Press **Q** to close the detection window.

**Important:** Use a separate terminal for each implementation, and ensure its corresponding virtual environment is activated.

## How It Works

### YOLOv4 Implementation

The YOLOv4 scripts:

1. Load the pretrained YOLOv4 network using OpenCV's DNN module.
2. Read class labels from `coco.names`.
3. Capture frames from the IP Webcam video stream.
4. Preprocess frames into input blobs.
5. Run inference through the YOLOv4 network.
6. Filter detections using a confidence threshold.
7. Apply Non-Maximum Suppression to remove overlapping detections.
8. Draw bounding boxes, labels, and confidence scores.
9. Display the processed video.

The `object2.py` implementation additionally uses a camera-reading thread and displays FPS and model inference time.

### YOLOv11 Implementation

The YOLOv11 script:

1. Loads the pretrained YOLO11 Nano model.
2. Connects to the IP Webcam video stream.
3. Reads incoming video frames.
4. Runs object detection using the Ultralytics YOLO API.
5. Draws detection results using the model's plotting functionality.
6. Displays the annotated video stream.

## Model Configuration

The YOLOv4 implementation uses the configuration and weights files loaded by OpenCV.

The YOLOv11 implementation uses the pretrained `yolo11n.pt` model, where `n` denotes the Nano model variant.

The two implementations use different model-loading APIs and should be configured and run independently.

## Troubleshooting

### Camera connection failed

- Verify that the IP Webcam server is running.
- Confirm that the smartphone and computer can communicate over the network.
- Check the IP address, port, and video-stream path.
- Confirm that the URL works in a browser or compatible video client.

### YOLOv4 weights file not found

Ensure `yolov4.weights` is present in the `v4` directory and that the script is run with the expected working directory.

### YOLO11 model download failed

Check your internet connection and verify that the Ultralytics package can access the model download source.

### Missing Python packages

Activate the correct virtual environment and reinstall the dependencies:

```bash
pip install -r requirements.txt
```

Run this command from the relevant `v4` or `v11` directory.

### Video window does not open

Check that OpenCV was installed correctly and that your Python environment supports graphical display windows.

## Future Improvements

- Compare YOLOv4 and YOLO11 detection performance
- Measure inference speed under the same test conditions
- Evaluate detection accuracy on a defined dataset
- Add support for local video files and static images
- Add configurable confidence thresholds
- Improve camera connection handling
- Support custom-trained object detection models
- Include example detection screenshots and benchmark results

## Author

**Subhransu**

GitHub: [@Subhransu-git](https://github.com/Subhransu-git)

## License

No license has been specified yet. Add an appropriate license if you intend to permit others to reuse, modify, or distribute this project.
