# Real-Time Object Detection with YOLOv8

Real-time object detection using YOLOv8 and OpenCV.
Detects 80 object classes from live webcam or image input at 30+ FPS.

## Live Demo
[Try it live](YOUR_HUGGINGFACE_URL)

## Demo
![Detection Demo](demo.gif)

## Tech Stack
- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV 4.8

## How to Run
git clone https://github.com/grgaryan2005-cloud/ai-object-detection-yolov8
pip install -r requirements.txt
python detect.py

## Results
- mAP50: 37.3 on COCO val2017
- Real-time at 30+ FPS on CPU
- Detects 80 object classes
## What I Learned
- YOLOv8 uses CSP backbone + PANet neck for multi-scale detection
- Non-maximum suppression (NMS) removes duplicate bounding boxes
- Confidence threshold controls precision vs recall tradeoff
- COCO dataset has 80 classes — model learns different features per class