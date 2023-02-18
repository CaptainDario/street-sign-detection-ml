#! /bin/bash

echo 'Installing requirements'
pip install -r requirements.txt
pip install ultralytics

# train with multi-gpu
# x
yolo train data=streetsign_data.yaml model=yolov8x.pt epochs=300 imgsz=1280 device=\'0,1,2,3,4,5,6,7\' batch=32 name=yolov8x
# l 
yolo train data=streetsign_data.yaml model=yolov8l.pt epochs=300 imgsz=1280 device=\'0,1,2,3,4,5,6,7\' batch=32 name=yolov8l
# m
yolo train data=streetsign_data.yaml model=yolov8m.pt epochs=300 imgsz=1280 device=\'0,1,2,3,4,5,6,7\' batch=32 name=yolov8m
# s
yolo train data=streetsign_data.yaml model=yolov8s.pt epochs=300 imgsz=1280 device=\'0,1,2,3,4,5,6,7\' batch=32 name=yolov8s
# n
yolo train data=streetsign_data.yaml model=yolov8n.pt epochs=300 imgsz=1280 device=\'0,1,2,3,4,5,6,7\' batch=32 name=yolov8n
