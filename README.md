# street_sign_detection_ml
The source code for the machine learning of the DCAITI project: KI-basierte Algorithmen zur Objektdetektion und Klassifizierung für mobile Plattformen

# Setup

## Data pre-procesing

wip...

## YOLO v5

### Install Requirements

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### Train

`python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128`

All parameters for training can be found [here](https://github.com/ultralytics/yolov5/blob/1ae91940abe9ca3e064784bb18c12271ab3157b4/train.py#L433)

Advanced:
* [How to choose parameters](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)
* [Train custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### Convert to TF Lite
Follow this [guide](https://github.com/ultralytics/yolov5/issues/251)

# Data
## Dataset
* https://benchmark.ini.rub.de/gtsdb_dataset.html
* https://www.cityscapes-dataset.com/dataset-overview/
* https://www.cvlibs.net/datasets/kitti-360/index.php

## Machine learning algorithms
* https://github.com/ultralytics/yolov5
* https://github.com/WongKinYiu/yolov7

## Data layout
**note: images and matching label need to have the same name** <br/>
Labels follow the formatting `class x_center y_center width height`

A sample dataset structure can be found in [data_layout.md](data_layout.md)
