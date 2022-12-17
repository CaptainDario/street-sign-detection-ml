# street_sign_detection_ml

The source code for the machine learning of the DCAITI project: KI-basierte Algorithmen zur Objektdetektion und Klassifizierung für mobile Plattformen

## Setup

### Data pre-procesing

Install the necessary packages and run the data conversion script.

``` bash
python -m pip install -r dataset_creation/requirements.txt
python dataset_creation/dataset_convert.py path/to/the/downloaded/dataset path/where/the/converted/dataset/should/be/stored
```

### YOLO v5

#### Install Requirements

``` bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

#### Train

Use `train.py` example:
`python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128`

All parameters for training can be found [here](https://github.com/ultralytics/yolov5/blob/1ae91940abe9ca3e064784bb18c12271ab3157b4/train.py#L433)

Advanced:

* [How to choose parameters](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)
* [Train custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

#### Convert to TF Lite

Follow this [guide](https://github.com/ultralytics/yolov5/issues/251)

## Resources

### Dataset

* [gtsdb dataset](https://benchmark.ini.rub.de/gtsdb_dataset.html)
* [cityscapes dataset](https://www.cityscapes-dataset.com/dataset-overview/)
* [kitti-360 dataset](https://www.cvlibs.net/datasets/kitti-360/index.php)

### Machine learning algorithms

* [YOLO v5](https://github.com/ultralytics/yolov5)
* [YOLO v7](https://github.com/WongKinYiu/yolov7)

### Data layout

**note: images and matching label need to have the same name** <br/>
Labels follow the formatting `class x_center y_center width height`

A sample dataset structure can be found in [data_layout.md](data_layout.md)
