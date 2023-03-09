<img class="center" src="https://github.com/CaptainDario/street_sign_detection_app/blob/main/assets/icon/icon.png?raw=true" height=75><img class="center" src="https://github.com/CaptainDario/street_sign_detection_app/blob/main/assets/icon/dcaiti.png?raw=true" height=75>

The source code of the machine learning part of the DCAITI project: "KI-basierte Algorithmen zur Objektdetektion und Klassifizierung für mobile Plattformen".
The project of the flutter app can be found [here](https://github.com/CaptainDario/street_sign_detection_app).

# Setup

## Data pre-procesing

### Data pre-procesing

Install the necessary packages and run the data scripts to convert the dataset.
The conversion includes converting the coordinates from leftmost, upmost, rightmost and downmost to center_x, center_y, width, height and prepending the class label.
The data will be distributed for their respective parts regarding training, validation and test percentage. 
As many images are multilabel-samples with multiple labels, it is not possible to perfectly partition the data for the respective percentages.
For the distribution, first multilabel samples are distributed and then singlelabel samples. This happens successively for training, validation and testing, which can, but must not benefit the training sample size, please keep that in mind. Before the distribution, the dataset gets shuffled to have as much variance for the samples and their labels. 

Afterwards annotating the data is possible with these converted coordinates. The annotated dataset should include BBoxes for all defined class-labels in each image.

``` bash
python -m pip install -r dataset_creation/requirements.txt
python dataset_creation/dataset_convert.py <path/to/the/downloaded/dataset> <path/where/the/converted/dataset/to/be/stored> <train_percentage> <val_percentage> <test_percentage>
python dataset_creation/dataset_annotation.py <path/to/the/downloaded/dataset> <path/to/the/converted/coordinates>** <path/to/the/annotated/dataset/to/be/stored>
```

## YOLO
In the respective YOLO-Directory can be found the source code of the YOLO-Algorithm as well as scripts for training a dataset. 

### Train

The configurations used to train yolo v5 can be found in [train folder](train_yolov5).

The parameters for training are explained [here](https://github.com/ultralytics/yolov5/blob/1ae91940abe9ca3e064784bb18c12271ab3157b4/train.py#L433)

Advanced:
* [How to choose parameters](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)
* [Train custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### Convert to TF Lite
Follow this [guide](https://github.com/ultralytics/yolov5/issues/251)

# Data
## Dataset
The following GTSDB dataset can be applied for the data preparation.
* https://benchmark.ini.rub.de/gtsdb_dataset.html

## Data layout
**note: images and matching label need to have the same name** <br/>
Labels follow the formatting `class x_center y_center width height`

A sample dataset structure can be found in [data_layout.md](data_layout.md)
