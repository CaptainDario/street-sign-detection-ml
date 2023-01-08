import sys
from os import path, makedirs, listdir
from shutil import copy2

import pandas as pd
import cv2 as cv

import utils
import dataset_distribution


def convert(df: pd.DataFrame) -> pd.DataFrame:
    ''' converts coordinates and returns them serialized into columns
    '''
    print("\tConverting coordinates")
    coords_tuples = df.apply(lambda row: convert_coordinates((row[1], row[2]), (row[3], row[4])), axis=1)
    coords_tuples = coords_tuples.apply(pd.Series)
    df.iloc[:, 1:5] = coords_tuples
    return df


def create_dataset_folder(dataset_path : str):
    """ Creates a dataset folder structure matching the YOLO v5/7
    input.

    Args:
        path (str): The path where the folder structure should be
                    created.
    """
    for i in ["images", "labels"]:
        for j in ["train", "test", "val"]:
            makedirs(path.join(dataset_path, i, j), exist_ok=True)


def read_data(in_file: str) -> pd.DataFrame:
    ''' reads data frome a relative or absolute path as a String,
        separates data with ';' \n
        returns dataframe
    '''
    print('\tReading data from input file')
    df = pd.read_csv(in_file, sep=';', header=None)
    return df


def write_data(df: pd.DataFrame, out_dir: str):
    ''' writes out data of each picture into a separate file,
        appends multiple labels of the same picture into the 
        same output file using newline as a delimiter
    '''
    print("\tWriting labels to files")
    makedirs(out_dir, exist_ok=True)
    for index, row in df.iterrows():
        utils.print_progress_bar(
            iteration= index+1,
            total= len(df),
            prefix= '\tWritten Files:\t\t\t',
            suffix= 'Complete',
            length= 30
        )
        file_name = row['sample'].replace(".ppm", ".txt")
        file_path = path.join(out_dir, "labels", str(row['flag']), file_name)
        if (path.exists(file_path)):
            with open(file_path, 'a') as f:
                f.write(f"\n{row['label']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}")
        else:
            with open(file_path, 'w') as f:
                f.write(f"{row['label']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}")


def convert_and_copy_relevant_pictures(coordinate_dir: str, samples_dir: str, output_dir: str, flag:str):
    ''' copies all pictures that have coordiantes into the specified `output_dir`,
        the location of the coordiante-pictures is based on the specified `coordiante_dir`
        and the pictures to be copies are taken from the specified `samples_dir`
    '''
    coordinate_elements = listdir(coordinate_dir)
    for index in range(len(coordinate_elements)):
        utils.print_progress_bar(
            iteration= index+1,
            total= len(coordinate_elements),
            prefix= f'\tConverted Images ({flag}):  \t',
            suffix= 'Complete',
            length= 30
        )
        sample_path = path.join(
            samples_dir,
            coordinate_elements[index].replace('.txt', '.ppm')
        )
        img = cv.imread(sample_path)
        converted_path = path.join(
            output_dir,
            coordinate_elements[index].replace('.txt', '.png')
        )
        cv.imwrite(converted_path, img)


def convert_coordinates(top_left: list, bottom_right: list) -> list:
    ''' top_left: [leftmost image: float, upmost image: float]\n
        bottom_right: [rightmost image: float, downmost image]\n
        returns: [x_center: float, y_center: float, width: float, height: float]

        conversion of given coordinates to center coordinates for x,y and with, height
        of the bounding box with normalization to [0,1] to a percentage value.
    '''
    pixels_x, pixels_y = 1360, 800
    x_center = (top_left[0] + bottom_right[0]) / 2 / pixels_x
    y_center = (top_left[1] + bottom_right[1]) / 2 / pixels_y
    width = (bottom_right[0] - top_left[0]) / pixels_x
    height = (bottom_right[1] - top_left[1]) / pixels_y
    return x_center, y_center, width, height


def parse_args():
    if len(sys.argv) == 8:
        return sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    elif len(sys.argv) == 6:
        return sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), 1360, 800, 
    else:
        print("Usage: python dataset_convert.py <input_directory> <output_directory> <train_percentage> <val_percentage> <test_percentage> [<pixels_x> <pixels_y>]")
        exit(1)


if __name__ == '__main__':
    in_dir, out_dir, train_percentage, val_percentage, test_percentage, pixels_x, pixels_y, = parse_args()
    label_file = path.join(in_dir, "gt.txt")

    create_dataset_folder(out_dir)
    df = read_data(label_file)
    convert(df)
    df = dataset_distribution.distribute_samples_flow(
        df= df,
        train_percentage= train_percentage,
        val_percentage= val_percentage,
        test_percentage= test_percentage
    )
    write_data(df, out_dir)
    for flag in ['train', 'val', 'test']:
        convert_and_copy_relevant_pictures(
            coordinate_dir= path.join(out_dir, 'labels', str(flag)),
            samples_dir= in_dir,
            output_dir= path.join(out_dir, 'images', str(flag)),
            flag= flag
        )
