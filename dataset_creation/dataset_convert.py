import sys
from os import path, makedirs, listdir
from shutil import copy2

import pandas as pd

import utils


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

    for i in ["train", "test", "val"]:
        for j in ["images", "labels"]:
            makedirs(path.join(dataset_path, i, j), exist_ok=True)


def copy_relevant_pictures(coordinate_dir: str, samples_dir: str, output_dir: str):
    ''' copies all pictures that have coordiantes into the specified `output_dir`,
        the location of the coordiante-pictures is based on the specified `coordiante_dir`
        and the pictures to be copies are taken from the specified `samples_dir`
    '''
    coordinate_elements = listdir(coordinate_dir)
    for index in range(len(coordinate_elements)):
        utils.print_progress_bar(
            iteration= index+1,
            total= len(coordinate_elements),
            prefix= '\tCopied Files:',
            suffix= 'Complete',
            length= 30
        )
        sample_path = path.join(
            samples_dir,
            coordinate_elements[index].replace('.txt', '.ppm')
        )
        copy2(
            src= sample_path,
            dst= output_dir
        )


def read_data(in_file: str) -> pd.DataFrame:
    ''' reads data frome a relative or absolute path as a String,
        separates data with ';' \n
        returns dataframe
    '''
    print('\tReading data from input file')
    df = pd.read_csv(in_file, sep=';', header=None)
    return df


def rename_df(df):
    return df.rename(columns= {
        0: 'sample',
        1: 'x_center',
        2: 'y_center',
        3: 'width',
        4: 'height',
        5: 'label'
    })


def get_label_count(df: pd.DataFrame) -> pd.DataFrame:
    ''' returns the count of the labels in the input DataFrame
    '''
    df_label_count = pd.DataFrame(columns=['label', 'label_count'])
    df_label_count.label = df.label.value_counts().sort_index().index
    df_label_count.label_count = df.label.value_counts().sort_index().values
    return df_label_count


def get_singlelabel_samples(df):
    df_singlelabel = pd.DataFrame(columns=['sample', 'x_center', 'y_center', 'width', 'height', 'label'])
    for index,row in df.iterrows():
        if(len(df[df['sample']== row['sample']]) == 1):
            df_singlelabel.loc[index] = row
    return df_singlelabel


def get_multilabel_samples(df):
    df_multilabel = pd.DataFrame(columns=['sample', 'x_center', 'y_center', 'width', 'height', 'label'])
    for index,row in df.iterrows():
        if(len(df[df['sample']== row['sample']]) > 1):
            df_multilabel.loc[index] = row
    return df_multilabel


def write_data(df: pd.DataFrame, out_dir: str):
    ''' writes out data of each picture into a separate file,
        appends multiple labels of the same picture into the 
        same output file using newline as a delimiter
    '''
    print("\tWriting data to files")
    makedirs(out_dir, exist_ok=True)
    for index, row in df.iterrows():
        utils.print_progress_bar(
            iteration= index+1,
            total= len(df),
            prefix= '\tWritten Files:',
            suffix= 'Complete',
            length= 30
        )
        file_name = row[0].replace(".ppm", ".txt")
        file_path = path.join(out_dir, "train", "labels", file_name)
        # append additional labels to existing file
        if (path.exists(file_path)):
            with open(file_path, 'a') as f:
                f.write(f"\n{row[5]} {row[1]} {row[2]} {row[3]} {row[4]}")
        else:
            with open(file_path, 'w') as f:
                f.write(f"{row[5]} {row[1]} {row[2]} {row[3]} {row[4]}")


def convert_coordinates(top_left: list, bottom_right: list) -> list:
    ''' top_left: [leftmost image: float, upmost image: float]\n
        bottom_right: [rightmost image: float, downmost image]\n
        returns: [x_center: float, y_center: float, width: float, height: float]

        conversion of given coordinates to center coordinates for x,y and with, height
        of the bounding box with normalization to [0,1] to a percentage value.
    '''
    x_center = (top_left[0] + bottom_right[0]) / 2 / pixels_x
    y_center = (top_left[1] + bottom_right[1]) / 2 / pixels_y
    width = (bottom_right[0] - top_left[0]) / pixels_x
    height = (bottom_right[1] - top_left[1]) / pixels_y
    return x_center, y_center, width, height


def parse_args():
    if len(sys.argv) == 5:
        return sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    elif len(sys.argv) == 3:
        return sys.argv[1], sys.argv[2], 1360, 800
    else:
        print("Usage: python dataset_convert.py <input_directory> <output_directory> [<pixels_x> <pixels_y>]")
        exit(1)


if __name__ == '__main__':
    in_dir, out_dir, pixels_x, pixels_y = parse_args()
    label_file = path.join(in_dir, "gt.txt")

    create_dataset_folder(out_dir)
    df = read_data(label_file)
    convert(df)
    write_data(df, out_dir)
