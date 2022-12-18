import os.path

import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def parse_args():
    if len(sys.argv) == 5:
        return sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    elif len(sys.argv) == 3:
        return sys.argv[1], sys.argv[2], 1360, 800
    else:
        print("Usage: python dataset_convert.py <input_file> <output_directory> [<pixels_x> <pixels_y>]")
        exit(1)


def convert(df: pd.DataFrame) -> pd.DataFrame:
    ''' converts coordinates and returns them serialized into columns
    '''
    # add converted coordinates column
    print("\tConverting coordinates")
    coords_tuples = df.apply(lambda row: convert_coordinates((row[1], row[2]), (row[3], row[4])), axis=1)
    coords_tuples = coords_tuples.apply(pd.Series)
    df.iloc[:, 1:5] = coords_tuples
    return df


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
    # write each row into a separate file
    print("\tWriting data to files")
    os.makedirs(out_dir, exist_ok=True)
    for index, row in df.iterrows():
        file_name = row[0].replace(".ppm", ".txt")
        file_path = os.path.join(out_dir, file_name)
        # append additional labels to existing file
        if (os.path.exists(file_path)):
            with open(file_path, 'a') as f:
                f.write(f"\n{row[5]} {row[1]} {row[2]} {row[3]} {row[4]}")
        else:
            with open(file_path, 'w') as f:
                f.write(f"{row[5]} {row[1]} {row[2]} {row[3]} {row[4]}")
        # print(f"File {file_name} written.")


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


def annotate_sample(sample_path: str, annotation_path: str):
    """ Loads a sample from `sample_path` and an annotation from `annotation_path`.
        Then draws the BBoxs from the annotations on the image and saves the image too 
    """
    with open(annotation_path, encoding="utf8", mode="r") as f:
        bboxs = [[float(__a) for __a in _a.split(" ")] for _a in f.read().split("\n")]

    im = Image.open(sample_path)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))

    # Display the image
    ax.imshow(im)

    # label, X, Y, Width, and Height
    for b in bboxs:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (b[1] * im.width - b[3] * im.width / 2, b[2] * im.height - b[4] * im.height / 2), b[3] * im.width,
                                                                                              b[4] * im.height,
            linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig("sample.png")


if __name__ == '__main__':
    in_file, out_dir, pixels_x, pixels_y = parse_args()
    df = read_data(in_file)
    convert(df)
    write_data(df, out_dir)
    # annotate_sample(sample_path, annotation_path)
