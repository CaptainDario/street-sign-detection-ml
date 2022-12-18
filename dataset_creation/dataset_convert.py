import os.path

import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List


def parse_args():
    if len(sys.argv) == 5:
        return sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    elif len(sys.argv) == 3:
        return sys.argv[1], sys.argv[2], 1360, 800
    else:
        print("Usage: python dataset_convert.py <input_file> <output_directory> [<pixels_x> <pixels_y>]")
        exit(1)


def convert(df):
    # add converted coordinates column
    print("Converting coordinates:")
    print(df)
    coords_tuples = df.apply(lambda row: convert_coordinates((row[1], row[2]), (row[3], row[4])), axis=1)
    coords_tuples = coords_tuples.apply(pd.Series)
    df.iloc[:, 1:5] = coords_tuples
    print('\nConverted coordinates:')
    print(df)
    return df


def read_data():
    # import data
    df = pd.read_csv(in_file, sep=';', header=None)
    return df


def write_data(df):
    # write each row into a separate file
    print("Writing data to files:")
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
        print(f"File {file_name} written.")
    #df.to_csv(out_file, sep=' ', header=None, index=None)


def convert_coordinates(top_left, bottom_right):
    # convert coordinates to center coordinate and width and height of bounding box from pixels to percentage
    x_center = (top_left[0] + bottom_right[0]) / 2 / pixels_x
    y_center = (top_left[1] + bottom_right[1]) / 2 / pixels_y
    width = (bottom_right[0] - top_left[0]) / pixels_x
    height = (bottom_right[1] - top_left[1]) / pixels_y
    return x_center, y_center, width, height


def annotate_sample(sample_path: str, annotation_path: str):
    """ Loads a sample from `sample_path` and an annotation from `annotation_path`.
    Than draws the BBoxs from the annotations on the image and save the image to 
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
    df = read_data()
    convert(df)
    write_data(df)
    # annotate_sample(sample_path, annotation_path)
