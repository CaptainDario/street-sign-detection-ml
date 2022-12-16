import pandas as pd
import sys


def parse_args():
    if len(sys.argv) == 5:
        return sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    elif len(sys.argv) == 3:
        return sys.argv[1], sys.argv[2], 1360, 800
    else:
        print("Usage: python dataset_convert.py <input_file> <output_file> [<pixels_x> <pixels_y>]")
        exit(1)


def convert(df):
    # add converted coordinates column
    print("Converting coordinates:")
    print(df)
    coords_tuples = df.apply(lambda row: convert_coordinates((row[1], row[2]), (row[3], row[4])), axis=1)
    coords_tuples = coords_tuples.apply(pd.Series)
    df.iloc[:, 1:5] = coords_tuples
    # move last column to first
    df.iloc[:, [0, 1, 2, 3, 4, 5]] = df.iloc[:, [5, 0, 1, 2, 3, 4]]
    print('\nConverted coordinates:')
    print(df)
    return df


def read_data():
    # import data
    df = pd.read_csv(in_file, sep=';', header=None)
    return df


def write_data(df):
    # write data to file
    df.to_csv(out_file, sep=' ', header=None, index=None)


def convert_coordinates(top_left, bottom_right):
    # convert coordinates to center coordinate and width and height of bounding box from pixels to percentage
    x_center = (top_left[0] + bottom_right[0]) / 2 / pixels_x
    y_center = (top_left[1] + bottom_right[1]) / 2 / pixels_y
    width = (bottom_right[0] - top_left[0]) / pixels_x
    height = (bottom_right[1] - top_left[1]) / pixels_y
    return x_center, y_center, width, height


if __name__ == '__main__':
    in_file, out_file, pixels_x, pixels_y = parse_args()
    df = read_data()
    convert(df)
    write_data(df)
