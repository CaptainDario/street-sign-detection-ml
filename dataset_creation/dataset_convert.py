import sys
from os import path, makedirs, listdir
from shutil import copy2
import math

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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


def split_number_into_3_weighted_parts(number, train_percentage, val_percentage, test_percentage):
    ''' takes the upper bound for the training and validation 
        and the lower bound for the testing part. 
        returns a list indicating the number of elements
        for training, validation and testing
    '''
    train_nr = math.ceil(number * train_percentage)
    val_nr = math.ceil(number * val_percentage)
    test_nr = math.floor(number * test_percentage)
    return [train_nr, val_nr, test_nr]


def distribute_samples_for_labels(df_samples_multilabel, df_samples_singlelabel, df_label_count):
    # shuffle label_count
    df_label_count = shuffle(df_label_count)

    # shuffle multilabel
    df_samples_multilabel = shuffle(df_samples_multilabel)

    # shuffle singlelabel
    df_samples_singlelabel = shuffle(df_samples_singlelabel)
    
    # iterate over all labels
    for index_label_count, row_label_count in df_label_count.iterrows():

        train_nr, val_nr, test_nr = split_number_into_3_weighted_parts(
            row_label_count['label_count'],
            0.7,
            0.2,
            0.1
        )
        print('label ', row_label_count['label'], '; label_count ', row_label_count['label_count'], '; train ', train_nr, '; val ', val_nr, '; test ', test_nr )
        
        # flag multilabel for training 
        print('\t training --- multilabel: ',row_label_count['label'])
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        print('\t\t', df_current_multilabel.shape)
        for index, row_current_label in df_current_multilabel.iterrows():
            # count train flag occurrences:
            df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
            el_w_train_flag = df_current_multilabel['flag']=='train'
            el_w_train_flag_sum = el_w_train_flag.sum()
            # break if the sum of train flags exceeds the number of training elements
            if(el_w_train_flag_sum>=train_nr):
                break
            # set flag for training
            df_samples_multilabel.at[index, 'flag'] = 'train'
            # set flag for all elements with same sample
            rows_2 = df_samples_multilabel[df_samples_multilabel['sample']== row_current_label['sample']]
            for index_2, row_2 in rows_2.iterrows():
                df_samples_multilabel.at[index_2, 'flag'] = 'train'
        
        
        # flag singlelabel for training
        print('\t training --- singlelabel: ', row_label_count['label'])
        df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']== row_label_count['label']]
        print('\t\t', df_current_singlelabel.shape)
        # occurrences of multilabel elements with train flag
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        el_w_train_flag_multilabel = df_current_multilabel['flag']=='train'
        el_w_train_flag_multilabel_sum = el_w_train_flag_multilabel.sum()
        print('\t\t mutlilabel_trainflag_sum: ', el_w_train_flag_multilabel_sum)

        for index, row_current_label in df_current_singlelabel.iterrows():
            df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']==row_label_count['label']]
            el_w_train_flag_singlelabel = df_current_singlelabel['flag']=='train'
            el_w_train_flag_singlelabel_sum = el_w_train_flag_singlelabel.sum()
            el_w_train_flag_sum = el_w_train_flag_multilabel_sum + el_w_train_flag_singlelabel_sum
            if(el_w_train_flag_sum>=train_nr):
                print('\t\t\t el_w_train_flag_sum', el_w_train_flag_sum)
                break
            print('\t\t\t labelling for singlelabel')
            # set flag for training
            df_samples_singlelabel.at[index, 'flag'] = 'train'


        # flag multilabel for validation
        print('\t validation --- multilabel: ',row_label_count['label'])
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        df_current_multilabel_not_training = df_current_multilabel[df_current_multilabel['flag'] != 'train']
        print('\t\t', df_current_multilabel_not_training.shape)
        for index, row_current_label in df_current_multilabel_not_training.iterrows():
            # count val flag occurrences:
            df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
            # TODO: is that actually needed?
            df_current_multilabel_not_training = df_current_multilabel[df_current_multilabel['flag'] != 'train']
            el_w_val_flag = df_current_multilabel_not_training['flag']=='val'
            el_w_val_flag_sum = el_w_val_flag.sum()
            # break if the sum of train flags exceeds the number of validation elements
            if(el_w_val_flag_sum>=val_nr):
                break
            # set flag for validation
            df_samples_multilabel.at[index, 'flag'] = 'val'
            # set flag for all elements with same sample
            rows_2 = df_samples_multilabel[df_samples_multilabel['sample']== row_current_label['sample']]
            for index_2, row_2 in rows_2.iterrows():
                df_samples_multilabel.at[index_2, 'flag'] = 'val'
            
        # flag singlelabel for validation
        print('\t validation --- singlelabel: ', row_label_count['label'])
        df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']== row_label_count['label']]
        df_current_singlelabel_not_training = df_current_singlelabel[df_current_singlelabel['flag'] != 'train']
        print('\t\t', df_current_singlelabel_not_training.shape)
        # occurrences of singlelable elements with validation flag
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        # TODO: is that actually needed?
        df_current_multilabel_not_training = df_current_multilabel[df_current_multilabel['flag'] != 'train']
        el_w_val_flag_multilabel = df_current_multilabel_not_training['flag']=='val'
        el_w_val_flag_multilabel_sum = el_w_val_flag_multilabel.sum()
        print('\t\t mutlilabel_valflag_sum: ', el_w_val_flag_multilabel_sum)

        for index, row_current_label in df_current_singlelabel_not_training.iterrows():
            df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']== row_label_count['label']]
            df_current_singlelabel_not_training = df_current_singlelabel[df_current_singlelabel['flag'] != 'train']
            el_w_val_flag_singlelabel = df_current_singlelabel_not_training['flag']=='val'
            el_w_val_flag_singlelabel_sum = el_w_val_flag_singlelabel.sum()
            el_w_val_flag_sum = el_w_val_flag_multilabel_sum + el_w_val_flag_singlelabel_sum
            if(el_w_val_flag_sum>=val_nr):
                print('\t\t\t el_w_val_flag_sum', el_w_val_flag_sum)
                break
            print('\t\t\t labelling for singlelabel')
            # set flag for training
            df_samples_singlelabel.at[index, 'flag'] = 'val'

        # flag multilabel for testing
        print('\t testing --- multilabel: ',row_label_count['label'])
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        df_current_multilabel_testing = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'val')]
        print('\t\t', df_current_multilabel_testing.shape)
        for index, row_current_label in df_current_multilabel_testing.iterrows():
            # count val flag occurrences:
            df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
            # TODO: is that actually needed?
            df_current_multilabel_testing = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'val')]
            el_w_test_flag = df_current_multilabel_testing['flag']=='test'
            el_w_test_flag_sum = el_w_test_flag.sum()
            # break if the sum of train flags exceeds the number of validation elements
            if(el_w_test_flag_sum>=test_nr):
                break
            # set flag for validation
            df_samples_multilabel.at[index, 'flag'] = 'test'
            # set flag for all elements with same sample
            rows_2 = df_samples_multilabel[df_samples_multilabel['sample']== row_current_label['sample']]
            for index_2, row_2 in rows_2.iterrows():
                df_samples_multilabel.at[index_2, 'flag'] = 'test'
            
        # flag singlelabel for testing
        print('\t testing --- singlelabel: ', row_label_count['label'])
        df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']== row_label_count['label']]
        df_current_singlelabel_testing = df_current_singlelabel[(df_current_singlelabel['flag'] != 'train') & (df_current_singlelabel['flag'] != 'val')]
        print('\t\t', df_current_singlelabel_testing.shape)
        # occurrences of singlelable elements with validation flag
        df_current_multilabel = df_samples_multilabel[df_samples_multilabel['label']==row_label_count['label']]
        # TODO: is that actually needed?
        df_current_multilabel_testing = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'val')]
        el_w_test_flag_multilabel = df_current_multilabel_testing['flag']=='test'
        el_w_test_flag_multilabel_sum = el_w_test_flag_multilabel.sum()
        print('\t\t mutlilabel_testflag_sum: ', el_w_test_flag_multilabel_sum)

        for index, row_current_label in df_current_singlelabel_testing.iterrows():
            df_current_singlelabel = df_samples_singlelabel[df_samples_singlelabel['label']== row_label_count['label']]
            df_current_singlelabel_testing = df_current_singlelabel[(df_current_singlelabel['flag'] != 'train') & (df_current_singlelabel['flag'] != 'val')]
            el_w_test_flag_singlelabel = df_current_singlelabel_testing['flag']=='test'
            el_w_test_flag_singlelabel_sum = el_w_test_flag_singlelabel.sum()
            el_w_test_flag_sum = el_w_test_flag_multilabel_sum + el_w_test_flag_singlelabel_sum
            if(el_w_test_flag_sum>=test_nr):
                print('\t\t\t el_w_test_flag_sum', el_w_test_flag_sum)
                break
            print('\t\t\t labelling for singlelabel')
            # set flag for training
            df_samples_singlelabel.at[index, 'flag'] = 'test'



'''WHERE AM I: 
    for functionality:
        [x] get constraints for nr of train, val samples
        [x] break only out of current_labels loop
        [x] implement for singlelabel afterwards 
        [x] implement for validation
        [x] implement for test
        [x] update method signature
        [x] shuffle before
    for readability:
        * source out checks, make agnostic
        * remove layers of nested loops
        * make agnostic: apply separately for train/val
    for further implementations:
        * calculate distribution
        * rerun if distribution exceeds certain threshold
'''



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
    pixels_x, pixels_y = 1360, 800
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
