import math

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def distribute_samples_flow(
    df: pd.DataFrame, 
    train_percentage: float, 
    val_percentage: float, 
    test_percentage: float
) -> list:
    ''' distributes samples for given percentages for 
        training, validation, testing as much as possible
    '''
    print('\tDistributing samples for each label')
    df = rename_df(df)
    df_label_count = get_label_count(df)
    df_multilabel_samples = get_multilabel_samples(df)
    df_singlelabel_samples = get_singlelabel_samples(df)
    df = distribute_samples_for_labels(
        df_samples_multilabel= df_multilabel_samples, 
        df_samples_singlelabel= df_singlelabel_samples, 
        df_label_count= df_label_count, 
        train_percentage= train_percentage, 
        val_percentage= val_percentage, 
        test_percentage= test_percentage
    )
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
    ''' The number will be splitted with their respective percentage and lower bounded. 
        The difference to the total number will then be then weighted and lower bounded again
        and added before being added to the parts. The remaining part will be added to the 
        training-part.

        returns a list indicating the number of elements
        for training, validation and testing
    '''
    train_nr = math.floor(number * train_percentage)
    val_nr = math.floor(number * val_percentage)
    test_nr = math.floor(number * test_percentage)
    difference = number - (train_nr + val_nr + test_nr)
    train_additional = math.floor(difference * train_percentage)
    val_additional = math.floor(difference * val_percentage)
    test_additional = math.floor(difference * test_percentage)
    difference_2 = number - (train_nr + val_nr + test_nr + train_additional + val_additional + test_additional)
    train_nr += train_additional + difference_2
    val_nr += val_additional
    test_nr += test_additional
    return [train_nr, val_nr, test_nr]


def distribute_samples_for_labels(df_samples_multilabel, df_samples_singlelabel, df_label_count, train_percentage, val_percentage, test_percentage):
    # flag column for multilabel, singlelabel
    df_samples_multilabel['flag'] = np.nan
    df_samples_singlelabel['flag'] = np.nan

    # shuffle label_count, multilabel, singlelabel
    df_label_count = shuffle(df_label_count)
    df_samples_multilabel = shuffle(df_samples_multilabel)
    df_samples_singlelabel = shuffle(df_samples_singlelabel)

    # iterate over all labels
    for index_label_count, row_label_count in df_label_count.iterrows():
        current_label = row_label_count['label']
        train_nr, val_nr, test_nr = split_number_into_3_weighted_parts(
            number= row_label_count['label_count'],
            train_percentage= train_percentage,
            val_percentage= val_percentage,
            test_percentage= test_percentage
        )
        
        # flag multilabel for training 
        df_samples_multilabel = flag_multilabel_dataframe(
            dataframe= df_samples_multilabel,
            flag= 'train',
            label= current_label,
            max_nr= train_nr
        )
        
        # flag singlelabel for training
        df_samples_singlelabel = flag_singlelabel_dataframe(
            dataframe_singlelabel= df_samples_singlelabel,
            dataframe_multilabel= df_samples_multilabel,
            flag= 'train',
            label= current_label,
            max_nr= train_nr
        )

        # flag multilabel for validation
        df_samples_multilabel = flag_multilabel_dataframe(
            dataframe= df_samples_multilabel,
            flag='val',
            label= current_label,
            max_nr= val_nr
        )
        
        # flag singlelabel for validation
        df_samples_singlelabel = flag_singlelabel_dataframe(
            dataframe_singlelabel= df_samples_singlelabel,
            dataframe_multilabel= df_samples_multilabel,
            flag= 'val',
            label= current_label,
            max_nr= val_nr
        )
        
        # flag multilabel for testing
        df_samples_multilabel = flag_multilabel_dataframe(
            dataframe= df_samples_multilabel,
            flag='test',
            label= current_label,
            max_nr= test_nr
        )
        
        # flag singlelabel for testing
        df_samples_singlelabel = flag_singlelabel_dataframe(
            dataframe_singlelabel= df_samples_singlelabel,
            dataframe_multilabel= df_samples_multilabel,
            flag= 'test',
            label= current_label,
            max_nr= test_nr
        )
    # concat multilabel and singlelabel df
    df_concat = pd.concat([
        df_samples_multilabel,
        df_samples_singlelabel
    ])
    # distribution per label
    for index_label_count, row_label_count in df_label_count.iterrows():
        train_percentage, val_percentage, test_percentage = get_flag_distribution_in_dataframe(df_concat, row_label_count['label'])
        print(f'\t\t label: {row_label_count["label"]}, elements: {row_label_count["label_count"]}, percentages: train: {train_percentage}, val: {val_percentage}, test: {test_percentage}')
    # overall distribution 
    overall_train_percentage, overall_val_percentage, overall_test_percentage = get_flag_distribution_in_dataframe(df_concat)
    print(f'\t\t---overall distribution: train: {overall_train_percentage}, val: {overall_val_percentage}, test: {overall_test_percentage}')
    return df_concat.sort_index()


def flag_multilabel_dataframe(dataframe, flag, label, max_nr):
    df_current_multilabel = dataframe[dataframe['label']==label]
    # filter out other samples
    if(flag == 'val'):
        df_current_multilabel = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'test')]
    if(flag == 'test'):
        df_current_multilabel = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'val')]
    for index, row_current_label in df_current_multilabel.iterrows():
        # count flag occurrences:
        if(flag =='train'):
            el_w_flag_sum = get_flag_count_in_dataframe(
                dataframe= dataframe,
                flag= str(flag),
                label= label
            )
        elif(flag == 'val'):
            df_current_multilabel = dataframe[dataframe['label']==label]
            df_current_multilabel = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'test')]
            el_w_flag_sum = get_flag_count_in_dataframe(
                dataframe= df_current_multilabel,
                flag= str(flag),
                label= label
            )
        elif(flag == 'test'):
            df_current_multilabel = dataframe[dataframe['label']==label]
            df_current_multilabel = df_current_multilabel[(df_current_multilabel['flag'] != 'train') & (df_current_multilabel['flag'] != 'val')]
            el_w_flag_sum = get_flag_count_in_dataframe(
                dataframe= df_current_multilabel,
                flag= str(flag),
                label= label
            )
        # break if the sum of flags exceeds the number of max defined elements
        if(el_w_flag_sum>=max_nr):
            break
        # set flag
        dataframe.at[index, 'flag'] = str(flag)
        # set flag for all elements with same sample
        rows_2 = dataframe[dataframe['sample']== row_current_label['sample']]
        for index_2, row_2 in rows_2.iterrows():
            dataframe.at[index_2, 'flag'] = str(flag)
    return dataframe


def flag_singlelabel_dataframe(dataframe_singlelabel, dataframe_multilabel, flag, label, max_nr):
    df_current_singlelabel = dataframe_singlelabel[dataframe_singlelabel['label']== label]
    if (flag == 'val'):
        df_current_singlelabel = df_current_singlelabel[df_current_singlelabel['flag'] != 'train']
    elif (flag == 'test'):
        df_current_singlelabel = df_current_singlelabel[(df_current_singlelabel['flag'] != 'train') & (df_current_singlelabel['flag'] != 'val')]
    # occurrences of multilabel elements
    el_w_flag_multilabel_sum = get_flag_count_in_dataframe(
        dataframe= dataframe_multilabel,
        flag= str(flag),
        label= label
    )

    for index, row_current_label in df_current_singlelabel.iterrows():
        if (flag == 'train'):
            el_w_flag_singlelabel_sum = get_flag_count_in_dataframe(
                dataframe= dataframe_singlelabel,
                flag= str(flag),
                label= label
            )
        elif (flag == 'val'):
            df_current_singlelabel = dataframe_singlelabel[dataframe_singlelabel['label']== label]
            df_current_singlelabel = df_current_singlelabel[df_current_singlelabel['flag'] != 'train']
            el_w_flag_singlelabel_sum = get_flag_count_in_dataframe(
                dataframe= df_current_singlelabel,
                flag= str(flag),
                label= label
            )
        elif (flag == 'test'):
            df_current_singlelabel = dataframe_singlelabel[dataframe_singlelabel['label']== label]
            df_current_singlelabel_testing = df_current_singlelabel[(df_current_singlelabel['flag'] != 'train') & (df_current_singlelabel['flag'] != 'val')]
            el_w_flag_singlelabel_sum = get_flag_count_in_dataframe(
                dataframe= df_current_singlelabel_testing,
                flag= str(flag),
                label= label
            )
        el_w_flag_sum = el_w_flag_multilabel_sum + el_w_flag_singlelabel_sum
        if(el_w_flag_sum>=max_nr):
            break
        # set flag
        dataframe_singlelabel.at[index, 'flag'] = str(flag)
    return dataframe_singlelabel


def get_flag_count_in_dataframe(dataframe, flag, label= None):
    if label is not None: 
        dataframe = dataframe[dataframe['label']==label]
    return (dataframe['flag']==flag).sum()


def get_flag_distribution_in_dataframe(dataframe, label= None):
    if label is not None: 
        dataframe = dataframe[dataframe['label']==label]
    train_count = get_flag_count_in_dataframe(dataframe, 'train', label)
    val_count = get_flag_count_in_dataframe(dataframe, 'val', label)
    test_count = get_flag_count_in_dataframe(dataframe, 'test', label)
    train_percentage = train_count / len(dataframe) 
    val_percentage = val_count / len(dataframe) 
    test_percentage = test_count / len(dataframe) 
    return train_percentage, val_percentage, test_percentage
