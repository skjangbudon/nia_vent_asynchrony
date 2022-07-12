import os
import yaml
import numpy as np
import pandas as pd

def load_data(filename, label_file=False):
    # file_extension = filename.split('.')[-1]
    X = pd.read_csv(filename, low_memory=False, nrows=None)
    X = X.loc[:,~X.columns.isin(['Unnamed: 0'])]
    print(filename, X.shape)
    return X

def load_yaml(fname):
    with open(fname, 'r') as fp:
        data = yaml.safe_load(fp)
    return data

def load_and_stack_data(data_files: list, label_file=False):
    if isinstance(data_files, str):
        data_files = [data_files]
    if len(data_files)>1:
        df_list = []
        for fi in data_files:
            if os.path.exists(fi):
                x_df = load_data(fi, label_file)
                df_list.append(x_df)
                colnames = x_df
        data_df = pd.concat(df_list)
    else:
        data_df = load_data(data_files[0], label_file)
    print(data_df.shape)
    return data_df