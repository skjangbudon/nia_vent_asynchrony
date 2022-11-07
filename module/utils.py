import os
import yaml
import json
import os.path as osp
import datetime
import logging

import numpy as np
import pandas as pd
import torch

def get_today_string(include_time=True):
    if include_time:
        return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    else:   
        return datetime.datetime.now().strftime('%Y-%m-%d')

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

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
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

def print_environment():
    import os
    import torch

    # 1. cpu
    print(os.system('lscpu | grep "Architecture"'))
    print(os.system('lscpu | grep "Model name"'))
    print(os.system('lscpu | grep "^CPU(s):"'))

    # 2. gpu
    print(os.system('nvidia-smi'))

    # 3. RAM
    print(os.system('free -g'))

    # 4.HDD
    print(os.system('df -h'))

    # 5. OS
    print(os.system('lsb_release -a'))

    # 6. Pytorch
    print(torch.__version__)


def get_logger(name, level=logging.DEBUG, resetlogfile=False, path='log'):
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    fname = os.path.join(path, name+'.log')
    os.makedirs(path, exist_ok=True) 
    if resetlogfile :
        if os.path.exists(fname):
            os.remove(fname) 
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_logger(tag, path='.'):
    logger = get_logger(f'{tag}', resetlogfile=True, path=path)
    logger.setLevel(logging.INFO)
    return logger

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='/', checkpoint_name='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.checkpoint_name = checkpoint_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), osp.join(self.path, self.checkpoint_name))
        self.val_loss_min = val_loss
