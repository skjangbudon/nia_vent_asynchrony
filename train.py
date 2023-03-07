import json
import time
import os
import glob
import os.path as osp
import multiprocessing
import argparse
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torchsummary import summary

import module.utils as cutils
from module.datasets import annotation_dict_to_dataframe, preprocess_label_file, annotate
from module.datasets import get_wav_firsttime, annotate_each_label, annotate_one_instance_file, preprocess_data
from model.AsynchModel import AsynchModel
from module.loss import calculate_ce_loss

def main():
    parser = argparse.ArgumentParser(description='train asynchrony model')
    parser.add_argument('--config', default='/VOLUME/nia_vent_asynchrony/config/train_config.yml', help='config file path')
    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    sample_num = config['sample_num'] if 'sample_num' in config else None

    nowDate = cutils.get_today_string()
    if os.path.exists(config['load_feature']):
        RESULT_PATH = os.path.dirname(config['load_feature'])
    else:
        RESULT_PATH = osp.join(config['result_dir'], nowDate)
        os.makedirs(RESULT_PATH, exist_ok=True)

    logger = cutils.set_logger('train', path=RESULT_PATH)
    logger.info(config)
    logger.info('RESULT_PATH:'+ RESULT_PATH)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['CUDA_VISIBLE_DEVICES'])
    logger.info('Current cuda device:'+str(torch.cuda.current_device()))
    
    if os.path.exists(config['load_feature']):
        feature_df = pd.read_pickle(config['load_feature'])
    else:
        # load label files
        logger.info('find label files')
        label_path_list = []
        for pat_range in config['train_id']+config['validation_id']:
            path_regex = osp.join(config['label_path'], pat_range+'.json')
            files = glob.glob(path_regex)
            logger.info(path_regex+' '+ str(len(files))+ ' files')
            label_path_list.extend(files)


        # 어노테이션 파일 모두 읽기
        logger.info('preprocess '+ str(len(label_path_list))+ ' label files')
        label_list = []
        for label_path in label_path_list:
            res = preprocess_label_file(label_path)
            label_list.append(res)
        ann_df = pd.concat(label_list)
        # 159 label files, single threads, elapsed 53 secs

        # 어노테이션 파일 정제
        ann_df = annotate(ann_df)
        ann_df = ann_df.rename(columns={'startTime':'ann_startTime_float', 'endTime':'ann_endTime_float'})
        # columns: ['ann_startTime', 'ann_endTime', 'duration', 'pat_id', 'wav_number','label_annotation']

        logger.info(str(len(ann_df[['hospital_id_patient_id']].drop_duplicates()))+' patients in label file')
        logger.info(str(len(ann_df[['hospital_id_patient_id','wav_number']].drop_duplicates()))+' wav in label file')
        logger.info(str(len(ann_df))+ ' annotations in label file')


        # feature 에 라벨 달기
        feature_df_list = []
        for data_path in config['data_path']:
            feature_df = annotate_one_instance_file(data_path, ann_df)
            feature_df_list.append(feature_df)
        feature_df = pd.concat(feature_df_list)

        # train, validation 나누기
        train_idx = [feature_df['hospital_id_patient_id'].str.match(pat.replace('*','[0-9]+')) for pat in config['train_id']]
        train_idx = pd.concat(train_idx, axis=1).sum(axis=1)>0
        validation_idx = [feature_df['hospital_id_patient_id'].str.match(pat.replace('*','[0-9]+')) for pat in config['validation_id']]
        validation_idx = pd.concat(validation_idx, axis=1).sum(axis=1)>0


        feature_df['split'] = ''
        feature_df.loc[train_idx,'split'] = 'train'
        feature_df.loc[validation_idx,'split'] = 'valid'

        if sample_num is not None:    
            feature_df = pd.concat([feature_df.query('split=="train"').query('label==0').sample(n=sample_num['train'][0]),
                                    feature_df.query('split=="train"').query('label==1').sample(n=sample_num['train'][1]),
                                    feature_df.query('split=="train"').query('label==2').sample(n=sample_num['train'][2]),
                                    feature_df.query('split=="valid"').query('label==0').sample(n=sample_num['validation'][0]),
                                    feature_df.query('split=="valid"').query('label==1').sample(n=sample_num['validation'][1]),
                                    feature_df.query('split=="valid"').query('label==2').sample(n=sample_num['validation'][2])
                                    ])

        logger.info(feature_df['split'].value_counts())
        logger.info('total'+ str(len(feature_df)))
        logger.info('no. hospital_id_patient_id by set')
        logger.info(feature_df.groupby(['split'])['hospital_id_patient_id'].nunique())


        logger.info('label statistics')
        logger.info(feature_df['label'].value_counts())
        logger.info(feature_df['label'].value_counts(normalize=True))
        logger.info('--train')
        logger.info(feature_df[feature_df['split']=='train']['label'].value_counts())
        logger.info(feature_df[feature_df['split']=='train']['label'].value_counts(normalize=True))
        logger.info('--valid')
        logger.info(feature_df[feature_df['split']=='valid']['label'].value_counts())
        logger.info(feature_df[feature_df['split']=='valid']['label'].value_counts(normalize=True))


        labelfreq = pd.concat([feature_df.groupby('split')['label'].value_counts(),
        feature_df.groupby('split')['label'].value_counts(normalize=True)], axis=1)
        labelfreq.columns = ['num', 'ratio']
        logger.info(labelfreq)
        labelfreq.to_csv(osp.join(RESULT_PATH, 'label_freq_trainval.csv'))
        
        if config['write_feature']:    
            feature_df.to_pickle(osp.join(RESULT_PATH, 'feature_df.pkl'))
    
    train_dataloader, val_dataloader, _ = preprocess_data(feature_df, set_train_weightedsampler=True, scale=True, target=None)
    # 89934 instance, 20 threads, elapsed 1 min
    
    # import pickle
    # def save_pickle(fname, data):
    #     with open(fname, 'wb') as f:
    #         pickle.dump(data, f)

    # def load_pickle(fname):
    #     with open(fname, 'rb') as f:
    #         data = pickle.load(f)
    #     return data

    # save_pickle(osp.join(RESULT_PATH, 'train_val_dataloader.pkl'), [train_dataloader, val_dataloader])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    ventdys_model = AsynchModel(input_dim=2, padding_mode='replicate', num_class=feature_df['label'].nunique()).to(device)

    logger.info(summary(ventdys_model, input_size=(2, 3600), device=str(device)))

    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(ventdys_model.parameters(), lr=learning_rate)
    early_stopping = cutils.EarlyStopping(patience = 5, verbose = True, path = RESULT_PATH, checkpoint_name='ventdys_model.pt', logger=logger)


    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []


    for epoch in range(n_epochs):
        logger.info(f'Epoch {epoch}')

        ventdys_model.train()
        for i_step, data in enumerate(train_dataloader):
            xi, target = data
            xi = xi.to(device)
            target = target.to(device).squeeze(-1).squeeze(-1).long()
            out = ventdys_model(xi)
            loss = calculate_ce_loss(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if i_step%100==0: logger.info(f'epoch {epoch} step {i_step}: loss {loss.item()}')
            # logger.info('step', i_step, 'loss', loss.item())
        logger.info(f'epoch {epoch} loss {loss.item()}')
        
        ######################    
        # validate the model #
        ######################
        ventdys_model.eval() 

        for i_step, data in enumerate(val_dataloader) :
            xi, target = data
            xi = xi.to(device)
            batch_size = len(xi)
            target = target.to(device).squeeze(-1).squeeze(-1).long()
            out = ventdys_model(xi)

            loss = calculate_ce_loss(out, target)
            
            valid_losses.append(loss.item())


        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
            
        epoch_len = len(str(n_epochs))


        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')

        logger.info(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, ventdys_model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

if __name__ == "__main__":
    main()



