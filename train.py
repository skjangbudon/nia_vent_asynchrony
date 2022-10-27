

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

os.chdir('/VOLUME/nia_vent_asynchrony')

import module.utils as cutils


def annotation_dict_to_dataframe(row):
    '''
    input : pd.Series(['1-206', '1-206-013.csv',13, '[{"startTime": 424.00848, "endTime": 431.99197300000003, "duration": 7.98349300000001, "extra": {"value": "noise", "label": "Noise"}}, ...')
    output : 2 rows, pd.DataFrame(columns=['startTime', 'endTime', 'duration', 'extra','pat_id','wav_number'], ... ) 
    '''
    df = pd.concat([pd.Series(i).to_frame().transpose() for i in eval(row['annotation'])])
    df['hospital_id_patient_id'] = row['pat_id']
    df['wav_number'] = row['wav_number']
    return df

def preprocess_label_file(filename:str)-> pd.DataFrame:
    '''
    input : '/ext_ssd2/nia_vent/label_annotation/1-206.json
    output :  54 rows, pd.DataFrame(columns=['startTime', 'endTime', 'duration', 'extra', 'pat_id', 'wav_number'], ...)
    '''    
    records = [json.loads(line) for line in open(label_path, encoding='utf-8-sig')]
    df = pd.DataFrame(records).drop(columns=['name_VentilatorWeaning','wav_result_PRESSURE'])
    res = df['csv_path_FLOW'].str.split('/').apply(pd.Series)
    res.columns = ['pat_id','dropcol','csv']
    res['wav_number'] = res['csv'].str.replace('.csv','').str.split('-').str[-1].astype(int)
    res = res.drop(columns='dropcol')
    res['annotation'] = df['wav_result_FLOW']
    res = pd.concat(res.apply(annotation_dict_to_dataframe, axis=1).tolist()) # ['startTime', 'endTime', 'duration', 'extra', 'pat_id', 'wav_number', 'label_annotation']
    return res

def annotate(tmp:pd.DataFrame, prt=True)-> pd.DataFrame:
    annv = tmp['extra'].apply(lambda x: x['value'])
    if prt: print(annv.value_counts())
    tmp['label_annotation'] = 0
    # tmp.loc[tmp['extra']=={'value': 'normal', 'label': 'Normal'}, 'label_annotation'] = 0
    tmp.loc[annv=='true','label_annotation'] = 1
    tmp.loc[annv=='false','label_annotation'] = 2
    tmp.loc[annv=='noise','label_annotation'] = 3
    return tmp.drop(columns=['extra'])


def get_wav_firsttime(fi):
    if os.path.exists(fi):
        return pd.to_datetime(pd.read_csv(fi, nrows=1)['Time'].iloc[0])
    else:
        return np.nan

def annotate_each_label(label_num:int, ann_df:pd.DataFrame, feature_df:pd.DataFrame):
    common_keys = ['hospital_id_patient_id','wav_number'] if 'wav_number' in ann_df.columns else ['hospital_id_patient_id']
    ann_data = pd.merge(feature_df, ann_df[ann_df['label_annotation']==label_num], on=common_keys)
    cond1 = (ann_data['starttime']<=ann_data['ann_start'])&(ann_data['endtime']>=ann_data['ann_end']) # anntation이 instance 에 포함
    cond2 = (ann_data['starttime']<=ann_data['ann_start'])&(ann_data['endtime']>=ann_data['ann_start']) # annotation start가 instance 에 포함
    cond3 = (ann_data['starttime']<=ann_data['ann_end'])&(ann_data['endtime']>=ann_data['ann_end']) # annotation end가 instance 에 포함
    ann_index = ann_data[cond1|cond2|cond3]['instance_index']
    return ann_index

def annotate_one_instance_file(data_path):
    '''
    input: instance file path (e.g. '/VOLUME/nia_vent_asynchrony/data/processed_data/snu/instance_snu_1-200_1-299_108046_2022-10-27.pkl')

    output: pd.DataFrame(columns=['flow_path', 'starttime', 'endtime', 'data', 'hospital_id',
        'patient_id', 'wav_number', 'hospital_id_patient_id', 'instance_index',
        'label'],...)
    '''
    print('read', data_path)
    # 1분단위 instance 데이터 파일 읽기
    # columns: ['flow_path', 'starttime', 'endtime', 'data', 'hospital_id', 'patient_id', 'wav_number', 'hospital_id_patient_id']
    feature_df = pd.read_pickle(data_path) # 6 secs
    print(len(feature_df[['hospital_id_patient_id']].drop_duplicates()),'patients in feature file')
    print(len(feature_df[['hospital_id_patient_id','wav_number']].drop_duplicates()),'waveforms in feature file')
    print(len(feature_df), 'instances in feature file')

    # 어노테이션 있는 파형만 가져옴
    feature_df = pd.merge(feature_df, ann_df[['hospital_id_patient_id','wav_number']].drop_duplicates())
    print(len(feature_df[['hospital_id_patient_id','wav_number']].drop_duplicates()),'waveforms exist annotations')
    print(len(feature_df), 'instances exist annotations')
        
    # 어노테이션 시간 기준을 float에서 timestamp로 변환. 이때 파형 파일을 읽어 첫 시작 시간 가져옴
    unique_wav_df = feature_df[['hospital_id_patient_id','wav_number','flow_path']].drop_duplicates()
    unique_wav_df['wav_firsttime'] = unique_wav_df['flow_path'].apply(get_wav_firsttime)
    assert unique_wav_df['wav_firsttime'].isna().sum()==0, 'first time of waveform file is unknown'

    ann_df_feature = pd.merge(ann_df, unique_wav_df, on=['hospital_id_patient_id','wav_number'])
    ann_df_feature['ann_start'] = ann_df_feature['wav_firsttime']+ann_df_feature['ann_startTime_float'].apply(lambda x: pd.Timedelta(seconds=x))
    ann_df_feature['ann_end'] = ann_df_feature['wav_firsttime']+ann_df_feature['ann_endTime_float'].apply(lambda x: pd.Timedelta(seconds=x))

    # instance 1분간 어노테이션 존재 시 라벨링 (우선 순위 1 true > 2 false > 3 noise )
    feature_df['instance_index'] = range(len(feature_df))
    ann_index_3 = annotate_each_label(3, ann_df_feature, feature_df)
    ann_index_2 = annotate_each_label(2, ann_df_feature, feature_df)
    ann_index_1 = annotate_each_label(1, ann_df_feature, feature_df)
    print(len(ann_index_1), len(ann_index_2), len(ann_index_3))

    feature_df['label'] = 0
    feature_df.loc[feature_df['instance_index'].isin(ann_index_3),'label'] = 3
    feature_df.loc[feature_df['instance_index'].isin(ann_index_2),'label'] = 2
    feature_df.loc[feature_df['instance_index'].isin(ann_index_1),'label'] = 1
    print(feature_df.label.value_counts())
    feature_df = feature_df.loc[:,feature_df.columns!='instance_index'] # aju
    return feature_df

def main():
    parser = argparse.ArgumentParser(description='feature data generation')
    parser.add_argument('--config', default='config/preprocess_config.yml', help='config file path')
    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    print(args)
    print(config)

    nowDate = cutils.get_today_string()
    RESULT_PATH = osp.join(config['result_dir'], nowDate)
    os.makedirs(RESULT_PATH, exist_ok=True)
    print(RESULT_PATH)


    # load label files
    print('find label files')
    label_path_list = []
    for pat_range in config['train_id']+config['validation_id']:
        path_regex = osp.join(config['label_path'], pat_range+'.json')
        files = glob.glob(path_regex)
        print(path_regex, len(files), 'files')
        label_path_list.extend(files)


    # 어노테이션 파일 모두 읽기
    print('preprocess', len(label_path_list), 'label files')
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

    print(len(ann_df[['hospital_id_patient_id']].drop_duplicates()),'patients in label file')
    print(len(ann_df[['hospital_id_patient_id','wav_number']].drop_duplicates()),'wav in label file')
    print(len(ann_df), 'annotations in label file')


    # feature 에 라벨 달기
    feature_df_list = []
    for data_path in config['data_path']:
        feature_df = annotate_one_instance_file(data_path)
        feature_df_list.append(feature_df)
    feature_df = pd.concat(feature_df_list)


    # train, validation 나누기
    train_idx = [feature_df['hospital_id_patient_id'].str.match(pat.replace('*','[0-9]+')) for pat in config['train_id']]
    train_idx = pd.concat(train_idx, axis=1).sum(axis=1)>0
    validation_idx = [feature_df['hospital_id_patient_id'].str.match(pat.replace('*','[0-9]+')) for pat in config['validation_id']]
    validation_idx = pd.concat(validation_idx, axis=1).sum(axis=1)>0


    feature_df['split']=''
    feature_df.loc[train_idx,'split'] = 'train'
    feature_df.loc[validation_idx,'split'] = 'valid'
    print(feature_df['split'].value_counts())
    print('total', len(feature_df))


    print('label statistics')
    print(feature_df['label'].value_counts())
    print(feature_df['label'].value_counts(normalize=True))
    print('--train')
    print(feature_df[feature_df['split']=='train']['label'].value_counts())
    print(feature_df[feature_df['split']=='train']['label'].value_counts(normalize=True))
    print('--valid')
    print(feature_df[feature_df['split']=='valid']['label'].value_counts())
    print(feature_df[feature_df['split']=='valid']['label'].value_counts(normalize=True))


    labelfreq = pd.concat([feature_df.groupby('split')['label'].value_counts(),
    dat.groupby('split')['label'].value_counts(normalize=True)], axis=1)
    labelfreq.columns = ['num', 'ratio']
    print(labelfreq)
    labelfreq.to_csv(osp.join(RESULT_PATH, 'label_freq_trainvaltest.csv'))


    train_dataloader, val_dataloader, _ = preprocess_data(feature_df, set_train_weightedsampler=True, scale=True)
    # 89934 instance, 20 threads, elapsed 1 min



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ventdys_model = AsynchModel(input_dim=2, padding_mode='replicate').to(device)

    print(summary(ventdys_model, input_size=(2, 3600), device='cuda'))

    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(ventdys_model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience = 10, verbose = True, path = RESULT_PATH, checkpoint_name='ventdys_model.pt')


    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []


    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')

        ventdys_model.train()
        for i_step, data in enumerate(train_dataloader):
            xi, target = data
            xi = xi.to(device)
            target = target.to(device).squeeze(-1).squeeze(-1).float()
            out = ventdys_model(xi)

            loss = calculate_bce_loss(out[:,1], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # print('step', i_step, 'loss', loss.item())
        print('epoch', epoch, 'loss', loss.item())
        
        ######################    
        # validate the model #
        ######################
        ventdys_model.eval() 

        for i_step, data in enumerate(val_dataloader) :
            xi, target = data
            xi = xi.to(device)
            batch_size = len(xi)
            target = target.to(device).squeeze(-1).squeeze(-1).float()
            out = ventdys_model(xi)
            loss = calculate_bce_loss(out[:,1], target)
            
            valid_losses.append(loss.item())


        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
            
        epoch_len = len(str(n_epochs))


        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, ventdys_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    main()



