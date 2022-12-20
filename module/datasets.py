import json
import time
import os
import os.path as osp

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler

class CustomDataset(Dataset): 
  def __init__(self, x_data, y_data, metainfo=None, prt=True):
    if prt: print('x_data', x_data.shape)
    self.metainfo = metainfo

    x_data = np.nan_to_num(x_data)
    # nan_index = np.unique(np.where(np.isnan(x_data))[0])
    # print('nan index', len(nan_index))
    # if len(nan_index)>0:
    #     for ni in  nan_index[0]:
    #         x_data[ni] = pd.DataFrame(x_data[ni]).interpolate().values


    trans = transforms.Compose([transforms.ToTensor()])
    self.x_data = trans(x_data).permute(1, 0, 2).float()
    self.y_data = trans(y_data.reshape(1, 1, -1)).long()
    if prt: print('trans x_data', self.x_data.shape)
    if prt: print('trans y_data', self.y_data.shape)
  
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx, prt=False): 
    if prt: print(self.x_data.shape)
    # sc = RobustScaler()
    x = self.x_data[idx]
    # x = sc.fit_transform(x)
    # x = self.x_data[idx] 
    y = self.y_data[idx]
    return x, y

def set_dataloader(dataset, drop_last=False, shuffle=False, batch_size=128, weightsampler=False):
    num_workers = 50
    if weightsampler:
      y_data = (dataset.y_data>0).long() # 0 vs 1+3
      class_sample_count = np.array([len(np.where(y_data==t)[0]) for t in np.unique(y_data)])

      class_weights = 1./class_sample_count 
      class_weights = torch.from_numpy(class_weights.copy()).type('torch.DoubleTensor')
      class_weights_all_instance = class_weights[y_data].squeeze()

      weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
          weights=class_weights_all_instance,
          num_samples=int(len(class_weights_all_instance)/10),
          replacement=True
      )

      return DataLoader(dataset,
                      batch_size=int(batch_size),
                      num_workers=num_workers,
                      drop_last=drop_last,
                      sampler=weighted_sampler)
    else:
      return DataLoader(dataset, batch_size=int(batch_size), drop_last=drop_last, shuffle=shuffle, 
              num_workers=num_workers)

def apply_scaling(X_test):
    # scaler_1 = RobustScaler() # 3600개 마다 median을 구함..
    # scaler_2 = RobustScaler()
    # X_train_1 = scaler_1.fit_transform(X_train[:,:,0])
    # X_train_2 = scaler_2.fit_transform(X_train[:,:,1])
    # X_train = np.concatenate([X_train_1.reshape(-1,3600,1),
    #     X_train_2.reshape(-1,3600,1)
    #     ], axis=2)
    X_test_1 = scaler_1.transform(X_test[:,:,0])
    X_test_2 = scaler_2.transform(X_test[:,:,1])
    return np.concatenate([X_test_1.reshape(-1,3600,1),
        X_test_2.reshape(-1,3600,1)
        ], axis=2)

def apply_minmaxscaling(X_test: np.array): #  # (339439, 3600, 2)
    min_stat = np.nanmin(X_test, axis=1) # (instance, 2)
    max_stat = np.nanmax(X_test, axis=1) # (instance, 2)
    return np.concatenate([((X_test[:,i,:]-min_stat)/(max_stat-min_stat+1e-5)).reshape(-1, 1, 2) for i in range(3600)], axis=1) # (339439, 3600, 2)

def preprocess_data(dat: pd.DataFrame, make_valset = True, set_train_weightedsampler=False, scale=False, target=[1,2]):
    '''
    dat: 'data'컬럼에 (3600, 2) 데이터 존재해야, label 컬럼, 그 밖의 meta info 컬럼 ('flow_path', 'starttime', 'endtime', 'hospital_id_patient_id', 'wav_number', 'instance_index', 'label', 'split')

    '''
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    dat = dat[dat['data'].apply(lambda x: x.shape)==(3600,2)] # data dimension 맞는 것만
    X = np.concatenate([i.reshape(1, 3600, -1) for i in dat['data'].tolist()])
    # feature = dat.loc[:,dat.columns!='label'].values
    
    if 'label' in dat:
      if isinstance(target, list):
        y = (dat['label'].isin(target)).values
      elif isinstance(target, int): 
        y = (dat['label']==target).values
      elif target is None: # multi-class
        y = dat['label'].values
        # label = (dat['label']>0).values
        # y = (dat['label'].isin([1,2])).values
    else:
        y = np.array([np.nan]*len(dat))

    metainfo = dat.loc[:,dat.columns!='data']

    if 'split' in dat.columns:
        train_index = np.where(dat['split']=='train')[0]
        val_index = np.where(dat['split']=='valid')[0]
        test_index = np.where(dat['split']=='test')[0]
    else:
        rs = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        trainval_index, test_index = next(iter(rs.split(X)))
        if make_valset:
            rs2 = ShuffleSplit(n_splits=1, test_size=.125, random_state=42)
            tr, val = next(iter(rs2.split(trainval_index)))
            train_index = trainval_index[tr]
            val_index = trainval_index[val]
        else:
            train_index = trainval_index
            val_index = []

    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    X_test, y_test = X[test_index], y[test_index]

    meta_train = metainfo.iloc[train_index]
    meta_val = metainfo.iloc[val_index]
    meta_test = metainfo.iloc[test_index]
    print('no of train, val, test', len(y_train), len(y_val), len(y_test))

    if (len(y_train)>0):
      if scale:
        for i in range(len(X_train)//100000):
          X_train[i*100000:min(i*100000+100000, len(X_train))] = apply_minmaxscaling(X_train[i*100000:min(i*100000+100000, len(X_train))])
      trainset = CustomDataset(X_train, y_train, meta_train)
      train_dataloader = set_dataloader(trainset, drop_last=True, shuffle=True, weightsampler=set_train_weightedsampler)
    if (len(y_test)>0):
      if scale:
        X_test = apply_minmaxscaling(X_test)
        # X_test = scaler.transform(X_test)
      testset = CustomDataset(X_test, y_test, meta_test)
      test_dataloader = set_dataloader(testset, batch_size=1024)

    if make_valset&(len(y_val)>0):
      if scale:
        X_val = apply_minmaxscaling(X_val)
      valset = CustomDataset(X_val, y_val, meta_val)
      val_dataloader = set_dataloader(valset)
      print('X.shape', X_train.shape, X_val.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/(len(y_train)+1e-4), sum(y_val==1)/(len(y_val)+1e-4), sum(y_test==1)/(len(y_test)+1e-4))
    else:
      print('X.shape', X_train.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/(len(y_train)+1e-4), sum(y_test==1)/(len(y_test)+1e-4))
    return train_dataloader, val_dataloader, test_dataloader

def setup_data(X, y, make_valset=False):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    if make_valset:
      X_train, X_val, y_train, y_val = train_test_split(
          X_trainval, y_trainval, test_size=0.125, random_state=42)
    else:
      X_train, y_train = X_trainval, y_trainval

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    trainset = CustomDataset(X_train, y_train)
    testset = CustomDataset(X_test, y_test)

    train_dataloader = set_dataloader(trainset, drop_last=True, shuffle=True)
    val_dataloader = None
    test_dataloader = set_dataloader(testset, batch_size=1)

    if make_valset:
      X_val = scaler.transform(X_val)
      valset = CustomDataset(X_val, y_val)
      val_dataloader = set_dataloader(valset, batch_size=1)
      print('X.shape', X_train.shape, X_val.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/len(y_train), sum(y_val==1)/len(y_val), sum(y_test==1)/len(y_test))
    else:
      print('X.shape', X_train.shape, X_test.shape)
      print('Y class distribution', sum(y_train==1)/len(y_train), sum(y_test==1)/len(y_test))

    return train_dataloader, val_dataloader, test_dataloader


def annotation_dict_to_dataframe(row):
    '''
    input : pd.Series(['1-206', '1-206-013.csv',13, '[{"startTime": 424.00848, "endTime": 431.99197300000003, "duration": 7.98349300000001, "extra": {"value": "noise", "label": "Noise"}}, ...')
    output : 2 rows, pd.DataFrame(columns=['startTime', 'endTime', 'duration', 'extra','pat_id','wav_number'], ... ) 
    '''
    df = pd.concat([pd.Series(i).to_frame().transpose() for i in row['annotation']])
    df['hospital_id_patient_id'] = row['pat_id']
    df['wav_number'] = row['wav_number']
    return df

def preprocess_label_file(filename:str)-> pd.DataFrame:
    '''
    input : '/ext_ssd2/nia_vent/label_annotation/1-206.json
    output :  54 rows, pd.DataFrame(columns=['startTime', 'endTime', 'duration', 'extra', 'pat_id', 'wav_number'], ...)
    '''    
    with open(filename, encoding='utf-8-sig') as f:
      records = json.load(f)['Patient_Data']
    df = pd.DataFrame(records).drop(columns=['name_VentilatorWeaning','wav_result_PRESSURE'])
    if records[0]['csv_path_FLOW']=='ALL':
      res = pd.DataFrame(records[0]['wav_result_FLOW'])
      res['hospital_id_patient_id'] = filename.split('/')[-1].split('.json')[0]
      res['wav_number'] = 1
    else:
      res = df['csv_path_FLOW'].str.replace('.csv.csv','.csv').str.replace('.vital','').str.split('/').apply(pd.Series)
      res.columns = ['pat_id','dropcol','csv']
      res['wav_number'] = res['csv'].str.replace('.csv','').str.split('-').str[-1].astype(int)
      res = res.drop(columns='dropcol')
      res['annotation'] = df['wav_result_FLOW']
      res = pd.concat(res.apply(annotation_dict_to_dataframe, axis=1).tolist()) # ['startTime', 'endTime', 'duration', 'extra', 'pat_id', 'wav_number']
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

def annotate_one_instance_file(data_path, ann_df):
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
    unique_wav_df['wav_firsttime'] = unique_wav_df['flow_path'].apply(get_wav_firsttime) # NOTE: It requires raw flow files exist
    assert unique_wav_df['wav_firsttime'].isna().sum()==0, 'first time of waveform file is unknown'+ ', '.join(unique_wav_df[unique_wav_df['wav_firsttime'].isna()]['flow_path'].tolist())

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
    feature_df.loc[feature_df['instance_index'].isin(ann_index_3),'label'] = 3 # noise
    feature_df.loc[feature_df['instance_index'].isin(ann_index_2),'label'] = 2 # false asynchrony
    feature_df.loc[feature_df['instance_index'].isin(ann_index_1),'label'] = 1 # true asynchrony
    print(feature_df.label.value_counts())
    feature_df.loc[feature_df['instance_index'].isin(ann_index_2),'label'] = 1
    feature_df.loc[feature_df['instance_index'].isin(ann_index_3),'label'] = 2
    print(' false asynchrony(2) merge with true asynchrony(1)')
    print(' noise (3) change to noise(2)')
    print(feature_df.label.value_counts())

    feature_df = feature_df.loc[:,feature_df.columns!='instance_index'] # aju
    return feature_df
