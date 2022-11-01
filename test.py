import os
import glob
import os.path as osp
import warnings
warnings.filterwarnings(action='ignore')
import multiprocessing
from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchsummary import summary

from module.utils import load_and_stack_data, EarlyStopping
from model.AsynchModel import AsynchModel
from module.metrics import calculate_multiclass_metrics
from module.loss import calculate_ce_loss
import module.utils as cutils
from module.datasets import preprocess_label_file, annotate
from module.datasets import annotate_one_instance_file, preprocess_data

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0


def main():
    parser = argparse.ArgumentParser(description='infer testset')
    parser.add_argument('--config', default='/VOLUME/nia_vent_asynchrony/config/test_config.yml', help='config file path')
    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    print(args)
    print(config)
        
    ckpt_path = config['ckpt_path']
    data_path = config['data_path']
    RESULT_PATH = config['result_path']
    os.makedirs(RESULT_PATH, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(config['CUDA_VISIBLE_DEVICES'])
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


    # load label files
    print('find label files')
    label_path_list = []
    for pat_range in config['test_id']:
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
        feature_df = annotate_one_instance_file(data_path, ann_df)
        feature_df_list.append(feature_df)
    feature_df = pd.concat(feature_df_list)
    
    feature_df['split']='test'
    
    print('label statistics')
    print(feature_df['label'].value_counts())
    print(feature_df['label'].value_counts(normalize=True))
    
    
    labelfreq = pd.concat([feature_df.groupby('split')['label'].value_counts(),
    feature_df.groupby('split')['label'].value_counts(normalize=True)], axis=1)
    labelfreq.columns = ['num', 'ratio']
    print(labelfreq)
    labelfreq.to_csv(osp.join(RESULT_PATH, 'label_freq_test.csv'))


    # load model
    ckpt_dict = torch.load(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(ckpt_path, device)
    ventdys_model = AsynchModel(input_dim=2, padding_mode='replicate', num_class=feature_df['label'].nunique()).to(device)
    ventdys_model.load_state_dict(ckpt_dict, strict=False)


    # testset = ...
    _, _, test_dataloader = preprocess_data(feature_df, scale=True, target=None)
    # 315386 rows, elapsed 5min


    # predict testset 
    ventdys_model.eval()

    y_prob = None
    y_pred = None
    for i_step, data in enumerate(test_dataloader):
        xi, target = data
        xi = xi.to(device)
        target = target.to(device).squeeze(-1).squeeze(-1).long() # torch.Size([bs, 1])
        out = ventdys_model(xi)

        
        loss = calculate_ce_loss(out, target)
        
        probs_all = F.softmax(out, dim=1).detach().cpu()
        prob, preds = torch.max(probs_all, dim=1)
        
        y_prob = prob if y_prob is None else torch.cat([y_prob, prob])
        y_pred = preds if y_pred is None else torch.cat([y_pred, preds])
        
        # pred_list.append(out.detach().cpu().numpy())
        if i_step%1000==0: print('step', i_step, 'loss', loss.item())



    testset = test_dataloader.dataset


    testset_pred = testset.metainfo
    testset_pred['y_pred_prob'] = y_prob
    testset_pred['y_pred'] = y_pred
    if 'label' in testset_pred.columns:
        testset_pred['y_target'] = testset_pred['label']

    eval_score = calculate_multiclass_metrics(testset_pred['y_target'].values, testset_pred['y_pred'].values, ['normal','asynchrony','noise'])
    path = osp.join(RESULT_PATH, 'score.csv')
    pd.DataFrame(eval_score, index=['score']).to_csv(path)
                    
    print(testset_pred['hospital_id_patient_id'].unique(), 'hospital_id_patient_id')


    perc =  [.05, .1 , .25, .5, .75, .9, .95, .99, .999]
    percentile = pd.DataFrame(np.percentile(testset_pred['y_pred_prob'], perc).reshape(1, -1), columns=perc, index=['y_prob'])
    print(percentile)


    # testset_pred['y_pred_prob'].hist()
    print(testset_pred['y_pred'].value_counts())
    print(testset_pred['y_pred'].value_counts(normalize=True))


    print(len(testset_pred[['hospital_id_patient_id','wav_number']].drop_duplicates()),
        len(testset_pred.query('y_pred==1')[['hospital_id_patient_id','wav_number']].drop_duplicates()),
        len(testset_pred.query('y_pred==1')[['hospital_id_patient_id','wav_number']].drop_duplicates())/len(testset_pred[['hospital_id_patient_id','wav_number']].drop_duplicates()),
        testset_pred['hospital_id_patient_id'].min()+'_'+testset_pred['hospital_id_patient_id'].max())


    # write inference results
    path = osp.join(RESULT_PATH, f'testset_{testset_pred["hospital_id_patient_id"].min()+"_"+testset_pred["hospital_id_patient_id"].max()}_{len(testset_pred)}_pred_{cutils.get_today_string(False)}.csv')
    testset_pred.to_csv(path)
    print(path)


if __name__ == "__main__":
    main()



