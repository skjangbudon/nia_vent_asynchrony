import os
import glob
import os.path as osp
import warnings
warnings.filterwarnings(action='ignore')
import multiprocessing
from functools import partial
import argparse
import datetime

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
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0



def main():
    parser = argparse.ArgumentParser(description='infer testset')
    parser.add_argument('--config', default='/VOLUME/nia_vent_asynchrony/config/test_config.yml', help='config file path')
    args = parser.parse_args()
    
    config = cutils.load_yaml(args.config)
    ckpt_path = config['ckpt_path']
    data_path = config['data_path']
    sample_num = config['sample_num'] if 'sample_num' in config else None
    
    nowDate = cutils.get_today_string()
    RESULT_PATH = osp.join(config['result_dir'], nowDate)
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    logger = cutils.set_logger('test', path=RESULT_PATH)
    logger.info('RESULT_PATH:'+RESULT_PATH)
    
    logger.info('config :')
    logger.info(config)
    logger.info('')
    logger.info(f'[ {datetime.datetime.now()} ] inference started')
       
    # 실행 명령어 출력
    logger.info('command :')
    logger.info(f'{os.path.basename(__file__)} --config {args.config}')
    logger.info('')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['CUDA_VISIBLE_DEVICES'])
    logger.info('Current cuda device:'+str(torch.cuda.current_device()))
    logger.info('Count of using GPUs:'+str(torch.cuda.device_count()))


    # load label files
    logger.info('find label files')
    label_path_list = []
    for pat_range in config['test_id']:
        path_regex = osp.join(config['label_path'], pat_range+'.json')
        files = glob.glob(path_regex)
        logger.info(path_regex+str(len(files))+' files')
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
    logger.info(str(len(ann_df))+' annotations in label file')


    # feature 에 라벨 달기
    feature_df_list = []
    for data_path in config['data_path']:
        feature_df = annotate_one_instance_file(data_path, ann_df)
        feature_df_list.append(feature_df)
    feature_df = pd.concat(feature_df_list)
    
    if sample_num is not None:    
        feature_df = pd.concat([feature_df[feature_df['label']==0].sample(n=sample_num[0]),
                                feature_df[feature_df['label']==1].sample(n=sample_num[1]),
                                feature_df[feature_df['label']==2].sample(n=sample_num[2])])
    
    feature_df['split']='test'
    
    logger.info('label statistics')
    logger.info(feature_df['label'].value_counts())
    logger.info(feature_df['label'].value_counts(normalize=True))
    logger.info('no. hospital_id_patient_id by set')
    logger.info(feature_df.groupby(['split'])['hospital_id_patient_id'].nunique())
    
    labelfreq = pd.concat([feature_df.groupby('split')['label'].value_counts(),
    feature_df.groupby('split')['label'].value_counts(normalize=True)], axis=1)
    labelfreq.columns = ['num', 'ratio']
    logger.info(labelfreq)
    labelfreq.to_csv(osp.join(RESULT_PATH, 'label_freq_test.csv'))


    # load model
    ckpt_dict = torch.load(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(ckpt_path)
    logger.info(device)
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
        if i_step%1000==0: logger.info(f'step {i_step} loss {loss.item()}')



    testset = test_dataloader.dataset


    testset_pred = testset.metainfo
    testset_pred['y_pred_prob'] = y_prob
    testset_pred['y_pred'] = y_pred
    if 'label' in testset_pred.columns:
        testset_pred['y_target'] = testset_pred['label']

    # 계산할 때 사용된 값 (ex. Confusion Matrix 기반 TP, FP, TN, FN), 최종 결과값 (f1-score)
    eval_score, report = calculate_multiclass_metrics(testset_pred['y_target'].values, testset_pred['y_pred'].values, ['normal','asynchrony','noise'])
    report.to_csv(osp.join(RESULT_PATH, 'classification_report.csv'))
    pd.DataFrame(eval_score, index=['score']).to_csv(osp.join(RESULT_PATH, 'score.csv'))
    logger.info('')
    logger.info('F1 score:')
    logger.info(eval_score['f1_score_micro'])
    # logger.info('report:')
    # logger.info(report)
                        
    logger.info(f'{testset_pred["hospital_id_patient_id"].nunique()} hospital_id_patient_id')


    logger.info(testset_pred['y_pred'].value_counts())
    logger.info(testset_pred['y_pred'].value_counts(normalize=True))


    # logger.info(len(testset_pred[['hospital_id_patient_id','wav_number']].drop_duplicates()),
    #     len(testset_pred.query('y_pred==1')[['hospital_id_patient_id','wav_number']].drop_duplicates()),
    #     len(testset_pred.query('y_pred==1')[['hospital_id_patient_id','wav_number']].drop_duplicates())/len(testset_pred[['hospital_id_patient_id','wav_number']].drop_duplicates()),
    #     testset_pred['hospital_id_patient_id'].min()+'_'+testset_pred['hospital_id_patient_id'].max())


    # write inference results
    # 문제 당 개별 결과값 
    path = osp.join(RESULT_PATH, f'testset_{testset_pred["hospital_id_patient_id"].min()+"_"+testset_pred["hospital_id_patient_id"].max()}_{len(testset_pred)}_pred_{cutils.get_today_string(False)}.csv')
    testset_pred.to_csv(path)
    logger.info('individual inference result saved in:')
    logger.info(path)
    
    logger.info('')
    logger.info('testset id list :')
    logger.info(testset_pred['hospital_id_patient_id'].unique())

    logger.info(f'[ {datetime.datetime.now()} ] inference ended')

if __name__ == "__main__":
    main()



