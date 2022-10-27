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

os.chdir('/VOLUME/nia_vent_asynchrony')
from module.datasets import preprocess_data#, set_dataloader, CustomDataset
from module.utils import load_and_stack_data, EarlyStopping
from model.VAE import VAE
from model.AsynchModel import AsynchModel
from module.metrics import calculate_any_metrics
from module.loss import calculate_vae_loss, calculate_bce_loss
import module.utils as cutils


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0


def main(config):
    ckpt_path = config['ckpt_path']
    data_path = config['data_path']
    threshold = config['threshold']
    result_path = config['result_path']
    
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config['CUDA_VISIBLE_DEVICES'])
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # load model
    ckpt_dict = torch.load(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(ckpt_path, device)
    ventdys_model = AsynchModel(input_dim=2, padding_mode='replicate').to(device)
    ventdys_model.load_state_dict(ckpt_dict, strict=False)

    # read data
    dat_infer = pd.read_pickle(data_path)
    dat_infer['split'] = 'test'
    print('len(dat_infer)', len(dat_infer))


    # testset = ...
    _, _, test_dataloader = preprocess_data(dat_infer, scale=True)
    # 315386 rows, elapsed 5min


    # predict testset 
    ventdys_model.eval()

    y_prob = None
    for i_step, data in enumerate(test_dataloader):
        xi, target = data
        xi = xi.to(device)
        target = target.to(device).squeeze(-1).squeeze(-1).float() # torch.Size([bs, 1])
        out = ventdys_model(xi)

        
        loss = calculate_bce_loss(out[:,1], target)

        prob = F.softmax(out, dim=1)[:,1].detach().cpu()
        # pred = y_proba>0.5
        y_prob = prob if y_prob is None else torch.cat([y_prob, prob])

        # pred_list.append(out.detach().cpu().numpy())
        if i_step%1000==0: print('step', i_step, 'loss', loss.item())



    testset = test_dataloader.dataset


    testset_pred = testset.metainfo
    testset_pred['y_pred_prob'] = y_prob
    testset_pred['y_pred'] = y_prob>threshold
    if 'label' in testset_pred.columns:
        testset_pred['y_target'] = testset_pred['label'].isin([2,1])

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
    os.makedirs(result_path, exist_ok=True)
    path = osp.join(result_path, f'testset_{testset_pred["hospital_id_patient_id"].min()+"_"+testset_pred["hospital_id_patient_id"].max()}_{len(testset_pred)}_pred_{cutils.get_today_string(False)}.csv')
    testset_pred.to_csv(path)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='infer testset')
    parser.add_argument('--config', default='config/test_config.yml', help='config file path')
    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    print(args)
    print(config)
    main(config)



