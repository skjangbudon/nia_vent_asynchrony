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

import module.utils as cutils

instance_length_sec = 60
instance_length_sec_td = pd.Timedelta(seconds=instance_length_sec)

def get_wave_instance_per_waveformfile(i, wavecsv_list):
    if i%100==0: print(i)
    wav = None
    wave_i = wavecsv_list[i]
    try:
        wav = pd.read_csv(wave_i, parse_dates=['Time'])
    except Exception as e:
        print(e, wave_i)
        wav = pd.read_csv(wave_i)
        wav['Time'] = pd.to_datetime(wav['Time'])
        
    if wav is None:
        print(wave_i, 'return None because wav is None')
        return None

    try:
        wav['Intellivue/AWP_WAV'] = pd.read_csv(wave_i.replace('flow','pre').replace('-w-','-c-').replace('AWF','AWP'), parse_dates=['Time'])['Intellivue/AWP_WAV']
        n_instance = int((wav['Time'].max()-wav['Time'].min()).total_seconds()/instance_length_sec)

        if n_instance==0:
            print(wave_i, 'return None because n_instance==0')
            return None

        wav_starttime = wav['Time'].min()
        df_list = []
        for tdi in range(n_instance):
            one_instance_wav = wav[(wav['Time']>=wav_starttime+tdi*instance_length_sec_td)&(wav['Time']<wav_starttime+((tdi+1)*instance_length_sec_td))]
            one_instance = pd.Series({'flow_path': wave_i, 
                'starttime': one_instance_wav.min()['Time'], 'endtime': one_instance_wav.max()['Time'], 
                'data': one_instance_wav[['Intellivue/FLOW_WAV','Intellivue/AWP_WAV']].values
                })
            df_list.append(one_instance)
    except Exception as e:
        print(e, wave_i)
        return None
    return pd.concat(df_list, axis=1).transpose()

def main():
    parser = argparse.ArgumentParser(description='feature data generation')
    parser.add_argument('--config', default='/VOLUME/nia_vent_asynchrony/config/preprocess_config.yml', help='config file path')
    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    print(args)
    print(config)
    
    org = config['org']
    flow_path_list = config['flow_path']
    dest_dir = config['dest_dir']
    n_threads = config['n_threads']
    
    wavecsv_list = []
    # flow_path = '/VOLUME/nia_vent_asynchrony/data/raw_data/snu/20220930/1-500/AWF/*.csv*'
    for flow_path in flow_path_list:
        wavecsv_list.extend(glob.glob(flow_path))

    print('len(wavecsv_list)', len(wavecsv_list))
    print('wavecsv_list[0]', wavecsv_list[0])

    tmp = pd.DataFrame({'flow_path': wavecsv_list})
    tmp = tmp['flow_path'].str.split('/').str[-1].str.replace('.csv','').str.split('-|_').apply(pd.Series)
    tmp.columns = ['hospital_id', 'patient_id', 'wav_number']
    tmp['hospital_id_patient_id'] = tmp['hospital_id'].astype(str)+'-'+tmp['patient_id'].astype(str)
    tmp['wav_number'] = tmp['wav_number'].astype(int)
    tmp_flow = tmp.copy()
    print('no of waveform files by patient')
    print(tmp['hospital_id_patient_id'].value_counts().sort_index())

    # check the number of files of flow and pressure
    wavecsv_list_p = []
    for i in wavecsv_list:
        wavecsv_list_p.extend(glob.glob(i.replace('AWF','AWP')))
    tmp = pd.DataFrame({'flow_path': wavecsv_list_p})
    tmp = tmp['flow_path'].str.split('/').str[-1].str.split('-|_').apply(pd.Series)
    tmp.columns = ['hospital_id', 'patient_id', 'wav_number']
    tmp['hospital_id_patient_id'] = tmp['hospital_id'].astype(str)+'-'+tmp['patient_id'].astype(str)
    tmp['wav_number'] = tmp['wav_number'].str.replace('.csv.gz','').str.replace('.csv','').astype(int)
    tmp_pres = tmp.copy()
    tmp_pres['type'] = 'pres'
    tmp_flow['type'] = 'flow'
    mg = pd.merge(tmp_pres, tmp_flow, how='outer', on=tmp.columns.tolist())
    gr_stat = pd.concat([tmp_pres.groupby('hospital_id_patient_id').count(),
    tmp_flow.groupby('hospital_id_patient_id').count()
    ], axis=1)
    # gr_stat.to_csv('file_count.csv')
    if any(gr_stat.iloc[:,0]!=gr_stat.iloc[:,5]):
        print('WARNING: unmatched', np.where(gr_stat.iloc[:,0]!=gr_stat.iloc[:,5]))


    # divide data into 60 sec 

    since = time.time()
    n_cpu = multiprocessing.cpu_count()
    print(f'no. cpu existed : {n_cpu}, use {n_threads} threads')

    print(len(wavecsv_list))
    pool = multiprocessing.Pool(processes=n_threads)
    func = partial(get_wave_instance_per_waveformfile, wavecsv_list=wavecsv_list)
    result = pool.map(func, range(len(wavecsv_list)))
    pool.close()
    pool.join()

    print('aggregate all instances')
    data = []
    for i in result :
        if i is not None:
            data.append(i)
    datadf = pd.concat(data)
    print('elapsed time (sec):', time.time()-since)

    # write results
    nowDate = cutils.get_today_string(False)
    # dest_path = osp.join(dest_dir, org, f'instance_{org}_{len(datadf)}_{nowDate}.pkl')
    # print(dest_path)
    # datadf.to_pickle(dest_path)

    tmp = datadf['flow_path'].str.split('/').str[-1].str.split('-|_').apply(pd.Series)
    tmp.columns = ['hospital_id', 'patient_id', 'wav_number']
    tmp['hospital_id_patient_id'] = tmp['hospital_id']+'-'+tmp['patient_id']
    tmp['wav_number'] = tmp['wav_number'].str.replace('.csv.gz','').str.replace('.csv','').astype(int)
    datadf_info = pd.concat([datadf, tmp], axis=1)
    # 3157055 rows, 12min

    # exclude na ratio >50%
    print(len(datadf_info))
    na_index = (datadf_info['data'].apply(lambda x:np.isnan(x).sum())>=instance_length_sec*60)
    datadf_info_na = datadf_info[na_index]

    datadf_info = datadf_info[~na_index]
    print(len(datadf_info))

    del datadf_info_na['data']
    nowDate = cutils.get_today_string(False)
    dest_path = osp.join(dest_dir, org, f'instance_{org}_{datadf_info_na["hospital_id_patient_id"].min()}_{datadf_info_na["hospital_id_patient_id"].max()}_{len(datadf_info)}_nan_{nowDate}.csv')
    print(dest_path)
    os.makedirs(osp.dirname(dest_path), exist_ok=True)
    datadf_info_na.to_csv(dest_path, index=False)

    nowDate = cutils.get_today_string(False)
    dest_path = osp.join(dest_dir, org, f'instance_{org}_{datadf_info["hospital_id_patient_id"].min()}_{datadf_info["hospital_id_patient_id"].max()}_{len(datadf_info)}_{nowDate}.pkl')
    print(dest_path)
    datadf_info.to_pickle(dest_path)

if __name__ == "__main__":
    main()



