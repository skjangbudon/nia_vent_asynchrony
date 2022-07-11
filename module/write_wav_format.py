import re
import os
import yaml
import glob
import pandas as pd
import numpy as np
import soundfile as sf

def load_yaml(fname):
    with open(fname, 'r') as fp:
        data = yaml.safe_load(fp)
    return data

def convert_and_write_wav_snu(config, save_format=['wav']):
    '''
    서울대 원본데이터를 읽어 FLOW없이 AWP만 wav 포맷 또는 csv 포맷으로 저장
    서울대 원본데이터 특징:
    파일명) MICU_01_220312_110000.vital.csv : 병실_환자번호_날짜_시간.vital.csv
    1 hour, around 60Hz, 216000 rows per raw file 
    columns: Time, Intellivue/AWP_WAV, Intellivue/FLOW_WAV
    '''
    sampling_rate_hz = 60
    print('target sampling_rate_hz :', sampling_rate_hz)

    filenames = glob.glob(os.path.join(config['data_dir'],'*'))
    print(len(filenames), 'files')

    for f in filenames:
        vit_raw = pd.read_csv(f, parse_dates=['Time'], index_col=0)
        # vit = vit_raw[vit_raw.notna().all(axis=1)] # drop na rows
        
        if 'csv' in save_format : 
            new_filename = os.path.basename(f)
            new_path = os.path.join(config['destination'], new_filename)
            vit.to_csv(new_path)

        # write wav file
        if 'wav' in save_format : 
            new_filename = re.sub('csv$','wav', os.path.basename(f))
            new_path = os.path.join(config['destination'], new_filename)
            sf.write(new_path, vit['Intellivue/AWP_WAV'].values, sampling_rate_hz)
            print(f, '->', new_path)

        # break

#TODO: concat 6*10min files to one file
def convert_and_write_wav_aju(config, resample_to_60hz=True, save_format=['wav']):
    '''
    아주대 원본데이터를 읽어 FLOW없이 AWP만 wav 포맷 또는 csv 포맷으로 저장
    아주대 원본데이터 특징:
    파일명) 8c91bd20c48a51487ef5_20190626075000_20190626080000_VENT_FLOW.csv 
    & 8c91bd20c48a51487ef5_20190626075000_20190626080000_VENT_PAW.csv 
    10min(600s), around 125Hz, 75000 rows per raw file
    '''

    sampling_rate_hz = 60 if resample_to_60hz else 125
    print('target sampling_rate_hz :', sampling_rate_hz)

    cols = ['Intellivue/AWP_WAV','Intellivue/FLOW_WAV']
    filenames = glob.glob(os.path.join(config['data_dir'],'*VENT_PAW.csv'))
    print(len(filenames), 'AWP files')

    for f in filenames:
        # read file
        # AWP
        vit = pd.read_csv(f, index_col=0).rename(columns={'signal': cols[0]})

        # FLOW -> not necessary
        # flow_name = f.replace('VENT_PAW', 'VENT_FLOW')
        # if os.path.exists(flow_name):
        #     vit_raw2 = pd.read_csv(f, index_col=0).rename(columns={'signal':cols[1]})
        #     # concat 2 channel
        #     vit = pd.concat([vit_raw1, vit_raw2], axis=1)
        # else:
        #     print('WARN :', flow_name, 'file does not exists')

        # vit = vit[vit.notna().all(axis=1)]  #drop na rows

        # set timestamp index
        tmp = os.path.basename(f).split('_')
        starttime = pd.to_datetime(tmp[1])
        endtime = pd.to_datetime(tmp[2])
        vit.index = [pd.Timestamp.fromtimestamp(i) for i in np.arange(starttime.timestamp(), endtime.timestamp(), step=1/125)]

        # resampling
        if resample_to_60hz:
            vit = vit.resample('0.008S').median()

        if 'csv' in save_format : 
            new_filename = os.path.basename(f)
            new_path = os.path.join(config['destination'], new_filename)
            vit.to_csv(new_path)

        # write wav file
        if 'wav' in save_format : 
            new_filename = re.sub('csv$','wav', os.path.basename(f))
            new_path = os.path.join(config['destination'], new_filename)
            sf.write(new_path, vit['Intellivue/AWP_WAV'].values, sampling_rate_hz)
            print(f, '->', new_path)

    def main():
        if os.path.exists(config['destination']):
            print('WARN:', len(os.listdir(config['destination'])), 'file exists in destination')

        os.makedirs(config['destination'], exist_ok=True)

        if config['from_ajou']:
            convert_and_write_wav_aju(config, save_format=[config['format']])
        else:
            convert_and_write_wav_snu(config, save_format=[config['format']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform raw waveform files and write csv or wav format')
    parser.add_argument('--config', default='config/wav_config.yml', help='config file')

    args = parser.parse_args()
    config = cutils.load_yaml(args.config)
    '''
    # configuration file example
    config = {
        'data_dir': '/VOLUME/sample/sample_AJ', # 원본데이터가 있는 폴더
        'from_ajou': True, # 아주대 형식인지 여부(파일명, sampling_rate)
        'destination': '/VOLUME/sample/sample_AJ_csv', # 변환된 데이터가 저장될 폴더 (생성 안되어있으면 자동으로 폴더 만듬)
        'format': 'csv'
    }

    config = {
        'data_dir': '/VOLUME/sample/sample_SNU',
        'from_ajou': False,
        'destination': '/VOLUME/sample/sample_SNU_csv',
        'format': 'csv'
    }
    '''

    main(args)

