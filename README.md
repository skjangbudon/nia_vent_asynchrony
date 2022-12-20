# Patient-ventilator asynchrony classification model 유효성 평가
- 2022년 인공지능 학습용 데이터 구축 사업의 일환으로, 인공호흡기 파형에서 정상과 노이즈, 부조화를 분류하는 모델을 개발함
- 환자와 인공호흡기의 호흡 주기가 맞지 않으면 환자의 불편감이 초래되며, 폐손상의 악화로 
인해 예후가 좋지 않음. 따라서 지속적으로 기계환기 그래프 파형을 관찰하여 비동시성을 
감시해야 함. 그러나 이에 따라 전문의료인의 노동력이 소모되므로 호흡 주기 부조화를 자동적으로 감지하는 모델은 불필요한 의료 비용을 감소시킬 수 있음

## User guides
### 0. Settings
```
# download a dokcer image file in s3 storage or github 
$ git clone https://github.com/skjangbudon/nia_vent_asynchrony.git

# create docker image using .tar file 
$ docker load -i vent_asynchrony.tar

# run bash command in a new container
$ docker run -dit -v (path):/data --name vent_async_eval vent_async:latest bash

# for example,
$ docker run -dit -v /data/project/nia_vent:/data --name vent_async_eval --shm-size 500G --gpus all vent_async:latest bash

# execute a running container
$ docker exec -it nia_vent_async bash

```

### 1. Data Preprocessing
> python preprocess.py --config config/preprocess_config.yml

csv형식의 waveform 파일을 입력받아 1분 단위의 instance로 변환한 뒤 pickle 파일로 저장함

### 2. Training (유효성 평가 시 생략)
> python train.py --config config/train_config.yml

전처리가 완료된 pickle 파일과 라벨 파일을 입력받아 훈련셋(training set)과 검증셋(validation set)으로 나눈 뒤 정상, 부조화, 노이즈로 분류하는 모델을 훈련함

데이터 크기에 따라 메모리 자원이 많이 필요할 수 있음 

### 3. Test
> python test.py --config config/test_config.yml

훈련에 사용되지 않은 테스트셋으로 훈련된 모델을 평가함

0: normal, 1: asynchrony, 2: noise

## Model description
### Model architecture
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
AsynchModel                              [16, 2, 3600]             [16, 3]                   --
├─Sequential: 1-1                        [16, 2, 3600]             [16, 64, 56]              --
│    └─Conv1d: 2-1                       [16, 2, 3600]             [16, 16, 3600]            176
│    └─BatchNorm1d: 2-2                  [16, 16, 3600]            [16, 16, 3600]            32
│    └─ReLU: 2-3                         [16, 16, 3600]            [16, 16, 3600]            --
│    └─MaxPool1d: 2-4                    [16, 16, 3600]            [16, 16, 900]             --
│    └─Conv1d: 2-5                       [16, 16, 900]             [16, 32, 900]             2,592
│    └─BatchNorm1d: 2-6                  [16, 32, 900]             [16, 32, 900]             64
│    └─ReLU: 2-7                         [16, 32, 900]             [16, 32, 900]             --
│    └─MaxPool1d: 2-8                    [16, 32, 900]             [16, 32, 225]             --
│    └─Conv1d: 2-9                       [16, 32, 225]             [16, 32, 225]             5,152
│    └─BatchNorm1d: 2-10                 [16, 32, 225]             [16, 32, 225]             64
│    └─ReLU: 2-11                        [16, 32, 225]             [16, 32, 225]             --
│    └─MaxPool1d: 2-12                   [16, 32, 225]             [16, 32, 56]              --
│    └─Conv1d: 2-13                      [16, 32, 56]              [16, 64, 56]              10,304
│    └─BatchNorm1d: 2-14                 [16, 64, 56]              [16, 64, 56]              128
│    └─ReLU: 2-15                        [16, 64, 56]              [16, 64, 56]              --
├─Sequential: 1-2                        [16, 64, 56]              [16, 64]                  --
│    └─AdaptiveAvgPool1d: 2-16           [16, 64, 56]              [16, 64, 1]               --
│    └─Flatten: 2-17                     [16, 64, 1]               [16, 64]                  --
├─Sequential: 1-3                        [16, 64]                  [16, 3]                   --
│    └─Linear: 2-18                      [16, 64]                  [16, 32]                  2,080
│    └─ReLU: 2-19                        [16, 32]                  [16, 32]                  --
│    └─Linear: 2-20                      [16, 32]                  [16, 3]                   99
===================================================================================================================
Total params: 20,691
Trainable params: 20,691
Non-trainable params: 0
Total mult-adds (M): 75.28
===================================================================================================================
Input size (MB): 0.46
Forward/backward pass size (MB): 24.88
Params size (MB): 0.08
Estimated Total Size (MB): 25.43
===================================================================================================================
```

## Default training settings
- Loss function: Cross Entropy Loss
- Optimizer: Adam
- Epoch: max epoch = 50, early stopping (patience=5)
- Learning rate: 0.001
- Batch size: 128
- Data sampler: Weighted Random Sampler

## Test environment
- CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10Ghz
- GPU: Tesla T4
- RAM: 512 GB
- OS: Ubuntu 18.04.5 LTS

# License

SPDX-FileCopyrightText: © 2022 Bud-on, Inc. <shjun@bud-on.com, dukyong.yoon@bud-on.com>

SPDX-License-Identifier: Apache-2.0