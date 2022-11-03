# nia_vent_asynchrony
- 2022년 인공지능 학습용 데이터 구축 사업의 일환으로, 인공호흡기 파형에서 정상과 노이즈, 부조화를 분류하는 모델을 개발함
- 환자와 인공호흡기의 호흡 주기가 맞지 않으면 환자의 불편감이 초래되며, 폐손상의 악화로 
인해 예후가 좋지 않음. 따라서 지속적으로 기계환기 그래프 파형을 관찰하여 비동시성을 
감시해야 함. 그러나 이에 따라 전문의료인의 노동력이 소모되므로 호흡 주기 부조화를 자동적으로 감지하는 모델은 불필요한 의료 비용을 감소시킬 수 있음

## 1. Data Preprocessing
> python preprocess.py --config config/preprocess_config.yml

csv형식의 waveform 파일을 입력받아 1분 단위의 instance로 변환한 뒤 pickle 파일로 저장함

## 2. Training
> python train.py --config config/train_config.yml

전처리가 완료된 pickle 파일과 라벨 파일을 입력받아 훈련셋(training set)과 검증셋(validation set)으로 나눈 뒤 정상, 부조화, 노이즈로 분류하는 모델을 훈련함

## 3. Test
> python test.py --config config/test_config.yml

훈련에 사용되지 않은 테스트셋으로 훈련된 모델을 평가함