# WakeWord BC-ResNet
이 프로젝트는 BC-ResNet 모델을 재학습하여 **“호리야” WakeWord 감지**용으로 사용하도록 만든 프로젝트입니다.  
원래 영어 단어 30~35개 분류용으로 사전학습된 모델을 **2개 클래스 (wakeword / non-wakeword)**로 재학습하였습니다.

---

# 1. 환경 세팅

### Conda 환경 생성
```bash
conda create -n bcresnet python=3.10
conda activate bcresnet
```
### 패키지 설치
```bash
pip install -r requirements.txt
```

# 2. 데이터 폴더 구조
`data/` 폴더는 DVC로 관리되며, 학습과 테스트에 필요한 모든 음원이 포함되어 있습니다.
### 상위 폴더
- `raw/` : 1초로 자르기 전 원본 음원
  - `kids_voice/` : 아이들이 발화한 문장
  - `noise_bg/` : 배경음, 사람 목소리 없음
  - `noise_speech/` : 사람 음성이 포함된 배경 소리
- `train/` : 모델 입력에 맞게 1초 단위로 잘라 학습용으로 준비
  - `AI_Hub/` : 다양한 단어를 발화한 음원, JSON 메타데이터 포함
    - `wav_bg_noise/` : noise_bg_cut과 혼합한 음원
    - `wav_speech_noise/` : noise_speech_cut과 혼합한 음원
    - 기타 하위 폴더는 비슷한 방식으로 믹싱된 데이터
  - `hard_negative_speech/` : 오탐 데이터 수집용
  - `noise_*` : WakeWord가 아닌 음원, 배경 소음/대화 소음 포함
  - `wakeword_*` : WakeWord 관련 음원, 청정/노이즈/증강 데이터 포함


# 3. 학습 (Fine-tuning)

본 프로젝트는 사전학습 모델 `model-sc-2.pt`를 기반으로 Wakeword 검출 모델로 재학습(fine-tuning)합니다.

### 1. 사전 준비
- 사전학습 모델 위치: `example_model/model-sc-2.pt`
- 학습용 데이터: `data/train/` (1초 단위로 분할된 음원)
- 학습 스크립트 위치: `bc_resnet_re9ulus/finetune2.py`

### 2. 실행 방법
```bash
cd bc_resnet_re9ulus
python finetune2.py
```

### 3. 스크립트 설명
- `finetune2.py` :
  - 사전학습 모델 로드(35개 클래스)
  - 출력 레이어를 Wakeword/Non-Wakeword용 2개 클래스로 교체
  - 백본 freeze 여부 선택 가능 (현재 기본: True → head만 학습)
  - 데이터 로드 및 전처리(custom_data2.py 사용)
  - 학습, 검증, 최고 성능 모델 저장
- `bc_resnet_model.py` :
  - BC-ResNet 모델 구조 정의
  - SubSpectralNorm, Residual Block, Transition Block 포함
- `custom_data2.py` :
  - 학습/검증용 데이터셋 구축
  - 1초 단위 패딩/트리밍, Mel Spectrogram 변환
  - 파일 리스트 분할(train/validation)
- `apply.py` :
  - 학습 완료 모델을 wav 파일에 적용
  - 예측 확률과 클래스 반환

### 4. 학습 결과
- 학습 후 모델은 `best_horiya2_<epoch>.pt` 형태로 저장
- Val Accuracy가 가장 높은 모델이 베스트 모델

# 4. 실시간 테스트 (Realtime Inference)
본 스크립트는 학습 완료된 모델을 사용하여 마이크 입력에서 실시간으로 Wakeword를 감지합니다.

### 1. 사전 준비
- 모델 경로: `./train_model/모델.pt`
- 샘플레이트: 16 kHz
- 추론 윈도우: 1초 (슬라이딩 스텝 0.03초)
- 감지 임계값: 0.8 이상일 때 웨이크워드 감지

### 2. 실행 방법
```bash
cd bc_resnet_re9ulus
python realtime_test2.py
```

### 3. 스크립트 설명
- 모델 로드
  - `BcResNetModel` 구조를 사용하여 학습 완료 모델 로드
  - 출력 레이어는 Wakeword/Non-Wakeword 2개 클래스
- 전처리
  - 입력 오디오를 Mel Spectrogram으로 변환 후 로그 스케일 적용
- 슬라이딩 윈도우 추론
  - 1초 윈도우를 0.03초 단위로 슬라이딩하며 추론
  - Wakeword 확률이 임계값 이상이면 감지 메시지 출력 및 WAV 파일 저장
  - 재저장 방지 쿨다운: 2초
- 실시간 스트리밍
  - `sounddevice`를 이용하여 마이크 입력 수집
  - `audio_callback`에서 추론 수행 및 출력
- 결과 저장
  - 감지된 음원은 `./detections/` 폴더에 저장
  - 감지 횟수 카운트 및 로그 출력