import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import sounddevice as sd
import collections
import time
import bc_resnet_model

import soundfile as sf
import os

# ================================
# 설정
# ================================

MODEL_PATH  = './train_model/best_horiya2_11.pt'
SCALE       = 2
N_CLASS     = 2
CLASSES     = ['non-wakeword', 'wakeword']
THRESHOLD   = 0.9      # 웨이크워드 감지 임계값

SAMPLE_RATE = 16000
WINDOW_SEC  = 1         # 추론 윈도우 크기 (초)
STEP_SEC    = 0.03      # 슬라이딩 스텝 (초마다 추론)

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
STEP_SAMPLES   = int(SAMPLE_RATE * STEP_SEC)

EPS = 1e-9 # Epsilon

SAVE_DIR = "./detections"
COOLDOWN_SEC = 2.0  # 2초 동안 재저장 금지
last_save_time = 0

os.makedirs(SAVE_DIR, exist_ok=True)

# ================================
# 모델 로드
# ================================
print("모델 로드 중...")
device = torch.device('cpu')

model = bc_resnet_model.BcResNetModel(
    n_class=N_CLASS,
    scale=SCALE,
    use_subspectral=True
)
model.head_conv = nn.Conv2d(32 * SCALE, N_CLASS, kernel_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("모델 로드 완료!\n")


# ================================
# 전처리
# ================================
to_mel = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    f_max=8000,
    n_mels=40
)

def preprocess(audio_np):
    """numpy 배열 → 로그 멜 스펙트로그램"""
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, samples)
    log_mel = (to_mel(waveform) + EPS).log2()
    return log_mel.unsqueeze(0)     # (1, 1, 40, T)


# ================================
# 슬라이딩 윈도우 버퍼
# ================================
audio_buffer = collections.deque(maxlen=WINDOW_SAMPLES)
# 버퍼 초기화 (무음으로 채움)
audio_buffer.extend(np.zeros(WINDOW_SAMPLES))

last_inference_time = time.time()
detected_count      = 0


def audio_callback(indata, frames, time_info, status):
    """마이크에서 오디오 청크를 받을 때마다 호출"""
    global last_inference_time, detected_count, last_save_time

    if status:
        print(f"[WARNING] {status}")

    # 모노로 변환 후 버퍼에 추가
    audio_chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    audio_buffer.extend(audio_chunk)

    # STEP_SEC마다 추론
    now = time.time()
    if now - last_inference_time < STEP_SEC:
        return
    last_inference_time = now

    # 버퍼 → numpy 배열
    window = np.array(audio_buffer, dtype=np.float32)

    # 추론
    with torch.no_grad():
        spec   = preprocess(window)
        output = model(spec)
        probs  = F.softmax(output.squeeze(), dim=0).numpy()

    wakeword_prob = probs[CLASSES.index('wakeword')]
    non_wakeword_prob  = probs[CLASSES.index('non-wakeword')]

    # 결과 출력
    bar_w = "█" * int(wakeword_prob * 20)
    bar_n = "█" * int(non_wakeword_prob  * 20)

    if wakeword_prob >= THRESHOLD:
        now_time = time.time()

        if now_time - last_save_time > COOLDOWN_SEC:
            last_save_time = now_time
            detected_count += 1

            filename = os.path.join(
                SAVE_DIR,
                f"detection_{int(now_time)}.wav"
            )

            sf.write(filename, window, SAMPLE_RATE)

            # 현재 줄 끝내고 감지 메시지 새 줄에 출력
            print(f"\n🔔 웨이크워드 감지! ({wakeword_prob*100:.1f}%)")
            # 다음 업데이트는 자동으로 새 줄에서 \r로 시작
            print(f"저장됨: {filename}")

    else:
        print(
            f"\rwakeword {wakeword_prob*100:5.1f}% {bar_w:<20} | "
            f"non-wakeword {non_wakeword_prob*100:5.1f}% {bar_n:<20}",
            end='', flush=True
        )


# ================================
# 메인
# ================================
if __name__ == "__main__":
    print(f"샘플레이트 : {SAMPLE_RATE}Hz")
    print(f"윈도우 크기: {WINDOW_SEC}초")
    print(f"추론 간격  : {STEP_SEC}초마다")
    print(f"감지 임계값: {THRESHOLD*100:.0f}%")
    print(f"\n마이크 장치 목록:")
    print(sd.query_devices())
    print(f"\n[Ctrl+C로 종료]")
    print("="*45)
    print("듣는 중...")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=STEP_SAMPLES,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n\n종료! 총 감지 횟수: {detected_count}회")