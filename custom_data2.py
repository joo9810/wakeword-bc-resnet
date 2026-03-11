import os
import math
import random
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader


EPS = 1e-9
SAMPLE_RATE = 16000
N_CLASS = 2  # non-wakeword, wakeword

LABELS = ['non-wakeword', 'wakeword']
_label_to_idx = {label: i for i, label in enumerate(LABELS)}
_idx_to_label = {i: label for i, label in enumerate(LABELS)}


# ================================
# 경로 설정 (여기만 수정!)
# ================================
WAKEWORD_ROOT1  = '/home/isol/work/data/train/wakeword'
WAKEWORD_ROOT2  = '/home/isol/work/data/train/wakeword_bg_noisy'
WAKEWORD_ROOT3  = '/home/isol/work/data/train/wakeword_speech_noisy'
WAKEWORD_ROOT4  = '/home/isol/work/data/train/wakeword_both_noisy'
WAKEWORD_ROOT5  = '/home/isol/work/data/train/wakeword_vad_noisy'

WAKEWORD_ROOT6  = '/home/isol/work/data/train/wakeword_bg_noisy_up'
WAKEWORD_ROOT7  = '/home/isol/work/data/train/wakeword_speech_noisy_up'
WAKEWORD_ROOT8  = '/home/isol/work/data/train/wakeword_both_noisy_up'
WAKEWORD_ROOT9  = '/home/isol/work/data/train/wakeword_vad_noisy_up'


AI_HUB_ROOT1   = '/home/isol/work/data/train/AI_Hub/wav'   # 가람아, 검은콩 등
AI_HUB_ROOT2   = '/home/isol/work/data/train/AI_Hub/wav_bg_noise'
AI_HUB_ROOT3   = '/home/isol/work/data/train/AI_Hub/wav_speech_noise'
AI_HUB_ROOT4   = '/home/isol/work/data/train/AI_Hub/wav_vad_noise'
AI_HUB_ROOT5   = '/home/isol/work/data/train/AI_Hub/wav_both_noise'

NOISE_SPEECH_ROOT = '/home/isol/work/data/train/noise_speech_cut'
NOISE_BOTH_ROOT = '/home/isol/work/data/train/noise_both_cut'
HARD_NEGATIVE_ROOT = '/home/isol/work/data/train/hard_negative_speech'


SILENCE_ROOT   = '/home/isol/work/data/train/noise_bg_cut'
VAD_SILENCE_ROOT = '/home/isol/work/data/train/noise_vad_silence'

UNKNOWN_SAMPLE_PER_FOLDER = 500  # 폴더당 최대 샘플 수
VAL_RATIO = 0.1
SEED = 42

wakeword_root_list = [WAKEWORD_ROOT1, WAKEWORD_ROOT2, WAKEWORD_ROOT3, WAKEWORD_ROOT4, WAKEWORD_ROOT5, WAKEWORD_ROOT6, WAKEWORD_ROOT7, WAKEWORD_ROOT8, WAKEWORD_ROOT9]
aihub_root_list    = [AI_HUB_ROOT1, AI_HUB_ROOT2, AI_HUB_ROOT3, AI_HUB_ROOT4, AI_HUB_ROOT5]
unknown_root_list  = [NOISE_SPEECH_ROOT, NOISE_BOTH_ROOT, HARD_NEGATIVE_ROOT]
silence_root_list  = [SILENCE_ROOT, VAD_SILENCE_ROOT]

def label_to_idx(label):
    return _label_to_idx[label]


def idx_to_label(idx):
    return _idx_to_label[idx]


# ================================
# 파일 수집
# ================================
def get_wavs(folder):
    if not os.path.exists(folder):
        print(f"[WARNING] 폴더 없음: {folder}")
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.wav')
    ]


def build_file_list(val_ratio=VAL_RATIO, seed=SEED):
    random.seed(seed)

    # 1. wakeword
    wakeword_files = []
    for wakeword_root in wakeword_root_list:
        wakeword_files.extend(get_wavs(wakeword_root))
    print(f"wakeword: {len(wakeword_files)}개")

    # 2. unknown (AI_Hub 각 폴더에서 최대 100개씩)
    unknown_files = []

    for aihub_root in aihub_root_list:
        all_words = sorted(os.listdir(aihub_root))
        for word in all_words:
            word_path = os.path.join(aihub_root, word)
            if os.path.isdir(word_path):
                wavs = get_wavs(word_path)
                sample_size = min(UNKNOWN_SAMPLE_PER_FOLDER, len(wavs))
                unknown_files.extend(random.sample(wavs, sample_size))

    for unknown_root in unknown_root_list:
        unknown_files.extend(get_wavs(unknown_root))
    print(f"unknown ({len(all_words)}개 단어 × {UNKNOWN_SAMPLE_PER_FOLDER}개): {len(unknown_files)}개")

    # 3. silence
    silence_files = []
    for silence_root in silence_root_list:
        silence_files.extend(get_wavs(silence_root))
    print(f"silence: {len(silence_files)}개")

    # 4. 9:1 분할
    random.shuffle(wakeword_files)
    random.shuffle(unknown_files)
    random.shuffle(silence_files)

    def split(files):
        n = int(len(files) * val_ratio)
        return files[n:], files[:n]

    ww_train,  ww_val  = split(wakeword_files)
    unk_train, unk_val = split(unknown_files)
    sil_train, sil_val = split(silence_files)

    train_files = (
        [(f, _label_to_idx['wakeword']) for f in ww_train] +
        [(f, _label_to_idx['non-wakeword'])  for f in unk_train] +
        [(f, _label_to_idx['non-wakeword'])  for f in sil_train]
    )
    val_files = (
        [(f, _label_to_idx['wakeword']) for f in ww_val] +
        [(f, _label_to_idx['non-wakeword'])  for f in unk_val] +
        [(f, _label_to_idx['non-wakeword'])  for f in sil_val]
    )

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(f"\n[Training]")
    print(f"  wakeword: {len(ww_train)}개")
    print(f"  non-wakeword:  {len(unk_train)+len(sil_train)}개")
    print(f"  합계:     {len(train_files)}개")
    print(f"\n[Validation]")
    print(f"  wakeword: {len(ww_val)}개")
    print(f"  non-wakeword:  {len(unk_val)+len(sil_val)}개")
    print(f"  합계:     {len(val_files)}개\n")

    return train_files, val_files


# ================================
# 전처리 (re9ulus 방식 그대로!)
# ================================
def prepare_wav(waveform, sample_rate):
    """사전학습 모델과 동일한 전처리"""
    if sample_rate != SAMPLE_RATE:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    to_mel = transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        f_max=8000,
        n_mels=40
    )
    log_mel = (to_mel(waveform) + EPS).log2()
    return log_mel


# ================================
# 데이터셋
# ================================
class CustomAudioDataset(Dataset):
    def __init__(self, file_list, is_training=False):
        self.file_list   = file_list
        self.is_training = is_training
        self.to_mel = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            f_max=8000,
            n_mels=40
        )
        self.target_len = SAMPLE_RATE  # 1초 (re9ulus 기본값)

    def __len__(self):
        return len(self.file_list)

    def _pad_or_trim(self, waveform):
        length = waveform.shape[1]
        if length > self.target_len:
            waveform = waveform[:, :self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def _shift_augment(self, waveform):
        shift = random.randint(0, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def __getitem__(self, idx):
        wav_path, label = self.file_list[idx]

        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")
            return torch.zeros(1, 40, 101), label

        # 스테레오 → 모노
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != SAMPLE_RATE:
            waveform = transforms.Resample(sr, SAMPLE_RATE)(waveform)

        # 패딩/자르기
        waveform = self._pad_or_trim(waveform)

        # # 학습 중 shift 증강
        # if self.is_training:
        #     waveform = self._shift_augment(waveform)

        # 로그 멜 스펙트로그램 (사전학습 모델과 동일한 방식)
        log_mel = (self.to_mel(waveform) + EPS).log2()

        return log_mel, label


def collate_fn(batch):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label)
    tensors = torch.nn.utils.rnn.pad_sequence(
        [t.permute(2, 1, 0) for t in tensors],
        batch_first=True
    )
    tensors = tensors.permute(0, 3, 2, 1)
    targets = torch.LongTensor(targets)
    return tensors, targets
