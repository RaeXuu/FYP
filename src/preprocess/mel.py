
import numpy as np
import librosa

def wav_to_logmel(
    y,
    sr,
    mel_cfg,
    eps=1e-6
):
    """
    将单段音频转成 Log-Mel Spectrogram。
    输入: y (长度固定的片段，例如 8000 点)
    输出: 2D Mel 特征 (n_mels × time_frames)
    """

    # 1. Mel 滤波器
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg.get("win_length", mel_cfg["n_fft"]),
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax", None),
        power=mel_cfg.get("power", 2.0),
    )
    # 2. 转 Log-Mel
    logmel = librosa.power_to_db(mel + eps)

    return logmel


def logmel_fixed_size(
    y,
    sr,
    mel_cfg,
    target_shape
):
    mel = wav_to_logmel(
        y=y,
        sr=sr,
        mel_cfg=mel_cfg
    )

    mel_resized = librosa.util.fix_length(
        mel,
        size=target_shape[1],
        axis=1
    )

    return mel_resized



if __name__ == "__main__":
    """
    测试流程：
    1. 加载 metadata
    2. load_wav
    3. bandpass
    4. segment audio
    5. 转成 log-mel
    """
    import pandas as pd
    from src.preprocess.load_wav import load_wav
    from src.preprocess.filters import apply_bandpass
    from src.preprocess.segment import segment_audio

    df = pd.read_csv("/mnt/d/FypProj/data/metadata1.csv")
    path = df.iloc[0]["filepath"]

    print("测试样本:", path)

    # Step1: load wav
    y, sr = load_wav(path, target_sr=4000)

    # Step2: bandpass
    y = apply_bandpass(y, fs=sr, lowcut=20, highcut=400)

    # Step3: segment
    segments = segment_audio(y, sr=sr, segment_sec=2.0)
    seg = segments[0]  # 取第一段测试

    print("单段长度:", len(seg))

    # Step4: log-mel
    mel = logmel_fixed_size(seg, sr=sr, target_shape=(64, 64))

    print("Log-Mel shape:", mel.shape)
    print("Mel 测试完成 ✅")
