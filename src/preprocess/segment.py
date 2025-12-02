
import numpy as np
from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass

def segment_audio(y, sr=4000, segment_sec=2.0):
    """
    将音频切成固定长度（秒）的片段。
    不足补零，超出则切多段。
    """
    seg_len = int(sr * segment_sec)
    if len(y) == 0:
        return []

    segments = []

    for start in range(0, len(y), seg_len):
        end = start + seg_len
        seg = y[start:end]

        # 长度不足 → 补零 zero-padding
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode='constant')

        segments.append(seg)

    return segments


if __name__ == "__main__":
    import pandas as pd

    # 读取 metadata
    df = pd.read_csv("/mnt/d/FypProj/data/metadata1.csv")
    path = df.iloc[0]["filepath"]

    print("测试样本:", path)

    # Step1 读取 + Step2 滤波
    y, sr = load_wav(path, target_sr=4000)
    y = apply_bandpass(y, fs=sr, lowcut=20  , highcut=400)

    # Step3 切片
    segments = segment_audio(y, sr=sr, segment_sec=2.0)

    print("切片数量:", len(segments))
    print("单个片段长度:", len(segments[0]))
    print("理论应为:", sr * 2)
    print("切片测试完成 ✅")
