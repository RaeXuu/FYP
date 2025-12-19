# 📥 读取 WAV
# 🎚 重采样到统一采样率（默认 4000 Hz）
# 🔄 幅度归一化（[-1, 1]）
# 📦 批量根据 metadata 读取所有音频
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]

import librosa
import numpy as np
import pandas as pd

def load_wav(filepath, target_sr=None):
    """
    加载 WAV 文件并做基础预处理：
    1. 读取音频
    2. 重采样到 target_sr
    3. 幅度归一化到 [-1, 1]
    """
    if target_sr is None:
        target_sr = data_cfg["sample_rate"]

    # 读取原始音频
    y, sr = librosa.load(filepath, sr=None)

    # 重采样
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 幅度归一化
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr


def batch_load_from_metadata(df, sr=4000):
    """
    根据 metadata DataFrame 批量加载音频。
    返回 list，每个元素包含 {audio, sr, label, filepath}
    """
    audio_items = []

    for idx, row in df.iterrows():
        filepath = row["filepath"]
        y, s = load_wav(filepath, target_sr=sr)

        audio_items.append({
            "audio": y,
            "sr": s,
            "label": row.get("label"),
            "sublabel": row.get("sublabel"),
            "dataset": row.get("dataset"),
            "filepath": filepath
        })

    return audio_items


if __name__ == "__main__":
    # TODO: 测试用：从 metadata1.csv 加载
    df = pd.read_csv("/mnt/d/FypProj/data/metadata1.csv")
    
    # 先加载前三个测试
    audios = batch_load_from_metadata(df.head(3))  

    print("测试加载成功，返回数量:", len(audios))
    print("第一个样本信息:")
    print("  采样率:", audios[0]["sr"])
    print("  音频长度:", len(audios[0]["audio"]))