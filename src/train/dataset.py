import os, sys

# === 在最前面添加路径 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size


class HeartSoundDataset(Dataset):
    """
    心音数据集（训练用）
    WAV → 滤波 → 切片 → Log-Mel(64×64) → Tensor
    只保留 5 类：
        artifact, extrahls, extrastole, murmur, normal
    """

    def __init__(self, metadata_path, sr=4000, segment_sec=2.0, n_mels=64, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.sr = sr
        self.segment_sec = segment_sec
        self.n_mels = n_mels
        self.transform = transform

        # === 显式保留 5 个有标签的类别 ===
        self.valid_labels = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
        self.df = self.df[self.df["label"].isin(self.valid_labels)].reset_index(drop=True)

        # 固定标签映射
        self.label_to_idx = {label: i for i, label in enumerate(self.valid_labels)}

        self.samples = []   # [(segment, label_str), ...]

        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            label = row["label"]

            y, sr = load_wav(filepath, target_sr=self.sr)
            y = apply_bandpass(y, fs=sr, lowcut=20, highcut=400)

            segments = segment_audio(y, sr=sr, segment_sec=self.segment_sec)

            for seg in segments:
                self.samples.append((seg, label))

        print(f"[Dataset] 只保留的标签: {self.valid_labels}")
        print(f"[Dataset] 总切片数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label_str = self.samples[idx]

        mel = logmel_fixed_size(seg, sr=self.sr, target_shape=(64, 64))
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

        label_idx = self.label_to_idx[label_str]

        return mel, label_idx


if __name__ == "__main__":
    ds = HeartSoundDataset("/mnt/d/FypProj/data/metadata1.csv")
    print("Dataset 总长度（切片数）:", len(ds))
    mel, label = ds[0]
    print("单个 Mel shape:", mel.shape)
    print("标签 index:", label)
    print("Dataset 自测完成 ✅")
