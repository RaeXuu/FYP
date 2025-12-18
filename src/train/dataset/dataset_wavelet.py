import os, sys

# === 项目根路径注入（与你原 dataset 完全一致）===
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
from src.preprocess.wavelet import cwt_fixed_size
from src.augment.wav_augment import waveform_augment

class HeartSoundWaveletDataset(Dataset):
    """
    心音数据集（Wavelet / CWT 版本）

    WAV → 滤波 → 切片 → Wavelet Scalogram → Tensor

    保留 5 类：
        artifact, extrahls, extrastole, murmur, normal
    """

    def __init__(
        self,
        metadata_path,
        sr=4000,
        segment_sec=2.0,
        out_size=64,          # 与 mel / bicoherence 对齐
        transform=None,
        augment=False,
    ):
        self.df = pd.read_csv(metadata_path)
        self.sr = sr
        self.segment_sec = segment_sec
        self.out_size = out_size
        self.transform = transform
        self.augment = augment

        # === 显式保留 5 类（与 mel / bicoherence 完全一致）===
        self.valid_labels = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
        self.df = self.df[self.df["label"].isin(self.valid_labels)].reset_index(drop=True)

        self.label_to_idx = {label: i for i, label in enumerate(self.valid_labels)}

        self.samples = []   # [(segment, label_str), ...]

        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            label = row["label"]

            # 1️⃣ 读 wav
            y, sr = load_wav(filepath, target_sr=self.sr)

            # 2️⃣ 带通滤波（心音）
            y = apply_bandpass(y, fs=sr, lowcut=20, highcut=400)

            # 3️⃣ 切成 2s 段
            segments = segment_audio(y, sr=sr, segment_sec=self.segment_sec)

            for seg in segments:
                self.samples.append((seg, label))

        print("[WaveletDataset] 使用特征: CWT scalogram")
        print("[WaveletDataset] 输出尺寸:", out_size, "x", out_size)
        print("[WaveletDataset] 保留标签:", self.valid_labels)
        print("[WaveletDataset] 总切片数:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label_str = self.samples[idx]

        if self.augment:
            seg = waveform_augment(seg)

        # === 核心区别：Wavelet 特征 ===
        wav_feat = cwt_fixed_size(
            seg,
            sr=self.sr,
            out_size=self.out_size,
        )

        wav_feat = torch.tensor(wav_feat, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        if self.transform:
            wav_feat = self.transform(wav_feat)

        label_idx = self.label_to_idx[label_str]

        return wav_feat, label_idx


# ==========================
# 自测（和你另外两个 dataset 风格完全一致）
# ==========================
if __name__ == "__main__":
    ds = HeartSoundWaveletDataset(
        "/mnt/d/FypProj/data/metadata1.csv",
        out_size=64,
    )

    print("Dataset 总长度（切片数）:", len(ds))
    x, y = ds[0]
    print("单个样本 shape:", x.shape)
    print("标签 index:", y)
    print("Dataset 自测完成 ✅")
