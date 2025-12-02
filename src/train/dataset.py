# 从 metadata 读取所有样本
# 对每个样本：
# load_wav
# bandpass
# segment
# mel (64×64)
# 输出 PyTorch 能直接训练的 (mel_tensor, label)
# 自动把 numpy → torch tensor
# label 自动映射为 int index

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# 引入前面步骤的 preprocess 模块
from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size


class HeartSoundDataset(Dataset):
    """
    心音数据集（训练用）
    自动执行：
    WAV → 滤波 → 切片 → Log-Mel(64×64) → Tensor
    """

    def __init__(self, metadata_path, sr=4000, segment_sec=2.0, n_mels=64, transform=None):
        """
        metadata_path: metadata.csv
        segment_sec: 每个切片长度（秒）
        """
        self.df = pd.read_csv(metadata_path)
        self.sr = sr
        self.segment_sec = segment_sec
        self.n_mels = n_mels
        self.transform = transform

        # 标签映射，例如：
        # artifact → 0
        # normal → 1
        # crackle → 2
        # wheeze → 3
        self.label_to_idx = {label: i for i, label in enumerate(sorted(self.df["label"].unique()))}

        # 为了训练，每个切片都要成为一个样本
        self.samples = []   # [(filepath, label), ...]

        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            label = row["label"]

            # 先 load + filter + segment（不生成 Mel）
            y, sr = load_wav(filepath, target_sr=self.sr)
            y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)

            segments = segment_audio(y, sr=sr, segment_sec=self.segment_sec)

            for seg in segments:
                self.samples.append((seg, label))

        print(f"[Dataset] 总切片数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label = self.samples[idx]

        # 生成 log-mel (64×64)
        mel = logmel_fixed_size(seg, sr=self.sr, target_shape=(64, 64))

        # numpy → torch tensor，增加 channel 维度
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

        label_idx = self.label_to_idx[label]

        return mel, label_idx


if __name__ == "__main__":
    # 简单自测：取一个 batch 看 shape 是否正常
    ds = HeartSoundDataset("/mnt/d/FypProj/data/metadata1.csv")
    print("Dataset 总长度（切片数）:", len(ds))

    mel, label = ds[0]
    print("单个 Mel shape:", mel.shape)  # (1, 64, 64)
    print("标签 index:", label)
    print("Dataset 自测完成 ✅")