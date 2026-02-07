# # src/train/dataset/dataset_mel.py

# import os, sys
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# import torch
# from torch.utils.data import Dataset
# import pandas as pd

# from src.preprocess.load_wav import load_wav
# from src.preprocess.filters import apply_bandpass
# from src.preprocess.segment import segment_audio
# from src.preprocess.mel import logmel_fixed_size
# from src.augment.wav_augment import waveform_augment


# class HeartSoundMelDataset(Dataset):
#     def __init__(self, metadata_path, sr, segment_sec, mel_cfg, transform=None, augment=False):
#         self.df = pd.read_csv(metadata_path)
#         self.sr = sr
#         self.segment_sec = segment_sec
#         self.transform = transform
#         self.augment = augment
#         self.mel_cfg = mel_cfg

#         self.valid_labels = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
#         self.df = self.df[self.df["label"].isin(self.valid_labels)].reset_index(drop=True)
#         self.label_to_idx = {label: i for i, label in enumerate(self.valid_labels)}

#         # ✅ samples: [(segment, label_str, fname), ...]
#         self.samples = []

#         for _, row in self.df.iterrows():
#             filepath = row["filepath"]
#             label = row["label"]

#             # ✅ 优先用 metadata 里的 fname；没有就用 filepath basename
#             fname = row["fname"] if "fname" in row else os.path.basename(filepath)

#             y, sr = load_wav(filepath, target_sr=self.sr)
#             y = apply_bandpass(y, fs=sr, lowcut=20, highcut=400)

#             segments = segment_audio(y, sr=sr)
#             for seg in segments:
#                 self.samples.append((seg, label, fname))

#         print(f"[Dataset] 只保留的标签: {self.valid_labels}")
#         print(f"[Dataset] 总切片数: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         seg, label_str, _fname = self.samples[idx]

#         if self.augment:
#             seg = waveform_augment(seg)

#         mel = logmel_fixed_size(
#             y=seg,
#             sr=self.sr,
#             mel_cfg=self.mel_cfg,
#             target_shape=(64, 64),
#         )

#         mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
#         label_idx = self.label_to_idx[label_str]
#         return mel, label_idx

#     # ✅ 给 split 用：按 idx 拿到 fname
#     def get_fname(self, idx: int) -> str:
#         return self.samples[idx][2]


# if __name__ == "__main__":
#     ds = HeartSoundMelDataset("/mnt/d/FypProj/data/metadata1.csv")
#     print("Dataset 总长度（切片数）:", len(ds))
#     mel, label = ds[0]
#     print("单个 Mel shape:", mel.shape)
#     print("标签 index:", label)
#     print("Dataset 自测完成 ✅")


import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size
from src.augment.wav_augment import waveform_augment

class HeartSoundMelDataset(Dataset):
    def __init__(self, metadata_path, sr, segment_sec, mel_cfg, transform=None, augment=False):
        # 1. 直接读取生成的 metadata_physionet.csv
        self.df = pd.read_csv(metadata_path)
        self.sr = sr
        self.segment_sec = segment_sec
        self.transform = transform
        self.augment = augment
        self.mel_cfg = mel_cfg

        # 2. ✅ 适配二分类：PhysioNet 2016 标签已经是 0 和 1，不再需要 valid_labels 列表
        # 直接使用 int(row["label"]) 即可

        self.samples = []

        print(f"[Dataset] 开始预处理音频并切片...")
        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            label = int(row["label"])  # 0: Normal, 1: Abnormal
            fname = row["fname"]

            # 加载并滤波
            y, _ = load_wav(filepath, target_sr=self.sr)
            y = apply_bandpass(y, fs=self.sr, lowcut=25, highcut=400)

            # 切片处理
            segments = segment_audio(y, sr=self.sr)
            for seg in segments:
                self.samples.append((seg, label, fname))

        print(f"[Dataset] 总音频数: {len(self.df)} | 总切片数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label, _fname = self.samples[idx]

        # 数据增强
        if self.augment:
            seg = waveform_augment(seg)

        # 转换为 Log-Mel 频谱
        # target_shape 这里可以根据你 YAML 里的 n_mels 动态调整
        # 假设你想要 (32, 64) 这种形状
        mel = logmel_fixed_size(
            y=seg,
            sr=self.sr,
            mel_cfg=self.mel_cfg,
            target_shape=(self.mel_cfg["n_mels"], 64), 
        )

        # 转为 Tensor 形状: [1, n_mels, time]
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return mel, label

    def get_fname(self, idx: int) -> str:
        return self.samples[idx][2]


if __name__ == "__main__":
    # 测试代码：使用真实的配置文件参数
    import yaml
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # ✅ 指向正确的文件
    METADATA_PATH = "/mnt/d/FypProj/data/metadata_physionet.csv"
    
    if os.path.exists(METADATA_PATH):
        ds = HeartSoundMelDataset(
            metadata_path=METADATA_PATH,
            sr=config["data"]["sample_rate"],
            segment_sec=config["data"]["segment_length"],
            mel_cfg=config["mel"]
        )
        print("Dataset 加载成功！")
        mel, label = ds[0]
        print(f"单个 Mel shape: {mel.shape}")
        print(f"标签值: {label}")
    else:
        print("请检查路径：", METADATA_PATH)