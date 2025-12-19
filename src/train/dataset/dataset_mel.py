# src/train/dataset/dataset_mel.py

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
import pandas as pd

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size
from src.augment.wav_augment import waveform_augment


class HeartSoundMelDataset(Dataset):
    def __init__(self, metadata_path, sr, segment_sec, mel_cfg, transform=None, augment=False):
        self.df = pd.read_csv(metadata_path)
        self.sr = sr
        self.segment_sec = segment_sec
        self.transform = transform
        self.augment = augment
        self.mel_cfg = mel_cfg

        self.valid_labels = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
        self.df = self.df[self.df["label"].isin(self.valid_labels)].reset_index(drop=True)
        self.label_to_idx = {label: i for i, label in enumerate(self.valid_labels)}

        # ✅ samples: [(segment, label_str, fname), ...]
        self.samples = []

        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            label = row["label"]

            # ✅ 优先用 metadata 里的 fname；没有就用 filepath basename
            fname = row["fname"] if "fname" in row else os.path.basename(filepath)

            y, sr = load_wav(filepath, target_sr=self.sr)
            y = apply_bandpass(y, fs=sr, lowcut=20, highcut=400)

            segments = segment_audio(y, sr=sr)
            for seg in segments:
                self.samples.append((seg, label, fname))

        print(f"[Dataset] 只保留的标签: {self.valid_labels}")
        print(f"[Dataset] 总切片数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, label_str, _fname = self.samples[idx]

        if self.augment:
            seg = waveform_augment(seg)

        mel = logmel_fixed_size(
            y=seg,
            sr=self.sr,
            mel_cfg=self.mel_cfg,
            target_shape=(64, 64),
        )

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label_idx = self.label_to_idx[label_str]
        return mel, label_idx

    # ✅ 给 split 用：按 idx 拿到 fname
    def get_fname(self, idx: int) -> str:
        return self.samples[idx][2]


if __name__ == "__main__":
    ds = HeartSoundMelDataset("/mnt/d/FypProj/data/metadata1.csv")
    print("Dataset 总长度（切片数）:", len(ds))
    mel, label = ds[0]
    print("单个 Mel shape:", mel.shape)
    print("标签 index:", label)
    print("Dataset 自测完成 ✅")
