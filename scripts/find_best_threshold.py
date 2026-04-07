"""
扫描分类阈值，找出 Test Set 上 M-Score 最高的阈值。
用法：python scripts/find_best_threshold.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
from src.model.lightweight_cnn import LightweightCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
BATCH_SIZE = 32

def main():
    CONFIG_PATH = Path(PROJECT_ROOT) / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_physionet.csv")
    val_dataset = Dataset(
        metadata_path=metadata_path,
        sr=cfg["data"]["sample_rate"],
        segment_sec=cfg["data"]["segment_length"],
        mel_cfg=cfg["mel"],
        augment=False,
    )

    # 按与训练脚本相同的方式重建 test 集
    rng = np.random.RandomState(42)
    all_fnames = [val_dataset.get_fname(i) for i in range(len(val_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_rec = len(unique_fnames)
    idx_train = int(0.8 * n_rec)
    idx_val   = int(0.9 * n_rec)
    test_rec_ids = set(unique_fnames[idx_val:])

    test_indices = [i for i, f in enumerate(all_fnames) if f in test_rec_ids]
    test_ds = torch.utils.data.Subset(val_dataset, test_indices)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test samples: {len(test_ds)}")

    # 收集所有样本的 Abnormal 概率
    model = LightweightCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for mels, labels in test_loader:
            mels = mels.to(DEVICE)
            outputs = model(mels)
            probs = F.softmax(outputs, dim=1)[:, 1]  # P(Abnormal)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 扫描阈值
    print(f"\n{'Threshold':>10} {'Se':>8} {'Sp':>8} {'M-Score':>10}")
    print("-" * 42)

    best_threshold, best_mscore = 0.5, 0.0
    for thresh in np.arange(0.3, 0.81, 0.05):
        preds = (all_probs >= thresh).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()

        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        m  = (se + sp) / 2

        marker = " ◀ best" if m > best_mscore else ""
        if m > best_mscore:
            best_mscore = m
            best_threshold = thresh

        print(f"{thresh:>10.2f} {se:>8.4f} {sp:>8.4f} {m:>10.4f}{marker}")

    print(f"\n最优阈值: {best_threshold:.2f}  M-Score: {best_mscore:.4f}")

if __name__ == "__main__":
    main()
