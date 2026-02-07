import os
import sys

# === 添加项目根目录到路径 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml
from pathlib import Path
from pprint import pprint


# =========================
# Experiment setting
# =========================
FEATURE_TYPE = "mel"
# options: "mel", "wavelet", "bicoherence"


# === 保证路径正确 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# === 再导入模块 ===
if FEATURE_TYPE == "mel":
    from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
elif FEATURE_TYPE == "wavelet":
    from src.train.dataset.dataset_wavelet import HeartSoundWaveletDataset as Dataset
elif FEATURE_TYPE == "bicoherence":
    from src.train.dataset.dataset_bicoherence import HeartSoundBicoherenceDataset as Dataset
else:
    raise ValueError(f"Unknown FEATURE_TYPE: {FEATURE_TYPE}")

from src.model.lightweight_cnn import LightweightCNN


# === 训练参数（保持与你现在 train_lightweight.py 一致） ===
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

# === Split（新增：为了 test）===
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for mels, labels in tqdm(dataloader, desc="Training", leave=False):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(mels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, desc="Eval"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    all_preds = []
    all_labels = []

    for mels, labels in tqdm(dataloader, desc=desc, leave=False):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        outputs = model(mels)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * mels.size(0)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss_avg = running_loss / total
    acc = correct / total

    return loss_avg, acc, np.array(all_labels), np.array(all_preds)


def print_metrics(title, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{title}] Confusion Matrix:")
    print(cm)

    print(f"\n[{title}] Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    ))


def main():
    print(f"Using device: {DEVICE}")

    # =========================
    # Load config
    # =========================
    CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print(f"[CONFIG] Using config.yaml at:\n{CONFIG_PATH}")
    pprint(cfg)
    print("=" * 60 + "\n")

    # =========================
    # Paths
    # =========================
    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata1.csv")

    # =========================
    # Dataset config (fail-fast)
    # =========================
    data_cfg = cfg["data"]
    mel_cfg = cfg["mel"]

    sr = data_cfg["sample_rate"]
    segment_sec = data_cfg["segment_length"]

    # =========================
    # Dataset (单个 full dataset，然后三分)
    # =========================
    full_dataset = Dataset(
        metadata_path=metadata_path,
        sr=sr,
        segment_sec=segment_sec,
        mel_cfg=mel_cfg,
        augment=False,
    )

    n_total = len(full_dataset)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    metadata_df = pd.read_csv(metadata_path)

    # =========================
    # Group split by fname (CORRECT: dataset-level)
    # =========================
    rng = np.random.RandomState(SPLIT_SEED)

    # 1️⃣ 从 dataset 中拿每个 sample 对应的 fname
    all_fnames = [full_dataset.get_fname(i) for i in range(len(full_dataset))]

    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_rec = len(unique_fnames)
    n_train_rec = int(TRAIN_RATIO * n_rec)
    n_val_rec = int(VAL_RATIO * n_rec)

    train_rec_ids = set(unique_fnames[:n_train_rec])
    val_rec_ids   = set(unique_fnames[n_train_rec:n_train_rec + n_val_rec])
    test_rec_ids  = set(unique_fnames[n_train_rec + n_val_rec:])

    train_indices = []
    val_indices   = []
    test_indices  = []

    # 2️⃣ 对「每一个切片 sample」分配集合
    for idx, fname in enumerate(all_fnames):
        if fname in train_rec_ids:
            train_indices.append(idx)
        elif fname in val_rec_ids:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(full_dataset, val_indices)
    test_ds  = torch.utils.data.Subset(full_dataset, test_indices)


    print("[Split]")
    print(f"  total = {n_total}")
    print(f"  train = {len(train_ds)}")
    print(f"  val   = {len(val_ds)}")
    print(f"  test  = {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # === 模型、loss、优化器 ===
    model = LightweightCNN(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    class_names = ['artifact', 'extrahls', 'extrastole', 'murmur', 'normal']

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, desc="Validation")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # === Validation metrics（保持与你现在脚本一致风格）===
        print_metrics("Validation", y_true, y_pred, class_names)

        # === 保存最优模型 ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ New best model saved! Acc={best_acc:.4f}")

    print(f"\nTraining finished. Best Val Acc={best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    # =========================
    # TEST: load best model and evaluate once
    # =========================
    print("\n" + "=" * 60)
    print("[TEST] Loading best model and evaluating on test set...")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_loss, test_acc, y_true_t, y_pred_t = evaluate(model, test_loader, criterion, desc="Test")

    print(f"\nTest  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
    print_metrics("Test", y_true_t, y_pred_t, class_names)


if __name__ == "__main__":
    main()
