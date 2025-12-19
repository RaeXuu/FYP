import os
import sys

# === 添加项目根目录到路径 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
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


# === 训练参数 ===
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")



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
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    all_preds = []
    all_labels = []

    for mels, labels in tqdm(dataloader, desc="Validation", leave=False):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        outputs = model(mels)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * mels.size(0)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc, np.array(all_labels), np.array(all_preds)



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
    # Datasets
    # =========================
    train_dataset = Dataset(
        metadata_path=metadata_path,
        sr=sr,
        segment_sec=segment_sec,
        mel_cfg=mel_cfg,
        augment=False,
    )

    val_dataset = Dataset(
        metadata_path=metadata_path,
        sr=sr,
        segment_sec=segment_sec,
        mel_cfg=mel_cfg,
        augment=False,
    )


    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_ds, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_ds = torch.utils.data.random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # === 模型、loss、优化器 ===
    model = LightweightCNN(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)


        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        # === Validation metrics (Confusion Matrix & Recall) ===
        class_names = ['artifact', 'extrahls', 'extrastole', 'murmur', 'normal']

        cm = confusion_matrix(y_true, y_pred)
        print("\n[Validation] Confusion Matrix:")
        print(cm)

        print("\n[Validation] Classification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        ))

        # === 保存最优模型 ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ New best model saved! Acc={best_acc:.4f}")

    print(f"\nTraining finished. Best Val Acc={best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()