import os
import sys

# === 添加项目根目录到路径 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import yaml
from pathlib import Path
from pprint import pprint
import wandb



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
RUN_NAME = "diagnostic-dev"   # <-- 每次改这里
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
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

    wandb.init(
        project="heart-sound-fyp",
        name=RUN_NAME,
        config={
            "model": "LightweightCNN",
            "feature": FEATURE_TYPE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "split": "80/20",
            "sampler": "WeightedRandomSampler",
            "save_criterion": "val_m_score",
            **cfg,
        }
    )

    # =========================
    # Paths
    # =========================
    # metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata1.csv")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_physionet.csv")

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
        augment=True,
    )

    val_dataset = Dataset(
        metadata_path=metadata_path,
        sr=sr,
        segment_sec=segment_sec,
        mel_cfg=mel_cfg,
        augment=False,
    )


    # =========================
    # Group split by fname（关键修改）
    # =========================
    SPLIT_SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    rng = np.random.RandomState(SPLIT_SEED)

    # 1️⃣ 从 dataset 拿到每个切片对应的 fname
    all_fnames = [train_dataset.get_fname(i) for i in range(len(train_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_rec = len(unique_fnames)
    n_train_rec = int(TRAIN_RATIO * n_rec)

    train_rec_ids = set(unique_fnames[:n_train_rec])
    val_rec_ids   = set(unique_fnames[n_train_rec:])

    train_indices = []
    val_indices   = []
    train_labels  = []

    # 2️⃣ 按 fname 分配每一个切片
    for idx, fname in enumerate(all_fnames):
        if fname in train_rec_ids:
            train_indices.append(idx)
            train_labels.append(train_dataset.samples[idx][1])
        else:
            val_indices.append(idx)

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(val_dataset, val_indices)

    print("[Group Split by fname]")
    print(f"  train samples = {len(train_ds)}")
    print(f"  val   samples = {len(val_ds)}")
    print(f"  unique train recordings = {len(train_rec_ids)}")
    print(f"  unique val   recordings = {len(val_rec_ids)}")

    # === WeightedRandomSampler（处理 4:1 类别不平衡）===
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts
    sample_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"  class counts = {class_counts} | weights = {weights.round(4)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # === 模型、loss、优化器 ===
    # model = LightweightCNN(num_classes=5).to(DEVICE)
    model = LightweightCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 在优化器定义后加入
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_mscore = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

        # 计算 M-Score = (Sensitivity + Specificity) / 2
        se = recall_score(y_true, y_pred, pos_label=1)
        sp = recall_score(y_true, y_pred, pos_label=0)
        m_score = (se + sp) / 2

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        scheduler.step(m_score)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Se: {se:.4f} | Sp: {sp:.4f} | M-Score: {m_score:.4f}")

        class_names = ['Normal', 'Abnormal']
        cm = confusion_matrix(y_true, y_pred)
        print("\n[Validation] Confusion Matrix:")
        print(cm)

        print("\n[Validation] Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        wandb.log({
            "epoch": epoch,
            "lr": current_lr,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/sensitivity": se,
            "val/specificity": sp,
            "val/m_score": m_score,
            "val/conf_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true.tolist(),
                preds=y_pred.tolist(),
                class_names=class_names,
            ),
        })

        # === 保存最优模型（基于 M-Score）===
        if m_score > best_mscore:
            best_mscore = m_score
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ New best model saved! M-Score={best_mscore:.4f} (Se={se:.4f}, Sp={sp:.4f})")
            wandb.summary["best_val_m_score"] = best_mscore
            wandb.summary["best_val_se"] = se
            wandb.summary["best_val_sp"] = sp
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹ Early stopping at epoch {epoch}.")
                wandb.log({"early_stop_epoch": epoch})
                break

    print(f"\nTraining finished. Best Val M-Score={best_mscore:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    wandb.finish()


if __name__ == "__main__":
    main()