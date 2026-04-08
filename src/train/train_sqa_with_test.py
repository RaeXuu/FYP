import os
import sys

# === 添加项目根目录到路径 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import csv
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

# === 保证路径正确 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# === 再导入模块 ===
if FEATURE_TYPE == "mel":
    from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
else:
    raise ValueError(f"Unknown FEATURE_TYPE: {FEATURE_TYPE}")

from src.model.lightweight_cnn import LightweightCNN

# === 训练参数 ===
RUN_NAME = "SQA-run-1"   # <-- 每次改这里
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_sqa.pth")

# 标签说明：metadata_quality_reversed.csv 中 label=1 为差质量（正类），label=0 为好质量
# Sensitivity = 差质量检出率，Specificity = 好质量正确率
CLASS_NAMES = ['Good', 'Bad']


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
            "task": "SQA",
            "model": "LightweightCNN",
            "feature": FEATURE_TYPE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "split": "80/10/10",
            "sampler": "WeightedRandomSampler",
            "save_criterion": "val_m_score",
            "label_convention": "bad=1(pos), good=0",
            **cfg,
        }
    )

    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_quality_reversed.csv")

    data_cfg = cfg["data"]
    mel_cfg = cfg["mel"]
    sr = data_cfg["sample_rate"]
    segment_sec = data_cfg["segment_length"]

    train_dataset = Dataset(metadata_path=metadata_path, sr=sr, segment_sec=segment_sec, mel_cfg=mel_cfg, augment=True)
    val_dataset   = Dataset(metadata_path=metadata_path, sr=sr, segment_sec=segment_sec, mel_cfg=mel_cfg, augment=False)

    SPLIT_SEED = 42
    TRAIN_RATIO, VAL_RATIO = 0.8, 0.1  # 剩余 0.1 自动分配给 Test

    rng = np.random.RandomState(SPLIT_SEED)
    all_fnames = [train_dataset.get_fname(i) for i in range(len(train_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_rec = len(unique_fnames)
    idx_train = int(n_rec * TRAIN_RATIO)
    idx_val   = int(n_rec * (TRAIN_RATIO + VAL_RATIO))

    train_rec_ids = set(unique_fnames[:idx_train])
    val_rec_ids   = set(unique_fnames[idx_train:idx_val])
    test_rec_ids  = set(unique_fnames[idx_val:])

    train_indices, val_indices, test_indices = [], [], []
    train_labels = []
    for idx, fname in enumerate(all_fnames):
        if fname in train_rec_ids:
            train_indices.append(idx)
            train_labels.append(train_dataset.samples[idx][1])
        elif fname in val_rec_ids:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(val_dataset, val_indices)
    test_ds  = torch.utils.data.Subset(val_dataset, test_indices)

    print("[Group Split by fname (80/10/10)]")
    print(f"  train samples = {len(train_ds)} | val samples = {len(val_ds)} | test samples = {len(test_ds)}")

    # === 持久化 SQA test 集 fname 列表（与诊断模型 test_split.csv 独立）===
    test_split_path = os.path.join(PROJECT_ROOT, "data", "test_split_sqa.csv")
    if not os.path.exists(test_split_path):
        test_fnames = [train_dataset.get_fname(i) for i in test_indices]
        with open(test_split_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fname"])
            for fname in test_fnames:
                writer.writerow([fname])
        print(f"  ✅ test_split_sqa.csv 已保存: {test_split_path}")
    else:
        print(f"  ℹ️  test_split_sqa.csv 已存在，跳过写入")

    # === WeightedRandomSampler（处理 8:1 类别不平衡）===
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts
    sample_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"  class counts = {class_counts} | weights = {weights.round(4)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = LightweightCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_mscore = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

        # Se = 差质量检出率（pos_label=1），Sp = 好质量正确率（pos_label=0）
        se = recall_score(y_true, y_pred, pos_label=1)
        sp = recall_score(y_true, y_pred, pos_label=0)
        m_score = (se + sp) / 2

        scheduler.step(m_score)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Se(bad): {se:.4f} | Sp(good): {sp:.4f} | M-Score: {m_score:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        print("\n[Validation] Confusion Matrix:")
        print(cm)

        print("\n[Validation] Classification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

        wandb.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
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
                class_names=CLASS_NAMES,
            ),
        })

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

    # ==========================================================================
    # 最终测试集评估
    # ==========================================================================
    print("\n" + "!"*20 + " FINAL TEST ON TEST SET " + "!"*20)
    model.load_state_dict(torch.load(MODEL_PATH))

    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, criterion)

    se = recall_score(y_true_test, y_pred_test, pos_label=1)
    sp = recall_score(y_true_test, y_pred_test, pos_label=0)
    m_score = (se + sp) / 2

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | M-Score: {m_score:.4f}")
    print(f"Test Se(bad): {se:.4f} | Sp(good): {sp:.4f}")
    print("\n[Test Set] Classification Report:")
    print(classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, digits=4))

    wandb.summary["test_loss"] = test_loss
    wandb.summary["test_acc"] = test_acc
    wandb.summary["test_sensitivity"] = se
    wandb.summary["test_specificity"] = sp
    wandb.summary["test_m_score"] = m_score
    wandb.finish()


if __name__ == "__main__":
    main()
