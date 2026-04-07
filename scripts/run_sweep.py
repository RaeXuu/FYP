"""
Bayesian Hyperparameter Sweep for preprocessing parameters.
Uses train_lightweight (80/20, no test set) logic for speed.

Usage:
    python scripts/run_sweep.py              # create sweep + run 40 trials
    python scripts/run_sweep.py --id <id>    # attach to existing sweep
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import recall_score
from tqdm import tqdm
import yaml
from pathlib import Path
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
from src.model.lightweight_cnn import LightweightCNN

# =====================================================
# Sweep 搜索空间
# =====================================================
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/m_score", "goal": "maximize"},
    "parameters": {
        "n_mels":       {"values": [32, 64]},
        "hop_length":   {"values": [64, 96, 128]},
        "n_fft":        {"values": [128, 256, 512]},
        "overlap":      {"values": [0.25, 0.5, 0.75]},
        "lr":           {"values": [1e-3, 5e-4, 3e-4]},
        "weight_decay": {"values": [1e-4, 1e-3]},
    },
}

# =====================================================
# 固定超参
# =====================================================
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SWEEP_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "sweep_temp.pth")


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for mels, labels in tqdm(dataloader, desc="Train", leave=False):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(mels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * mels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for mels, labels in tqdm(dataloader, desc="Val", leave=False):
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)
        outputs = model(mels)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * mels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def train():
    wandb.init()
    c = wandb.config

    # 从 config.yaml 读基础配置，再用 sweep 参数覆盖
    CONFIG_PATH = Path(PROJECT_ROOT) / "config.yaml"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sr = cfg["data"]["sample_rate"]
    segment_sec = cfg["data"]["segment_length"]

    mel_cfg = {
        **cfg["mel"],
        "n_mels":     c.n_mels,
        "hop_length":  c.hop_length,
        "n_fft":       c.n_fft,
        "win_length":  c.n_fft,   # win_length 跟 n_fft 同步
    }

    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_physionet.csv")

    train_dataset = Dataset(
        metadata_path=metadata_path, sr=sr, segment_sec=segment_sec,
        mel_cfg=mel_cfg, augment=True, overlap=c.overlap,
    )
    val_dataset = Dataset(
        metadata_path=metadata_path, sr=sr, segment_sec=segment_sec,
        mel_cfg=mel_cfg, augment=False, overlap=c.overlap,
    )

    # 80/20 group split（dev 版，不动 test 集）
    rng = np.random.RandomState(42)
    all_fnames = [train_dataset.get_fname(i) for i in range(len(train_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_train = int(0.8 * len(unique_fnames))
    train_rec_ids = set(unique_fnames[:n_train])

    train_indices, val_indices, train_labels = [], [], []
    for idx, fname in enumerate(all_fnames):
        if fname in train_rec_ids:
            train_indices.append(idx)
            train_labels.append(train_dataset.samples[idx][1])
        else:
            val_indices.append(idx)

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(val_dataset, val_indices)

    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts
    sample_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LightweightCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_mscore, patience_counter = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

        se = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        sp = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        m_score = (se + sp) / 2

        scheduler.step(m_score)

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
        })

        if m_score > best_mscore:
            best_mscore = m_score
            patience_counter = 0
            torch.save(model.state_dict(), SWEEP_CKPT)
            wandb.summary["best_val_m_score"] = best_mscore
            wandb.summary["best_val_se"] = se
            wandb.summary["best_val_sp"] = sp
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                wandb.log({"early_stop_epoch": epoch})
                break

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help="Existing sweep ID to attach to")
    parser.add_argument("--count", type=int, default=40, help="Number of trials to run")
    args = parser.parse_args()

    if args.id:
        sweep_id = args.id
        print(f"Attaching to existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="heart-sound-fyp")
        print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=train, count=args.count)
