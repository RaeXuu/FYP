import os
import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import yaml
from pathlib import Path
from pprint import pprint
import wandb

# === 1. 项目路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
from src.model.lightweight_cnn import LightweightCNN

# === 2. 训练参数（完全对齐诊断模型） ===
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "quality_model_best.pth")

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for mels, labels in tqdm(dataloader, desc="Training Quality", leave=False):
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
    all_preds, all_labels = [], []
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
    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)

def main():
    print(f"Using device: {DEVICE}")

    # === 3. 加载配置与新 Metadata ===
    CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(CONFIG_PATH, "r") as f: cfg = yaml.safe_load(f)
    
    print("\n" + "=" * 60)
    print(f"[QUALITY CONFIG] Using config.yaml at:\n{CONFIG_PATH}")
    pprint(cfg)
    print("=" * 60 + "\n")

    wandb.init(
        project="heart-sound-fyp",
        name="sqa-model",
        config={
            "model": "LightweightCNN",
            "feature": "mel",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "split": "80/10/10",
            "sampler": "WeightedRandomSampler",
            "save_criterion": "val_m_score",
            **cfg,
        }
    )

    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_quality.csv")

    # 分别创建训练和验证 Dataset，确保验证集不加数据增强
    train_dataset = Dataset(metadata_path=metadata_path, sr=cfg["data"]["sample_rate"], 
                            segment_sec=cfg["data"]["segment_length"], mel_cfg=cfg["mel"], augment=True)
    val_dataset = Dataset(metadata_path=metadata_path, sr=cfg["data"]["sample_rate"], 
                            segment_sec=cfg["data"]["segment_length"], mel_cfg=cfg["mel"], augment=False)

    # === 4. Group Split 80/10/10 (按记录划分) ===
    rng = np.random.RandomState(42)
    all_fnames = [train_dataset.get_fname(i) for i in range(len(train_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)

    n_rec = len(unique_fnames)
    idx_train = int(0.8 * n_rec)
    idx_val   = int(0.9 * n_rec)

    train_rec_ids = set(unique_fnames[:idx_train])
    val_rec_ids   = set(unique_fnames[idx_train:idx_val])
    test_rec_ids  = set(unique_fnames[idx_val:])

    train_indices, val_indices, test_indices = [], [], []
    train_labels = []
    for idx, fname in enumerate(all_fnames):
        label = train_dataset.samples[idx][1]
        if fname in train_rec_ids:
            train_indices.append(idx)
            train_labels.append(label)
        elif fname in val_rec_ids:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    train_ds = Subset(train_dataset, train_indices)
    val_ds   = Subset(val_dataset, val_indices)
    test_ds  = Subset(val_dataset, test_indices)

    print("[Group Split by fname (80/10/10)]")
    print(f"  train samples = {len(train_ds)} | val = {len(val_ds)} | test = {len(test_ds)}")

    # === 持久化 quality test 集 fname 列表 ===
    quality_test_split_path = os.path.join(PROJECT_ROOT, "data", "quality_test_split.csv")
    if not os.path.exists(quality_test_split_path):
        test_fnames = [train_dataset.get_fname(i) for i in test_indices]
        with open(quality_test_split_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fname"])
            for fname in test_fnames:
                writer.writerow([fname])
        print(f"  ✅ quality_test_split.csv 已保存")
    else:
        print(f"  ℹ️  quality_test_split.csv 已存在，跳过写入")

    # === 5. Weighted Sampling (解决不平衡) ===
    class_counts = np.bincount(train_labels) 
    weights = 1. / class_counts
    samples_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # === 6. 训练循环（针对 Loss 优化的改动） ===
    model = LightweightCNN(num_classes=2).to(DEVICE)
    # 1. 使用标签平滑，防止过度拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
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

        class_names = ['Poor_Quality', 'Good_Quality']
        print("\n[Validation] Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

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
                class_names=['Poor_Quality', 'Good_Quality'],
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

    print(f"\nTraining finished. Best Val M-Score={best_mscore:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    # === 最终 Test 集评估（FP32 基准）===
    print("\n" + "!" * 20 + " FINAL TEST ON TEST SET " + "!" * 20)
    model.load_state_dict(torch.load(MODEL_PATH))

    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, criterion)

    se = recall_score(y_true_test, y_pred_test, pos_label=1)
    sp = recall_score(y_true_test, y_pred_test, pos_label=0)
    m_score = (se + sp) / 2

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | M-Score: {m_score:.4f}")
    print("\n[Test Set] Classification Report:")
    print(classification_report(y_true_test, y_pred_test, target_names=['Poor_Quality', 'Good_Quality'], digits=4))

    wandb.summary["test_loss"] = test_loss
    wandb.summary["test_acc"] = test_acc
    wandb.summary["test_sensitivity"] = se
    wandb.summary["test_specificity"] = sp
    wandb.summary["test_m_score"] = m_score
    wandb.finish()

if __name__ == "__main__":
    main()