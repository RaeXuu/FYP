import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import yaml
from pathlib import Path
from pprint import pprint

# === 1. 项目路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train.dataset.dataset_mel import HeartSoundMelDataset as Dataset
from src.model.lightweight_cnn import LightweightCNN

# === 2. 训练参数（完全对齐诊断模型） ===
BATCH_SIZE = 16
EPOCHS = 25  # 统一为 25
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

    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata_quality.csv")

    # 分别创建训练和验证 Dataset，确保验证集不加数据增强
    train_dataset = Dataset(metadata_path=metadata_path, sr=cfg["data"]["sample_rate"], 
                            segment_sec=cfg["data"]["segment_length"], mel_cfg=cfg["mel"], augment=True)
    val_dataset = Dataset(metadata_path=metadata_path, sr=cfg["data"]["sample_rate"], 
                            segment_sec=cfg["data"]["segment_length"], mel_cfg=cfg["mel"], augment=False)

    # === 4. Group Split (按记录划分) ===
    rng = np.random.RandomState(42)
    all_fnames = [train_dataset.get_fname(i) for i in range(len(train_dataset))]
    unique_fnames = np.unique(all_fnames)
    rng.shuffle(unique_fnames)
    n_train_rec = int(0.8 * len(unique_fnames))
    train_rec_ids = set(unique_fnames[:n_train_rec])
    
    train_indices, val_indices = [], []
    train_labels = [] 
    for idx, fname in enumerate(all_fnames):
        label = train_dataset.samples[idx][1]
        if fname in train_rec_ids:
            train_indices.append(idx)
            train_labels.append(label)
        else:
            val_indices.append(idx)

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    print("[Group Split by fname]")
    print(f"  train samples = {len(train_ds)}")
    print(f"  val   samples = {len(val_ds)}")

    # === 5. Weighted Sampling (解决不平衡) ===
    class_counts = np.bincount(train_labels) 
    weights = 1. / class_counts
    samples_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # === 6. 训练循环（针对 Loss 优化的改动） ===
    model = LightweightCNN(num_classes=2).to(DEVICE)
    # 1. 使用标签平滑，防止过度拟合
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf') # 改为监控最小 Loss

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        # 2. 根据 Val Loss 来调整学习率
        scheduler.step(val_loss) 

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 补上混淆矩阵
        class_names = ['Poor_Quality', 'Good_Quality']
        print("\n[Validation] Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        print("\n[Validation] Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # === 3. 核心保存逻辑修改：保存 Loss 最小的模型 ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ New lowest loss model saved! Loss={best_val_loss:.4f}")

    print(f"\nTraining finished. Best Val Acc={best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()