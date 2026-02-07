import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 1. 修复路径问题
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.lightweight_cnn import LightweightCNN
from src.train.dataset.dataset_mel import HeartSoundMelDataset

# =========================
# Config
# =========================
NUM_CLASSES = 2
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

@torch.no_grad()
def main():
    # A. 加载 config.yaml 确保预处理一致
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"🧪 正在验证 PyTorch 推理逻辑: {os.path.basename(MODEL_PATH)}")
    
    # B. 初始化模型
    model = LightweightCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # C. 加载真实的验证集样本 (补齐参数)
    # 使用 os.path.join 确保在任何地方运行路径都正确
    # 1. 先加载全量数据集
    full_dataset = HeartSoundMelDataset(
        metadata_path=os.path.join(PROJECT_ROOT, "data/metadata_physionet.csv"),
        sr=cfg["data"]["sample_rate"],
        segment_sec=cfg["data"]["segment_length"],
        mel_cfg=cfg["mel"]
    )
    

    # 2. 调用我们之前写的隔离逻辑
    # 确保 test_subset 里全是模型没见过的 20%
    from scripts.evaluate_tflite import get_test_subset 
    test_subset = get_test_subset(full_dataset)

    # 3. 从这 20% 里随便挑一个
    x, label = test_subset[0] 
    x = x.unsqueeze(0)


    # D. 执行推理
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

    print("-" * 30)
    print(f"真实标签: {'Abnormal' if label == 1 else 'Normal'}")
    print(f"预测结果: {'Abnormal' if pred.item() == 1 else 'Normal'}")
    print(f"置信度 (Probabilities): {probs.squeeze().numpy()}")
    print("-" * 30)
    print("✅ 逻辑验证通过！")

if __name__ == "__main__":
    main()