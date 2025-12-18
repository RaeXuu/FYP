import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

# ① 导入你已有的 Dataset 和 model
from src.train.dataset.dataset_bicoherence import HeartSoundDataset
from src.train.model.lightweight_cnn import YourModelClass   # ⚠️换成你自己的
from config import LABEL_NAMES            # 比如 ['artifact', 'extrahls', ...]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_inference(model, dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())

    return preds


def main():
    # ② 加载模型
    model = YourModelClass(num_classes=5)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # ③ 加载 unlabelled dataset
    dataset = HeartSoundDataset(
        data_dir="data/Aunlabelledtest",   # 或 Bunlabelledtest
        split="unlabelled",                # ⚠️如果你有这个参数
        return_label=False                 # 关键点：不返回 y
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ④ 跑 inference
    preds = run_inference(model, dataloader)

    # ⑤ 统计预测分布
    counter = Counter(preds)
    total = sum(counter.values())

    print("Prediction distribution:")
    for k, v in counter.items():
        print(f"{LABEL_NAMES[k]:12s}: {v / total:.2%} ({v})")


if __name__ == "__main__":
    main()
