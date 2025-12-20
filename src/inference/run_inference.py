import torch
import numpy as np


from src.model.lightweight_cnn import LightweightCNN

# =========================
# Config
# =========================
NUM_CLASSES = 5
MODEL_PATH = "checkpoints/best_model.pth"

INPUT_SHAPE = (1, 1, 64, 64)  # 和训练、benchmark保持一致


@torch.no_grad()
def main():
    # 1. 加载模型结构
    model = LightweightCNN(num_classes=NUM_CLASSES)

    # 2. 加载训练好的参数（关键）
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    # 3. 切换到 inference 模式
    model.eval()

    # 4. 构造一个“待推理样本”
    # 这里先用 dummy，后面可以换成真实特征
    x = torch.randn(*INPUT_SHAPE)


    # 5. forward = inference
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

    # 6. 输出结果
    print("Predicted class:", pred.item())
    print("Probabilities:", probs.squeeze().numpy())


if __name__ == "__main__":
    main()
