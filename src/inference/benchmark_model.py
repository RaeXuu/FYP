import os
import sys
import time
import torch
import numpy as np

# 1. 修复路径问题
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.lightweight_cnn import LightweightCNN




# =========================
# Config - 统一参数
# =========================
NUM_CLASSES = 2
# 你可以手动修改这里来测不同的模型
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

BATCH_SIZE = 1
INPUT_SHAPE = (BATCH_SIZE, 1, 32, 64) # 32条Mel刻度

WARMUP = 20
RUNS = 200

# =========================
# Utils
# =========================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_macs(model, input_shape):
    macs = 0
    def hook(module, inp, out):
        nonlocal macs
        if isinstance(module, torch.nn.Conv2d):
            out_h, out_w = out.shape[2], out.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            macs += (module.in_channels * module.out_channels * kernel_ops * out_h * out_w)
        elif isinstance(module, torch.nn.Linear):
            macs += module.in_features * module.out_features

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(torch.randn(*input_shape))
    for h in hooks: h.remove()
    return macs

@torch.no_grad()
def benchmark_cpu_latency(model, x):
    model.eval().cpu()
    for _ in range(WARMUP): _ = model(x)
    t0 = time.perf_counter()
    for _ in range(RUNS): _ = model(x)
    return (time.perf_counter() - t0) * 1000.0 / RUNS

def main():
    print(f"\n==== 🚀 正在审计模型: {os.path.basename(MODEL_PATH)} ====")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    model = LightweightCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 指标计算
    params = count_params(model)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    macs = count_macs(model, INPUT_SHAPE)
    x = torch.randn(*INPUT_SHAPE)
    latency_ms = benchmark_cpu_latency(model, x)

    print(f"[参数量] {params:,}")
    print(f"[文件大小] {size_mb:.2f} MB")
    print(f"[计算量] {macs/1e6:.2f} M MACs")
    print(f"[CPU延迟] {latency_ms:.3f} ms / 样本")
    print(f"[吞吐量] {1000.0 / latency_ms:.2f} samples/s")
    print("=========================================\n")

if __name__ == "__main__":
    main()