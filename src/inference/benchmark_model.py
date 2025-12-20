import os
import time
import torch
import numpy as np

from src.model.lightweight_cnn import LightweightCNN

# =========================
# Config
# =========================
NUM_CLASSES = 5
MODEL_PATH = os.path.join("checkpoints", "best_model.pth")

BATCH_SIZE = 1
INPUT_SHAPE = (BATCH_SIZE, 1, 64, 64)

WARMUP = 20
RUNS = 200


# =========================
# Utils
# =========================
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_shape):
    """
    Rough MACs estimation for Conv2d & Linear
    """
    macs = 0

    def hook(module, inp, out):
        nonlocal macs
        if isinstance(module, torch.nn.Conv2d):
            out_h, out_w = out.shape[2], out.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            macs += (
                module.in_channels
                * module.out_channels
                * kernel_ops
                * out_h
                * out_w
            )
        elif isinstance(module, torch.nn.Linear):
            macs += module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        model(dummy)

    for h in hooks:
        h.remove()

    return macs


@torch.no_grad()
def benchmark_cpu_latency(model, x):
    model.eval()
    model.cpu()
    x = x.cpu()

    for _ in range(WARMUP):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(RUNS):
        _ = model(x)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / RUNS
    return avg_ms


# =========================
# Main
# =========================
def main():
    print("==== Deployment Benchmark ====")

    model = LightweightCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # ---- Model size ----
    params = count_params(model)
    print(f"[Params] {params:,}")

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[File ] {size_mb:.2f} MB")
    else:
        print("[File ] model file not found")

    # ---- MACs ----
    macs = count_macs(model, INPUT_SHAPE)
    print(f"[MACs ] {macs/1e6:.2f} M")

    # ---- Latency ----
    x = torch.randn(*INPUT_SHAPE)
    latency_ms = benchmark_cpu_latency(model, x)
    print(f"[CPU  ] Latency = {latency_ms:.3f} ms / sample")

    # ---- Throughput ----
    throughput = 1000.0 / latency_ms
    print(f"[TPS  ] {throughput:.2f} samples/s")

    print("===============================")


if __name__ == "__main__":
    main()
