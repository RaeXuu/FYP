# LightweightCNN 模型剪枝方案

## 1. 背景

当前项目训练/部署流程：

```
PyTorch (.pth)  →  ai_edge_torch  →  TFLite  →  树莓派 (ai_edge_litert)
```

- 模型：`LightweightCNN`（CoordAtt 版），约 100k 参数
- 输入：mel 频谱 `(1, 64, 64)`
- 已有量化：FP32 / INT8 动态 / INT8 全整型
- 训练脚本：`src/train/train_lightweight_with_test.py`、`src/train/train_sqa_with_test.py`
- 转换脚本：`scripts/convert_to_tflite.py`
- 评估脚本：`scripts/evaluate_tflite_on_pi.py`

## 2. 两种剪枝策略

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **非结构化剪枝** | 逐权重置零，权重矩阵变稀疏 | 精度损失小，压缩率高（理论） | TFLite 支持有限，未必加速 CPU 推理 |
| **结构化剪枝** | 剪整个通道/卷积核，改变张量形状 | 直接减少 MACs，任何硬件都加速 | 精度损失更大，需仔细调参和微调 |

**建议：优先结构化剪枝**，目标是在树莓派 CPU 上实际加速推理。

---

## 3. 结构化剪枝流程

### 3.1 分析各层冗余度（L1-norm）

```python
import torch
import torch.nn as nn
from src.model.lightweight_cnn import LightweightCNN

def analyze_channels(model):
    """统计每个 pointwise Conv2d 的每通道 L1 范数"""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size == (1, 1):
            l1 = m.weight.data.abs().sum(dim=(1, 2, 3))  # (out_channels,)
            stats[name] = {
                "channels": len(l1),
                "l1_mean": l1.mean().item(),
                "l1_std": l1.std().item(),
                "l1_min": l1.min().item(),
                "below_10pct_mean": (l1 < l1.mean() * 0.1).sum().item(),  # 极弱通道数
            }
    return stats

# 加载模型分析
model = LightweightCNN(num_classes=2, in_channels=1)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
model.eval()
for name, s in analyze_channels(model).items():
    print(f"{name:30s}  ch={s['channels']:3d}  L1_mean={s['l1_mean']:.4f}  弱通道={s['below_10pct_mean']}")
```

### 3.2 剪枝实现

```python
def prune_structured(model, prune_ratio=0.2):
    """
    对每个 pointwise Conv2d 剪掉 L1-norm 最小的 prune_ratio 通道。
    注意：需要同步修改后续 BN / 下一层卷积的 in_channels。
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size == (1, 1):
            l1 = m.weight.data.abs().sum(dim=(1, 2, 3))
            keep_n = int(m.out_channels * (1 - prune_ratio))
            keep_idx = l1.topk(keep_n).indices.sort().values

            # 裁剪 weight + bias
            m.weight.data = m.weight.data[keep_idx]
            if m.bias is not None:
                m.bias.data = m.bias.data[keep_idx]
            m.out_channels = keep_n

            # 如果紧跟着 BN，也要裁剪
            # （需要根据具体网络结构处理，见下方完整实现）
    return model
```

**实际剪枝时要处理的连锁修改：**

以 `DepthwiseSeparableConv` 为例：
```
Pointwise Conv2d (in, out) → BN(out) → ReLU → CoordAtt(out, out)
```
剪掉 pointwise 的 N 个输出通道后，必须同步裁剪：
1. `BN` 的 `weight`、`bias`、`running_mean`、`running_var`
2. `CoordAtt` 内部所有相关的 Conv2d 和 BN
3. 下一层的 `in_channels`

---

## 4. 非结构化剪枝（快速验证用）

PyTorch 内置 API，适合快速实验：

```python
import torch.nn.utils.prune as prune

model = LightweightCNN(num_classes=2, in_channels=1)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))

# 对所有 Conv2d 做 30% L1 非结构化剪枝
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        prune.l1_unstructured(m, name="weight", amount=0.3)

# 永久化 mask（去掉 hook）
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        try:
            prune.remove(m, "weight")
        except:
            pass

# 微调 + 保存 + 转换（同正常流程）
```

**注意：** 非结构化剪枝后权重是稀疏的，但 TFLite 转换可能将稀疏矩阵还原为稠密，不保证减小模型体积或加速。

---

## 5. 完整工作流

```
1. 基线评估
   python scripts/evaluate_tflite_on_pi.py --mode all

2. 剪枝 + 微调（新脚本 scripts/prune_model.py）
   python scripts/prune_model.py --prune_ratio 0.2 --finetune_epochs 20

3. 转换 TFLite
   python scripts/convert_to_tflite.py   # 指向 pruned 权重

4. 剪枝后评估
   python scripts/evaluate_tflite_on_pi.py --mode all

5. 对比剪枝前后：参数量 / 推理延迟 / 准确率 / M-Score
```

---

## 6. 建议的剪枝比例梯度

| 剪枝比例 | 预期效果 | 建议 |
|---------|---------|------|
| 10% | 参数量下降 10-15%，精度几乎无损 | 先跑这个 |
| 20% | 参数量下降 20-30%，可能有轻微精度损失 | 主力方案 |
| 30% | 显著缩小，可能需要更多微调 epoch | 试到极限后回调 |
| >30% | 风险较大，需要逐层差异化剪枝 | 不推荐一刀切 |

---

## 7. 剪枝 + 量化的叠加效果

你已有的 INT8 全整型量化（`convert_to_tflite.py`）可以和剪枝叠加：

```
FP32 基线 → 剪枝 FP32 → 剪枝 INT8 全整型
```

这两个技术是正交的——剪枝减少计算量（MACs），量化减少每次计算的位宽。叠加使用通常比单独使用效果更好。

---

## 8. 参考

- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Torch Pruning Library (结构化)](https://github.com/VainF/Torch-Pruning)
- 项目关键文件：
  - `src/model/lightweight_cnn.py` — 模型定义
  - `src/train/train_lightweight_with_test.py` — 训练脚本
  - `scripts/convert_to_tflite.py` — TFLite 转换
  - `scripts/evaluate_tflite_on_pi.py` — Pi 端评估
