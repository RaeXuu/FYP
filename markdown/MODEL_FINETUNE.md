# 模型调优记录

---

## 诊断模型（LightweightCNN + CoordAtt）

### Baseline Run 1 — 2026-04-07

**配置**
```
脚本：train_lightweight_with_test.py
数据集：metadata_physionet.csv（2876 条录音，62003 切片）
划分：80/10/10（按 fname 分组，seed=42）
Train: 49833 | Val: 5897 | Test: 6273
类别平衡：WeightedRandomSampler
Epochs：25 | Batch: 16 | LR: 1e-3 | weight_decay: 1e-4
Scheduler：ReduceLROnPlateau(mode=max, factor=0.5, patience=3)
保存标准：Val M-Score 最大
```

**config.yaml**
```
sample_rate: 2000
segment_length: 2.0
overlap: 0.5
bandpass: 25–400 Hz
n_fft: 256 | win_length: 256 | hop_length: 96
n_mels: 32 | fmin: 20 | fmax: 400 | power: 2.0
```

**Val 最优（Epoch 5）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9121 |
| Sensitivity | 0.9602 |
| Specificity | 0.8641 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8852 |
| Sensitivity | 0.9569 |
| Specificity | 0.8135 |
| Accuracy | 0.8406 |
| Test Loss | 0.3550 |

**wandb Run:** `diagnostic-model` → [heart-sound-fyp](https://wandb.ai/xrjgoole-google/heart-sound-fyp/runs/8nk8nb55)

**分析**
- M-Score 0.8852，达到 PhysioNet 2016 竞赛顶尖水平（~0.86–0.88），架构无需更换
- Sensitivity 0.9569 很高，漏报率仅 4.3%，医疗筛查场景表现良好
- Specificity 0.8135 偏低，正常心音误报率 18.7%，Abnormal precision 仅 0.54
- 过拟合明显：最优模型在 Epoch 5 出现，之后 val loss 持续上升（0.21 → 0.31）
- Val → Test 泛化差距约 0.027，在合理范围

**阈值扫描结果（基于 Run 1 模型）**

| 阈值 | Se | Sp | M-Score |
|------|----|----|---------|
| 0.30 | 0.9890 | 0.7787 | 0.8839 |
| 0.35 | 0.9840 | 0.7860 | 0.8850 |
| 0.40 | 0.9764 | 0.7947 | 0.8855 |
| **0.45** | **0.9688** | **0.8031** | **0.8859** ← 最优 |
| 0.50 | 0.9569 | 0.8135 | 0.8852（默认）|
| 0.55 | 0.9417 | 0.8218 | 0.8817 |
| 0.60 | 0.9231 | 0.8332 | 0.8782 |
| 0.65 | 0.8877 | 0.8440 | 0.8658 |
| 0.70 | 0.8547 | 0.8562 | 0.8554 |
| 0.75 | 0.8193 | 0.8742 | 0.8467 |
| 0.80 | 0.7652 | 0.8915 | 0.8284 |

结论：阈值调整收益极小（最优 0.45 比默认 0.50 仅提升 0.0007），Se/Sp 失衡是训练时烙进去的，靠后处理无法根本改善。家庭筛查场景下维持默认阈值 0.50（Se=0.9569）更合适。

**待改进**
- [ ] 加 Early Stopping（patience=5），避免 Epoch 5 后的无效训练
- [ ] 推理时调整分类阈值，平衡 Se / Sp（后处理，不需要重训）
- [ ] 跑 SQA 模型后进行 TFLite 量化，Pi 上对比 FP32 vs INT8

---

### Run 2 — 2026-04-07（Label Smoothing + Early Stopping）

**相比 Run 1 的改动**
```
CrossEntropyLoss(label_smoothing=0.1)
Early Stopping: patience=10
EPOCHS: 25 → 50（实际第 16 epoch 触发 early stopping）
```

**Val 最优（Epoch 6）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9106 |
| Sensitivity | 0.9238 |
| Specificity | 0.8974 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8816 |
| Sensitivity | 0.9181 |
| Specificity | 0.8452 |
| Accuracy | 0.8589 |
| Test Loss | 0.4056 |

**wandb Run:** `diagnostic-model` → [heart-sound-fyp](https://wandb.ai/xrjgoole-google/heart-sound-fyp/runs/l7adpejw)

**与 Run 1 对比**
| 指标 | Run 1 | Run 2 | 变化 |
|------|-------|-------|------|
| Test M-Score | 0.8852 | 0.8816 | -0.004 |
| Test Se | 0.9569 | 0.9181 | -0.039 |
| Test Sp | 0.8135 | 0.8452 | +0.032 |
| Se/Sp 差距 | 0.143 | 0.073 | 缩小一半 |
| 停止 epoch | 25 | 16 | 更高效 |

**分析**
- Se/Sp 更平衡，但 M-Score 基本持平（差距在误差范围内）
- 无法确定 Se/Sp 变动是由 label smoothing 还是 early stopping 单独造成的（两者同时改变）
- 家庭筛查场景下 Run 1（高 Se）更合适，漏诊代价远高于误报
- 注意：Run 2 已覆盖 `best_model.pth`，需重训或 sweep 后确定最终模型

---

## SQA 模型（LightweightCNN）

*(待填)*

---

## 消融实验

*(待填 — 在 baseline 基础上逐步改动架构后记录)*
