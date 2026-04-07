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

**待改进**
- [ ] 加 Early Stopping（patience=5），避免 Epoch 5 后的无效训练
- [ ] 推理时调整分类阈值，平衡 Se / Sp（后处理，不需要重训）
- [ ] 跑 SQA 模型后进行 TFLite 量化，Pi 上对比 FP32 vs INT8

---

## SQA 模型（LightweightCNN）

*(待填)*

---

## 消融实验

*(待填 — 在 baseline 基础上逐步改动架构后记录)*
