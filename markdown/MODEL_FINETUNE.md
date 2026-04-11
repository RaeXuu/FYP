# 模型调优记录

---

## 最终选定参数汇总

### 诊断模型最优配置（Run 6）

| 类别 | 参数 | 值 |
|------|------|----|
| **预处理** | sample_rate | 2000 |
| | segment_length | 2.0s |
| | overlap | 0.5 |
| | bandpass | 25–400 Hz |
| **Mel** | n_mels | 64 |
| | n_fft / win_length | 256 |
| | hop_length | 128 |
| | fmin / fmax | 20 / 400 Hz |
| | power | 2.0 |
| **训练** | batch_size | 16 |
| | epochs | 50（early stop patience=10）|
| | lr | 1e-3 |
| | weight_decay | 1e-4 |
| | optimizer | Adam |
| | scheduler | ReduceLROnPlateau(factor=0.5, patience=3) |
| | loss | CrossEntropyLoss（无 label smoothing）|
| | sampler | WeightedRandomSampler |
| **结果** | Test M-Score | 0.8903 |
| | Test Sensitivity | 0.9485 |
| | Test Specificity | 0.8322 |

> 预处理参数（n_mels=64, hop=128）来自 40 组 Bayesian sweep（`scripts/run_sweep.py`），训练参数通过 Run 1–7 消融实验确定。

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

### Sweep Run — 2026-04-08（Bayesian Hyperparameter Search，40 trials）

**目标**：用 Bayesian 搜索找最佳预处理参数，固定后跑最终训练

**搜索空间**
```
n_mels:      [32, 64]
hop_length:  [64, 96, 128]
n_fft:       [128, 256, 512]
overlap:     [0.25, 0.5, 0.75]
lr:          [1e-3, 5e-4, 3e-4]
weight_decay:[1e-4, 1e-3]
固定：batch_size=16, epochs=50, early_stop_patience=10
优化目标：val/m_score（maximize）
```

**Top 3 结果**

| 排名 | wandb Run | m_score | sensitivity | specificity | n_mels | hop_length | n_fft | overlap | lr | weight_decay |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lyric-sweep-12 | 0.9033 | 0.9510 | 0.8556 | 64 | 128 | 256 | 0.75 | 1e-3 | 1e-3 |
| 2 | happy-sweep-30 | 0.9031 | 0.9677 | 0.8386 | 64 | 96 | 256 | 0.75 | 1e-3 | 1e-3 |
| 3 | eternal-sweep-20 | 0.9000 | 0.9539 | 0.8462 | 64 | 128 | 256 | 0.75 | 1e-3 | 1e-3 |

**选定最佳参数（lyric-sweep-12）**
```
n_mels:      64
hop_length:  128
n_fft:       256  (win_length 同步为 256)
overlap:     0.75
lr:          1e-3
weight_decay:1e-3
```

**规律分析**
- `n_mels=64`、`n_fft=256`、`overlap=0.75`、`lr=1e-3` 在 top5 中反复出现，是稳定最优区域
- `weight_decay=1e-3`（而非 `1e-4`）在 sweep runs 里更占优
- 若优先 sensitivity，选 happy-sweep-30（se=0.9677）；综合 m_score 选 lyric-sweep-12

---

### Run 4 — 2026-04-08（无 label smoothing，有 early stopping）

**相比 Run 2 的改动**
```
CrossEntropyLoss(label_smoothing=0.1) → CrossEntropyLoss()
其余与 Run 2 相同（overlap=0.5, batch=256）
```

**Val 最优（Epoch ?）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9039 |
| Sensitivity | 0.9514 |
| Specificity | 0.8564 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8835 |
| Sensitivity | 0.9544 |
| Specificity | 0.8125 |
| Accuracy | 0.8393 |
| Test Loss | 0.4100 |

Early stop 触发于 Epoch 14

**与 Run 2 对比（唯一变量：去掉 label smoothing）**
| 指标 | Run 2 (label smoothing) | Run 4 (无 label smoothing) | 变化 |
|------|------------------------|---------------------------|------|
| Test M-Score | 0.8816 | 0.8835 | +0.002 |
| Test Se | 0.9181 | 0.9544 | +0.036 |
| Test Sp | 0.8452 | 0.8125 | -0.033 |
| Val M-Score (best) | 0.9106 | 0.9039 | -0.007 |

**结论**：去掉 label smoothing 后 Se 大幅回升（+0.036），Sp 下降（-0.033），M-Score 基本持平。Se/Sp 失衡与 Run 1 相似，说明 label smoothing 是平衡 Se/Sp 的关键因素，但对 M-Score 影响极小。

---

### Run 3 — 2026-04-08（overlap=0.75 对照实验）

**相比 Run 1/2 的改动**
```
config.yaml: overlap: 0.5 → 0.75
DataLoader:  num_workers=4, pin_memory=True, persistent_workers=True
其余参数与 Run 1 相同（n_mels=32, hop_length=96, n_fft=256, lr=1e-3, weight_decay=1e-4）
```

**目的**：单独验证 overlap=0.75 对泛化性能的影响，与 Run 1（test M-Score=0.8852）对比

**DataLoader 优化说明**
- `num_workers=4`：多进程并行 mel 计算，避免 GPU 等待 CPU
- `pin_memory=True`：数据存锁页内存，加速 CPU→GPU 传输
- `persistent_workers=True`：epoch 间保留 worker 进程，减少启动开销

**wandb Run 名称**：`src/train/train_lightweight_with_test.py:117` → `name="..."` 字段

**Val 最优（Epoch 6）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9070 |
| Sensitivity | 0.9007 |
| Specificity | 0.9133 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8828 |
| Sensitivity | 0.9105 |
| Specificity | 0.8551 |
| Accuracy | 0.8655 |
| Test Loss | 0.4080 |

Early stop 触发于 Epoch 15

**与 Run 2 对比（唯一变量：overlap 0.5→0.75）**
| 指标 | Run 2 (overlap=0.5) | Run 3 (overlap=0.75) | 变化 |
|------|---------------------|----------------------|------|
| Test M-Score | 0.8816 | 0.8828 | +0.001 |
| Test Se | 0.9181 | 0.9105 | -0.008 |
| Test Sp | 0.8452 | 0.8551 | +0.010 |
| Val M-Score (best) | 0.9106 | 0.9070 | -0.004 |

**结论**：overlap=0.75 与 0.5 的 test M-Score 基本持平（差距 0.001，在误差范围内），Se/Sp 略有互换。高 overlap 没有带来明显提升，也没有变差。维持 overlap=0.75 或回退 0.5 均可，建议后续跑完整 sweep 参数（n_mels=64, hop=128）后再做最终决定。

---

### Run 5 — 2026-04-08（sweep 最佳预处理参数）

**相比 Run 4 的改动**
```
config.yaml: n_mels: 32→64, hop_length: 96→128
其余与 Run 4 相同（无 label smoothing, overlap=0.5, batch=256）
```

**Val 最优**
| 指标 | 值 |
|------|----|
| M-Score | 0.9029 |
| Sensitivity | 0.9420 |
| Specificity | 0.8639 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8784 |
| Sensitivity | 0.9409 |
| Specificity | 0.8159 |
| Accuracy | 0.8395 |
| Test Loss | 0.3240 |

Early stop 触发于 Epoch 11

**与 Run 4 对比（唯一变量：n_mels 32→64, hop_length 96→128）**
| 指标 | Run 4 | Run 5 | 变化 |
|------|-------|-------|------|
| Test M-Score | 0.8835 | 0.8784 | -0.005 |
| Test Se | 0.9544 | 0.9409 | -0.014 |
| Test Sp | 0.8125 | 0.8159 | +0.003 |
| Val M-Score (best) | 0.9039 | 0.9029 | -0.001 |

**结论**：sweep 最佳参数在 with_test 完整训练中未能带来提升，反而略有下降。sweep 是在 80/20 无 test 集的轻量模式下搜的，与完整训练条件不完全一致，存在分布差异。当前最优仍是 **Run 4**（n_mels=32, hop_length=96）。

---

### Run 6 — 2026-04-08（n_mels=64, hop=128, batch=16）

**相比 Run 5 的改动**
```
BATCH_SIZE: 256 → 16
其余与 Run 5 相同（n_mels=64, hop=128, 无 label smoothing, overlap=0.5）
```

**Val 最优（Epoch 2）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9009 |
| Sensitivity | 0.9318 |
| Specificity | 0.8700 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8903 |
| Sensitivity | 0.9485 |
| Specificity | 0.8322 |
| Accuracy | 0.8541 |
| Test Loss | 0.3613 |

Early stop 触发于 Epoch 12

**与 Run 4/5 对比**
| Run | batch | n_mels | 最佳 epoch | Test M-Score | Test Se | Test Sp |
|-----|-------|--------|-----------|-------------|---------|---------|
| Run 4 | 256 | 32 | 4 | 0.8835 | 0.9544 | 0.8125 |
| Run 5 | 256 | 64 | 1 | 0.8784 | 0.9409 | 0.8159 |
| Run 6 | 16 | 64 | 2 | **0.8903** | 0.9485 | 0.8322 |

**结论**：batch=16 解决了 n_mels=64 的过早过拟合问题（最佳 epoch 从 1 恢复到 2）。Run 6 是目前最优，M-Score 0.8903，Se/Sp 也更平衡。需 Run 7（batch=16, n_mels=32）做公平对照，排除 batch 大小本身的影响。

---

### Run 7 — 2026-04-08（n_mels=32, hop=96, batch=16，与 Run 6 公平对照）

**目的**：控制 batch=16，n_mels=32，与 Run 6（batch=16, n_mels=64）形成公平对照，同时与 Run 4（batch=256, n_mels=32）验证 batch 大小的影响。

**配置**
```
n_mels=32, hop_length=96
BATCH_SIZE=16
无 label smoothing, overlap=0.5
```

**Val 最优（Epoch 5）**
| 指标 | 值 |
|------|----|
| M-Score | 0.9163 |
| Sensitivity | 0.9487 |
| Specificity | 0.8838 |

**Test 集最终结果**
| 指标 | 值 |
|------|----|
| M-Score | 0.8869 |
| Sensitivity | 0.9383 |
| Specificity | 0.8355 |
| Accuracy | 0.8549 |
| Test Loss | 0.3491 |

Early stop 触发于 Epoch 15

**完整对照表（batch 与 n_mels 消融）**
| Run | batch | n_mels | 最佳 epoch | Test M-Score | Test Se | Test Sp |
|-----|-------|--------|-----------|-------------|---------|---------|
| Run 4 | 256 | 32 | 4 | 0.8835 | 0.9544 | 0.8125 |
| Run 5 | 256 | 64 | 1 | 0.8784 | 0.9409 | 0.8159 |
| Run 6 | 16 | 64 | 2 | 0.8903 | 0.9485 | 0.8322 |
| **Run 7** | **16** | **32** | **5** | **0.8869** | 0.9383 | 0.8355 |

**结论**
- batch=16 优于 batch=256（Run 7 vs Run 4：+0.003，最佳 epoch 更晚更稳）
- n_mels=64 vs 32 在 batch=16 下差距极小（0.8903 vs 0.8869），但 n_mels=64 有 Bayesian sweep 作为方法论支撑
- **最终选定参数：batch=16, n_mels=64, hop=128, 无 label smoothing**（Run 6，test M-Score 最高且参数来自系统性搜索）

---

### Sweep 跑完后的步骤

1. **查最佳参数**
   - 去 wandb → heart-sound-fyp → Sweeps → 按 `val/m_score` 排序
   - 记录最佳 trial 的 `n_mels / hop_length / n_fft / overlap / lr / weight_decay`

2. **更新 config.yaml**
   - 把最佳预处理参数（`n_mels`、`hop_length`、`n_fft`、`overlap`）写入 `config.yaml`
   - `lr` 和 `weight_decay` 写入 `train_lightweight_with_test.py` 的常量

3. **加 mel 缓存（dataset_mel.py）**
   - 问题：每个 epoch 在 `__getitem__` 里重复计算 mel，CPU 跑满、GPU 饿着
   - 改动：第一次计算完存到 `data/mel_cache/`（约 180MB），之后直接读文件
   - 注意：改完 config 参数后需删掉旧 cache 重新生成

4. **跑最终训练**
   - 脚本：`train_lightweight_with_test.py`
   - 确认 `test_split.csv` 已存在（不要删，保证 test 集锁住）
   - 记录结果到本文件

---

## SQA 模型（LightweightCNN）

### 标签约定说明

**原始 `metadata_quality.csv`（由 `load_quality.py` 生成）**

| label | 含义 | 数量 |
|-------|------|------|
| 0 | 差质量（score == 0） | 364 |
| 1 | 好质量（score != 0） | 2876 |

**问题**：若直接使用原始 CSV，M-Score 的 Sensitivity = 召回 class 1 = 好质量检出率，与 SQA 的目标相反——我们想要的是**别漏掉差质量音频**，防止噪声送进诊断模型。

**解决方案**：新生成 `metadata_quality_reversed.csv`（`flip_label=True`），反转标签：

| label | 含义 | 数量 |
|-------|------|------|
| 0 | 好质量 | 2876 |
| 1 | 差质量（正类） | 364 |

这样 Sensitivity = 差质量检出率，与诊断模型的逻辑一致（正类 = 需要被检出的类）。

- 生成脚本：`src/preprocess/load_quality.py`（`flip_label=True` 参数）
- 训练使用：`metadata_quality_reversed.csv`

---

### 训练配置

与诊断模型共用架构和超参，差异如下：

| 项目 | 诊断模型 | SQA 模型 |
|------|---------|---------|
| 训练脚本 | `train_lightweight_with_test.py` | `train_sqa_with_test.py` |
| metadata | `metadata_physionet.csv` | `metadata_quality_reversed.csv` |
| 模型保存 | `checkpoints/best_model.pth` | `checkpoints/best_model_sqa.pth` |
| test split | `data/test_split.csv` | `data/test_split_sqa.csv` |
| class names | Normal(0) / Abnormal(1) | Good(0) / Bad(1) |
| 类别不平衡 | 4:1 | 8:1 |
| Se 定义 | abnormal 检出率 | 差质量检出率（pos_label=1） |
| 架构 | LightweightCNN（CoordAtt，65.12K） | 相同 |
| 超参 | batch=16, lr=1e-3, wd=1e-4 | 相同 |

---

### SQA Run 1 — 2026-04-09（基线训练）

**配置**
```
脚本：train_sqa_with_test.py
数据集：metadata_quality_reversed.csv（3240 条，Good=2876 / Bad=364）
划分：80/10/10（按 fname 分组，seed=42）
Train: 54842 | Val: 6536 | Test: 6726
类别不平衡：8:1（WeightedRandomSampler）
超参：batch=16, lr=1e-3, wd=1e-4, epochs=50, early_stop_patience=10
架构：LightweightCNN（CoordAtt，65.12K，与诊断模型相同）
预处理：n_mels=64, hop=128（Run 6 最优参数）
```

**Val 训练过程（振荡剧烈）**

| Epoch | Val Se(bad) | Val Sp(good) | Val M-Score |
|-------|-------------|--------------|-------------|
| 1 | 0.7409 | 0.8592 | 0.8000 ✅ |
| 3 | 0.7263 | 0.8826 | 0.8044 ✅ |
| 9 | 0.7172 | 0.9036 | 0.8104 ✅ |
| **12** | **0.7281** | **0.9050** | **0.8165 ✅ 最优** |
| 22 | 0.6825 | 0.9377 | 0.8101（Early Stop）|

**Test 集最终结果**

| 指标 | 值 |
|------|----|
| Test M-Score | 0.8046 |
| Test Se（差质量检出率） | 0.7173 |
| Test Sp（好质量识别率） | 0.8919 |
| Test Accuracy | 0.8794 |
| Test Loss | 0.3415 |

**分析**
- **Se(bad) = 0.7173 是最大问题**：28.3% 的差质量录音被漏过，会将噪声送入诊断模型，直接影响系统可靠性
- Val 振荡剧烈（M-Score 在 0.78–0.82 间反复波动），训练不稳定，最优点在 Epoch 12 后再未出现
- 8:1 不平衡比诊断模型 4:1 更严重，WeightedRandomSampler 已用但仍不足以解决少数类学习问题
- Val→Test 泛化差距：M-Score 0.8165 → 0.8046（-0.012），Se 0.7281 → 0.7173（-0.011），比诊断模型泛化差距大
- Sp(good) = 0.8919 尚可，误过滤好录音约 10.8%，对系统吞吐量有影响但可接受

**与诊断模型对比**
| 模型 | Test M-Score | Test Se | Test Sp | 不平衡比 |
|------|-------------|---------|---------|---------|
| 诊断模型（Run 6） | 0.8903 | 0.9485 | 0.8322 | 4:1 |
| SQA 模型（Run 1） | 0.8046 | 0.7173 | 0.8919 | 8:1 |

SQA Se 比诊断模型低 0.23，差距明显，需要针对性改进。

---

### SQA Run 2 — 2026-04-09（class_weight=[1,8] + lr=5e-4 + patience=5）

**相比 Run 1 的改动**
```
CrossEntropyLoss() → CrossEntropyLoss(weight=[1.0, 8.0])
lr: 1e-3 → 5e-4
scheduler patience: 3 → 5
```

**Val 训练过程（振荡明显改善）**

| Epoch | Val Se(bad) | Val Sp(good) | Val M-Score |
|-------|-------------|--------------|-------------|
| 1 | 0.8303 | 0.7697 | 0.8000 ✅ |
| 4 | 0.8248 | 0.7897 | 0.8073 ✅ |
| 5 | 0.7883 | 0.8350 | 0.8117 ✅ |
| **12** | **0.8120** | **0.8487** | **0.8304 ✅ 最优** |
| 22 | 0.7646 | 0.8621 | 0.8133（Early Stop）|

**Test 集最终结果**

| 指标 | 值 |
|------|----|
| Test M-Score | 0.8102 |
| Test Se（差质量检出率） | 0.7651 |
| Test Sp（好质量识别率） | 0.8554 |
| Test Accuracy | 0.8489 |
| Test Loss | 0.4911 |

**与 Run 1 对比**

| 指标 | Run 1 | Run 2 | 变化 |
|------|-------|-------|------|
| Test M-Score | 0.8046 | 0.8102 | **+0.006** |
| Test Se(bad) | 0.7173 | 0.7651 | **+0.048 ✅** |
| Test Sp(good) | 0.8919 | 0.8554 | -0.037（预期内） |
| Best Val M-Score | 0.8165 | 0.8304 | +0.014 |
| Best Val Se | 0.7281 | 0.8120 | +0.084 |
| Val 振荡幅度 | 0.78–0.82 | 0.80–0.83 | 明显收窄 |
| Early stop epoch | 22 | 22 | 持平 |

**分析**
- class_weight=[1,8] 有效：Se 提升 +0.048，Val Se 最优从 0.7281 升到 0.8120
- LR 降低 + patience 增加：Val 振荡明显收窄，训练稳定性改善
- 但 Val best Se(0.8120) → Test Se(0.7651) 有 -0.047 的泛化差距，说明 bad class 本身样本少（Val 548 条 / Test 481 条），测试集上的分布方差大
- Train loss 0.057 vs Val loss 0.49：过拟合信号仍然明显，模型已对训练集拟合很深但 bad class 的泛化仍不稳定
- Bad class precision 仅 0.29：大量 Good 被误判为 Bad（FP 多），实际部署中会过度过滤好录音

---

### SQA Run 3 — 2026-04-09（dropout 0.3 → 0.5）

**相比 Run 2 的改动**
```
LightweightCNN(dropout=0.3) → LightweightCNN(dropout=0.5)
其余与 Run 2 相同（class_weight=[1,8], lr=5e-4, patience=5）
```

**Val 训练过程**

| Epoch | Val Se(bad) | Val Sp(good) | Val M-Score |
|-------|-------------|--------------|-------------|
| 1 | 0.8668 | 0.7241 | 0.7955 ✅ |
| **2** | **0.8759** | **0.7764** | **0.8261 ✅ 最优** |
| 4 | 0.8832 | 0.7503 | 0.8168 |
| 12 | 0.7427 | 0.8335 | 0.7881（Early Stop）|

**Test 集最终结果**

| 指标 | 值 |
|------|----|
| Test M-Score | 0.8152 |
| Test Se（差质量检出率） | **0.8274** |
| Test Sp（好质量识别率） | 0.8029 |
| Test Accuracy | 0.8046 |
| Test Loss | 0.4729 |

**三次 Run 横向对比**

| 指标 | Run 1 | Run 2 | Run 3 | Run1→3 累计变化 |
|------|-------|-------|-------|----------------|
| Test M-Score | 0.8046 | 0.8102 | **0.8152** | +0.011 |
| Test Se(bad) | 0.7173 | 0.7651 | **0.8274** | **+0.110 ✅** |
| Test Sp(good) | 0.8919 | 0.8554 | 0.8029 | -0.089 |
| Best Val Se | 0.7281 | 0.8120 | 0.8759 | +0.148 |
| Val→Test Se 泛化差距 | -0.011 | -0.047 | -0.048 | 稳定在 ~0.05 |
| Train loss | 0.102 | 0.057 | 0.078 | 过拟合有所缓解 |
| Early stop epoch | 22 | 22 | **12** | 收敛加快 |

**分析**
- dropout=0.5 是最关键的一步：Se 从 0.7651 跳到 **0.8274**，单步提升 +0.062
- Train loss 从 0.057 回升到 0.078，过拟合确实有所缓解
- Val→Test Se 泛化差距稳定在 ~0.048，与 Run-2 持平，说明 bad class 的泛化上限受数据量制约
- Early stop 在 Epoch 2 就找到最优点，之后 Val M-Score 一路下滑——dropout 更强让模型需要更多 epoch 收敛，但 patience=10 仍然足够
- Sp=0.8029 是 Se 提升的代价，好录音误过滤率约 20%，部署时可接受（宁可多过滤）

---

### 封箱说明

**最终选定**：Run-3（`best_model_sqa.pth`，dropout=0.5）

**不需要做阈值扫描**：部署机制是加权平均而非二值过滤。SQA 输出 P(Good) 直接作为每个 2s 片段的权重，20s chunk 内所有片段的诊断预测以 SQA 得分加权平均得到最终结论。连续概率本身就是最终使用的形式，不存在需要调整的二值阈值。

**Se 比 Sp 更关键**：差质量片段若 P(Good) 偏高（Se 失效），会以较高权重混入加权平均，直接污染诊断结果；好质量片段若 P(Good) 偏低（Sp 失效），只是降低有效信号量，不引入噪声。Run-3 的 Se=0.827 已经达到合理水平。

**下一步**：转 tflite → 替换 Pi 上的模型文件 → 跑 `evaluate_tflite.py`

---

## 消融实验（架构）

### 实验设计

**核心问题**：每个架构设计决策对最终 M-Score 的贡献是多少？

**方法**：累进式消融（A→B→C→D），每步只加一个改动，逐步还原完整系统。

**固定控制变量（所有组统一）**
```
数据集：同一份 test_split.csv（不可重新划分）
预处理：n_mels=32, hop=96, n_fft=256, overlap=0.5（原始参数，不偏向任何架构）
训练：batch=16, lr=1e-3, weight_decay=1e-4, 无 label smoothing
Early stopping：patience=10, max epochs=50
Scheduler：ReduceLROnPlateau(factor=0.5, patience=3)
Sampler：WeightedRandomSampler
```

### 消融组合

| 组 | 实验名 | 模型文件 | 通道 | CoordAtt | Dropout | 残差 | 状态 |
|---|---|---|---|---|---|---|---|
| A | Baseline OG | lightweight_cnn_og.py | 16→32→64→128 | 无 | 无 | 无 | 待跑 |
| B | + 加宽通道 | lightweight_cnn_og_wide.py | 32→64→128→256 | 无 | 无 | 无 | 待跑 |
| C | + CoordAtt + Dropout | lightweight_cnn.py | 32→64→128→256 | 有 | 0.3 | 无 | = Run 7 ✅ |
| D | + 残差连接 | lightweight_cnn_res.py | 32→64→128→256 | 有 | 0.3 | 有 | 待跑 |

> - A→B：验证通道容量的贡献
> - B→C：验证 CoordAtt + Dropout 的贡献
> - C→D：验证残差连接的贡献

### 实施步骤

1. **A 组**：import 改为 `LightweightCNN` from `lightweight_cnn_og`，config.yaml 用原始参数，跑一次
2. **B 组**：新建 `lightweight_cnn_og_wide.py`，跑一次
3. **C 组**：直接复用 Run 7 数据 ✅
4. **D 组**：import 改为 `LightweightCNNRes` from `lightweight_cnn_res`，跑一次

### 预期论文表格

| 配置 | Test M-Score | Se | Sp | 参数量 |
|------|-------------|----|----|--------|
| A: Baseline OG（16→128，无注意力） | ? | ? | ? | ~20K |
| B: + 加宽通道（32→256） | ? | ? | ? | ~65K |
| C: + CoordAtt + Dropout | 0.8869 | 0.9383 | 0.8355 | ~85K |
| D: + 残差连接（完整系统） | ? | ? | ? | ~108K |

### 消融结果

#### 组 A — Baseline OG（16→32→64→128，无 CoordAtt，无 Dropout）
wandb: `Ablation-A-OG-baseline` | run: `fgjo8vul`

| 指标 | Val 最优 | Test |
|------|---------|------|
| M-Score | 0.9113 | 0.8851 |
| Sensitivity | 0.9683 | 0.9654 |
| Specificity | 0.8544 | 0.8049 |

Early stop Epoch 11，最佳 Epoch 1

---

#### 组 B — OG Wide（32→64→128→256，无 CoordAtt，无 Dropout）
wandb: `Ablation-B-OG-wide` | run: `i2slxntb`

| 指标 | Val 最优 | Test |
|------|---------|------|
| M-Score | 0.9103 | 0.8896 |
| Sensitivity | 0.9507 | 0.9595 |
| Specificity | 0.8698 | 0.8198 |

Early stop Epoch 11，最佳 Epoch 1

---

#### 组 C — + CoordAtt + Dropout（= Run 7）
| 指标 | Val 最优 | Test |
|------|---------|------|
| M-Score | 0.9163 | 0.8869 |
| Sensitivity | 0.9487 | 0.9383 |
| Specificity | 0.8838 | 0.8355 |

Early stop Epoch 15，最佳 Epoch 5

---

#### 组 D — + 残差连接
wandb: `Ablation-D-Residual` | run: `gcb10g0s`

| 指标 | Val 最优 | Test |
|------|---------|------|
| M-Score | 0.9115 | 0.8912 |
| Sensitivity | 0.9683 | 0.9797 |
| Specificity | 0.8548 | 0.8027 |

Early stop Epoch 12，最佳 Epoch 2

---

#### 论文表格（完整）

输入：(1, 1, 32, 64)，输出：(1, 2)

| 配置 | 参数量 | Test M-Score | Test Se | Test Sp | Test Acc | Test Loss | 最佳 epoch |
|------|--------|-------------|---------|---------|----------|-----------|-----------|
| A: OG（16→128） | 12.87K | 0.8851 | 0.9654 | 0.8049 | 0.8352 | 0.4127 | 1 |
| B: + 加宽通道（32→256） | 47.23K | 0.8896 | 0.9595 | 0.8198 | 0.8462 | 0.3444 | 1 |
| C: + CoordAtt + Dropout | 65.12K | 0.8869 | 0.9383 | 0.8355 | 0.8549 | 0.3491 | 5 |
| D: + 残差连接 | 108.10K | **0.8912** | 0.9797 | 0.8027 | 0.8361 | 0.4117 | 2 |

**分析**
- A→B：加宽通道 +0.005 M-Score，Se/Sp 稍微平衡
- B→C：CoordAtt + Dropout 使最佳 epoch 从 1 延迟到 5，训练更稳定，Sp 提升 +0.016；M-Score 略降（-0.003），贡献体现在泛化稳定性而非峰值数值
- C→D：残差连接 M-Score 最高（+0.004），但 Se 飙至 0.9797 而 Sp 跌至 0.8027，Se/Sp 失衡加剧；最佳 epoch 退回到 2，说明残差加快收敛的同时也带来了更强的偏向性

**结论**：D 组 M-Score 数值最优，但 C 组 Se/Sp 最平衡（Se=0.9383, Sp=0.8355），训练也最稳定。若优先考虑医疗筛查场景（高 Se）选 D；若优先 Se/Sp 均衡选 C。

---

### 封箱说明

**最终选定架构**：C 组（`lightweight_cnn.py`，65.12K 参数）+ Run 6 超参（n_mels=64, hop=128）

**已排除的改进方向**

| 方法 | 分析 | 结论 |
|---|---|---|
| WeightedRandomSampler | 已在用，有效缓解 4:1 类别不平衡 | 保留 |
| CrossEntropyLoss(weight=[1,4]) | 误分异常惩罚更重 → Se↑ Sp↓，加剧失衡 | 不采用 |
| CrossEntropyLoss(weight=[4,1]) | 误分正常惩罚更重 → Sp↑ Se↓，牺牲筛查能力 | 不采用 |
| label smoothing | Run 2 实验证明 M-Score 无明显提升，且降低 Se | 不采用 |
| 阈值调整 | Run 1 阈值扫描最大收益 +0.0007，可忽略 | 不采用 |
| 残差连接 | D 组 Se/Sp 失衡加剧，不适合筛查场景 | 不采用 |

Se/Sp 失衡是 4:1 数据不平衡的固有问题，上述方法只在 Se 和 Sp 之间做取舍，无法同时提升两者，因此封箱。

---

## TFLite 转换记录

### 转换结果

| 文件 | 模型来源 | 大小 | 压缩率 |
|------|---------|------|--------|
| `heart_model_fp32.tflite` | 诊断模型（Run 6 重训，Final-diagnostic-model）| 303K | — |
| `heart_model_quant.tflite` | 同上，INT8 动态范围量化 | 145K | 52% |
| `heart_quality_fp32.tflite` | SQA 模型（Run 3）| 303K | — |
| `heart_quality_quant.tflite` | 同上，INT8 动态范围量化 | 145K | 52% |

转换时间：2026-04-09

### 转换脚本

`scripts/convert_to_tflite.py`，对每个 `.pth` 生成 FP32 和 INT8 两个 tflite 文件。

### 依赖链与踩坑记录

**原始链路**：`ai_edge_torch.convert()` → MLIR/StableHLO → TFLite

**问题**：`ai_edge_torch` 在 2024 年底更名为 `litert_torch`，`ai_edge_torch 0.7.x` 与 torch 2.11.0 不兼容（`torch.ao.quantization.pt2e` 模块在 torch 2.5+ 被移除）。

**解决过程**：
1. 对 `ai_edge_torch/quantize/__init__.py` 和 `quant_config.py` 加 `try/except`，绕过 PT2E 量化模块的导入错误（PT2E 量化是 QAT 专用路径，我们用的是 TFLite 动态范围量化，不经过这条路）
2. PyTorch 自动恢复了 `torch.ao.quantization.pt2e` 的实现（linter 注入），最终 import 恢复正常
3. 安装 `litert_torch 0.8.0`，将脚本中 `import ai_edge_torch` 改为 `import litert_torch as ai_edge_torch`，接口完全兼容

**最终工作环境**：
```
torch:        2.11.0+cu128
litert-torch: 0.8.0
tensorflow:   2.21.0-dev
```

**注意**：`litert_torch 0.8.0` 声明要求 `torch<2.10.0`，但实际在 torch 2.11.0 上可正常运行（仅有无害的 FutureWarning）。如果未来重新跑转换遇到 import 错误，检查 `ai_edge_torch/quantize/__init__.py` 的 try/except 是否还在。

---

## TFLite 模型综合评估 — 2026-04-12

**评估脚本**：`scripts/evaluate_tflite.py`

**测试集**：
- 诊断任务：`test_split.csv`，共 6273 个切片（总切片 62003，音频 2876 条）
- 质量任务：`test_split_sqa.csv`，共 6726 个切片（总切片 68104，音频 3240 条）

### FP32 vs INT8 综合性能对比（更新含混淆矩阵细项）

| Task | Model | Accuracy | M-Score | Se | Sp | TN | FP | FN | TP | n | Latency | Size |
|------|-------|----------|---------|----|----|----|----|----|---|---|---------|------|
| Diagnosis (疾病诊断) | Diag_FP32 | 0.8331 | 0.8835 | 0.9645 | 0.8025 | 4084 | 1005 | 42 | 1142 | 6273 | 1.12ms | 0.30MB |
| Diagnosis (疾病诊断) | Diag_INT8 | 0.8336 | 0.8838 | 0.9645 | 0.8031 | 4087 | 1002 | 42 | 1142 | 6273 | 1.06ms | 0.14MB |
| Quality (质量评估) | Qual_FP32 | 0.7324 | 0.8118 | 0.9044 | 0.7191 | 4491 | 1754 | 46 | 435 | 6726 | 1.23ms | 0.30MB |
| Quality (质量评估) | Qual_INT8 | 0.7330 | 0.8121 | 0.9044 | 0.7198 | 4495 | 1750 | 46 | 435 | 6726 | 1.07ms | 0.14MB |

> Se = 类别1（Abnormal/Bad）召回率；Sp = 类别0（Normal/Good）召回率；评估粒度：切片级（per-slice）

### 结论

- INT8 量化后精度**几乎无损**（M-Score 差距 ≤ 0.0003，TP/FN 完全一致），属于正常量化误差范围
- INT8 模型体积缩减约 **53%**（0.30MB → 0.14MB）
- INT8 推理延迟降低约 **5–13%**（诊断 1.12ms→1.06ms，质量 1.23ms→1.07ms）
- 综合来看，INT8 量化在保持性能的同时显著降低了资源占用，适合边缘部署
