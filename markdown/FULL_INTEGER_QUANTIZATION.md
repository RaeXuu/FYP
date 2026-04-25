# 全整型量化 (Full Integer Quantization) 记录

---

## 背景

项目已有两种 TFLite 模型：
- **FP32**：`heart_model_fp32.tflite` / `heart_quality_fp32.tflite`（303KB）
- **INT8 动态范围量化**：`heart_model_quant.tflite` / `heart_quality_quant.tflite`（145KB，仅权重量化为 INT8，激活值仍为 FP32）

动态量化在 x86 CPU 上已有不错表现，但：
- 输入/输出仍是 FP32，每层计算前后有 Quantize/Dequantize 开销
- Edge TPU、DSP、部分 NPU 只支持纯 INT8 计算图
- ARM Cortex-M 等 MCU 的 TFLite Micro 要求全整型模型
- 全整型可以在 ARM NEON 上获得更优的 SIMD 吞吐

目标：生成真正的全整型量化模型——**输入 INT8、内部算子全部 INT8、输出 INT8**。

---

## 全整型 vs 动态量化的区别

| 特性 | FP32 基准 | INT8 动态量化 | INT8 全整型 |
|------|----------|-------------|-----------|
| 权重存储 | FP32 | INT8 | INT8 |
| 权重计算 | FP32 | INT8→FP32→INT8 | INT8 |
| 激活值存储 | FP32 | FP32 | INT8 |
| 激活值计算 | FP32 | FP32 | INT8 |
| 模型输入 | FP32 | FP32 | **INT8** |
| 模型输出 | FP32 | FP32 | **INT8** |
| Edge TPU | 不支持 | 不支持 | **支持** |
| ARM NEON 加速 | 一般 | 一般 | **最优** |
| 模型体积 | 303KB | 145KB | 145KB |

---

## 实现路径探索

### 路径 A：PyTorch → ONNX → onnx2tf → TFLiteConverter（废弃）

原始代码使用此路径，但存在两个问题：

1. **输入/输出类型错误**：`converter.inference_input_type = tf.float32`，并未实现真正的全整型量化
2. **onnx2tf 兼容性故障**：torch 2.11 的 dynamo 导出器生成的 ONNX 图中，CoordAtt 的 `expand_as` 操作被分解为 Expand/Mul 节点，onnx2tf 无法正确处理 shape 广播：

```
ValueError: Dimensions must be equal, but are 8 and 64
input shapes: [1,8,1,64], [1,64,64,64]
```

这是 onnx2tf 的已知局限，无法通过调整模型前向逻辑规避。

### 路径 B：litert_torch 原生全整型量化（采用）

`litert_torch.convert()` 有 `_ai_edge_converter_flags` 参数，会透传到底层 `tf.lite.TFLiteConverter` 的 `setattr()`。可以直接传递全整型量化的全部参数：

```python
edge_model = ai_edge_torch.convert(
    model, (sample_input,),
    _ai_edge_converter_flags={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": rep_dataset,        # 校准数据生成器
        "target_spec": {
            "supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        },
        "inference_input_type": tf.int8,              # 输入为 INT8
        "inference_output_type": tf.int8,             # 输出为 INT8
    }
)
```

转换流水线：`PyTorch → torch.export → StableHLO → SavedModel → TFLiteConverter(全整型) → .tflite`

完全绕开了 ONNX/onnx2tf，无兼容性问题。

---

## 转换结果

| 文件 | 任务 | 大小 | 转换时间 |
|------|------|------|---------|
| `heart_model_int8full.tflite` | 诊断（Normal/Abnormal） | 0.14MB | 2026-04-25 |
| `heart_quality_int8full.tflite` | 质量（Good/Bad） | 0.14MB | 2026-04-25 |

转换日志确认了全整型标志生效：
```
fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: INT8
```

---

## 精度评估

**测试环境**：x86 CPU, XNNPACK delegate, 单线程推理

**校准数据**：每任务 200 个 mel 切片（`N_CALIB_SAMPLES=200`），随机采样

### 诊断模型（test_split.csv, n=6273）

| Model | Accuracy | M-Score | Se | Sp | TN | FP | FN | TP | Latency | Size |
|-------|----------|---------|----|----|----|----|----|----|---------|------|
| FP32 | 0.8331 | 0.8835 | 0.9645 | 0.8025 | 4084 | 1005 | 42 | 1142 | 1.19ms | 0.30MB |
| INT8 动态 | 0.8336 | 0.8838 | 0.9645 | 0.8031 | 4087 | 1002 | 42 | 1142 | 1.35ms | 0.14MB |
| **INT8 全整型** | 0.8325 | **0.8828** | 0.9637 | 0.8019 | 4081 | 1008 | 43 | 1141 | 1.40ms | 0.14MB |

- M-Score 下降 **0.0007**，TP 减少 1，FN 增加 1 → 精度几乎无损
- Se 下降 0.0008，Sp 下降 0.0006

### 质量模型（test_split_sqa.csv, n=6726）

| Model | Accuracy | M-Score | Se | Sp | TN | FP | FN | TP | Latency | Size |
|-------|----------|---------|----|----|----|----|----|----|---------|------|
| FP32 | 0.7324 | 0.8118 | 0.9044 | 0.7191 | 4491 | 1754 | 46 | 435 | 1.48ms | 0.30MB |
| INT8 动态 | 0.7330 | 0.8121 | 0.9044 | 0.7198 | 4495 | 1750 | 46 | 435 | 1.42ms | 0.14MB |
| **INT8 全整型** | 0.7331 | **0.8102** | 0.9002 | 0.7203 | 4498 | 1747 | 48 | 433 | **1.16ms** | 0.14MB |

- M-Score 下降 **0.0016**，TP 减少 2，FN 增加 2 → 精度几乎无损
- Se 下降 0.0042，Sp 上升 0.0012
- 延迟比 FP32 快 **21.6%**（1.16ms vs 1.48ms）

### 评估脚本改动

为兼容全整型模型的 INT8 输入/输出，评估脚本新增了自动量化/反量化逻辑：

```python
# 检测输入/输出 dtype
is_int8_in  = input_dtype in (np.int8, np.uint8)
is_int8_out = output_dtype in (np.int8, np.uint8)

# INT8 输入：手动量化
if is_int8_in:
    input_q = (input_data / input_scale + input_zp)
    input_q = np.clip(input_q, -128, 127).astype(np.int8)

# INT8 输出：手动反量化
if is_int8_out:
    output_data = (output_raw.astype(np.float32) - output_zp) * output_scale
```

---

## x86 延迟分析

x86 CPU 上全整型延迟改善不显著（甚至略慢），原因：

1. **FP32 在 x86 上有 AVX/AVX2 深度优化**，而 INT8 SIMD（AVX-512 VNNI / AVX-VNNI）在当前测试环境未启用
2. **XNNPACK delegate 对 FP32 的优化更成熟**
3. **Python 层面的手动量化引入了额外开销**（NumPy clip/astype）

在目标部署平台（ARM Cortex-A / Edge TPU / DSP）上，全整型的加速效果会更明显：

| 平台 | FP32 加速方式 | INT8 加速方式 | 全整型预期收益 |
|------|-------------|-------------|-------------|
| ARM Cortex-A | NEON FP32 | NEON INT8 dot-product | 2-4x |
| ARM Cortex-M | 不支持 | TFLite Micro INT8 | 必需 |
| Edge TPU | 不支持 | 仅支持 INT8 | 必需 |
| DSP (Hexagon/Cadence) | 有限 | 原生 INT8 | 3-5x |

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/convert_to_tflite.py` | 全流程转换（FP32 / 动态INT8 / 全整型INT8），全部通过 litert_torch |
| `scripts/evaluate_tflite.py` | 全模型评估，自动适配 INT8 输入/输出的量化反量化 |
| `heart_model_int8full.tflite` | 诊断模型全整型 |
| `heart_quality_int8full.tflite` | 质量模型全整型 |
| `src/model/lightweight_cnn.py` | 模型架构（LightweightCNN + CoordAtt, 65.12K 参数） |
| `config.yaml` | 预处理参数（n_mels=64, hop=128） |
| `checkpoints/best_model.pth` | 诊断模型权重 |
| `checkpoints/best_model_sqa.pth` | 质量模型权重 |

---

## 部署注意事项

1. **输入预处理**：部署时需对 mel 频谱做与训练一致的量化（scale / zero_point 由 TFLite 模型内置），推理代码读取 `input_details[0]['quantization']` 获取参数后量化即可
2. **输出后处理**：读取 `output_details[0]['quantization']` 反量化得到 logits，再 softmax/argmax
3. **校准数据代表性**：当前用 200 个随机采样切片，若精度不满足可增至 500-1000 或使用全量训练集
4. **算子兼容性**：模型仅用 Conv2D / DepthwiseConv2D / AvgPool / MaxPool / ReLU / Sigmoid / FC，均为 TFLite INT8 内置算子，无兼容性问题

---

## 后续可尝试

- [ ] 在 Raspberry Pi 上跑 `evaluate_tflite.py` 对比 ARM NEON INT8 vs FP32 的延迟
- [ ] 使用 `tensorflow/lite/tools/benchmark` 做 C++ 原生 benchmark（避免 Python 开销）
- [ ] 增加校准样本数（200 → 500）观察精度变化
- [ ] 尝试 INT8 量化感知训练（QAT），理论上可进一步减小精度损失

---

## 后续压缩方向探索

当前模型 145KB（INT8），精度 M-Score 0.89。以下方向按优先级排列：

### 1. 剪枝 (Pruning)

| 项目 | 说明 |
|------|------|
| 原理 | 移除接近零的权重，稀疏化后再量化 |
| 预期收益 | 体积再减 30–50%，延迟同步下降 |
| 风险 | 低。65K 参数的小 CNN 通常有冗余，结构化剪枝后 fine-tune 几个 epoch 即可恢复精度 |
| 推荐工具 | `torch.nn.utils.prune` 或 `tensorflow-model-optimization` |
| 实施步骤 | 训练后全局稀疏剪枝 → 移除零权重 → fine-tune 5–10 epoch → 重新导出 TFLite |

### 2. INT4 量化

| 项目 | 说明 |
|------|------|
| 原理 | 权重和激活值量化为 4-bit |
| 预期收益 | 体积约 72KB（再减一半） |
| 风险 | **高**。小模型对低比特敏感，尤其 Se 可能明显下降。需要在 Pi 上实测精度 |
| 推荐工具 | `litert_torch` generative quantize recipe（INT4 + blockwise） |
| 前提 | 当前 INT8 已经接近无损，INT4 是进一步探索 |

### 3. 结构化剪枝 (Structured Pruning)

| 项目 | 说明 |
|------|------|
| 原理 | 按通道/层级裁剪，缩减模型宽度而非稀疏化 |
| 预期收益 | 降低 MACs 而非仅体积，延迟收益更直接 |
| 风险 | **中高**。需要重训，且 LightweightCNN 每层通道数（32→64→128→256）是精心调过的，砍错层可能崩 |
| 建议 | 先对每层做 sensitivity analysis（逐层置零测精度损失），再决定砍哪层 |

### 4. 输入降维

| 项目 | 说明 |
|------|------|
| 原理 | Mel 频谱从 64×64 缩小到 32×32，输入张量缩小 4 倍 |
| 预期收益 | 模型参数和计算量大幅缩减 |
| 风险 | **高**。必须从零重训，且频率分辨率减半可能丢失关键病理信息 |
| 建议 | 仅在当前方案到达瓶颈时考虑 |

### 5. 算子融合 (Operator Fusion)

| 项目 | 说明 |
|------|------|
| 原理 | Conv + BN + ReLU 合并为单一算子 |
| 预期收益 | 推理延迟降低 10–20% |
| 风险 | **极低**。TFLite 转换器会自动做一部分，不需要重训 |
| 实施 | 导出 TFLite 时已自动应用（`tf.lite.Optimize.DEFAULT`） |

### 优先级建议

```
精度优先 → 不动了，145KB 足够小
延迟优先 → 先排查 mel 预处理是否是真正瓶颈
体积优先 → 剪枝（风险最低、收益可预期）
```

---

> 转换日期：2026-04-25
