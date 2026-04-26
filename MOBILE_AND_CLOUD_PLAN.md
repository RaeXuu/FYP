# 移动端部署与边缘-云协同架构方案

> 撰写日期：2026-04-25 | 状态：方案设计阶段

---

## 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户端（手机/Pi）                          │
│                                                                 │
│   电子听诊器 ──→ 音频录制 ──→ SQA 过滤 ──→ 诊断推理 (145KB)      │
│                                     │              │            │
│                                Bad 丢弃      Normal → 本地出结果  │
│                                            Abnormal → 上传云端   │
│                                           不确定  → 上传云端    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         云平台                                   │
│                                                                 │
│   Abnormal 样本入库 ──→ StethoLM 二次诊断 ──→ 结构化报告           │
│                         (4B 参数心音大模型)                        │
│                                                                 │
│   数据集持续积累 ──→ 定期重训边缘模型 ──→ OTA 推送更新             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、移动端（iOS）

### 技术栈

| 层 | 技术 | 用途 |
|----|------|------|
| 开发环境 | Xcode + SwiftUI | IDE 和 UI 框架 |
| AI 推理 | TensorFlow Lite Swift | 加载并运行 `.tflite` 模型 |
| 音频采集 | AVAudioEngine | 麦克风录音，设置采样率 |
| 信号处理 | Accelerate (vDSP) | 带通滤波、STFT、Mel 频谱 |
| 本地存储 | Core Data / FileManager | 缓存录音和历史结果 |

### 模型清单

| 文件 | 大小 | 用途 |
|------|------|------|
| `heart_quality_int8full.tflite` | 145KB | SQA 质量过滤 |
| `heart_model_int8full.tflite` | 145KB | Normal/Abnormal 诊断 |

### 预处理流水线

```
Python (训练端)                        Swift (iOS 端)
─────────────────                ─────────────────
load_wav(target_sr=2000)   →     AVAudioEngine (采样率 2000Hz)
apply_bandpass(25-400Hz)   →     vDSP FIR/IIR 带通滤波
segment_audio(2s, 0.5 ov)  →     stride loop (seg_len=4000, hop=2000)
normalize(max_abs)         →     手动逐窗口归一化
logmel_fixed_size(64×64)   →     vDSP DFT + Mel filterbank
```

### 双阶段推理流程

```
每次推理（一个 2s 窗口）：

1. Mel 频谱 float32 → 量化为 INT8
2. SQA 推理 → P(Bad)
3. 若 P(Bad) > 0.5 → 丢弃（差质量片段）
4. 若 P(Bad) ≤ 0.5 → 诊断推理 → P(Normal)
5. 累积所有有效窗口，加权平均得出文件级预测
```

### 结果输出

```
筛查完成 ──────────────────────────
│                                  │
Normal (置信度 > 阈值)          Abnormal / 不确定
│                                  │
本地显示 "正常"                   上传云端
建议定期体检                  ──→ StethoLM 二次诊断
                                  ──→ 推送报告 + 就医建议
```

### 开发步骤

1. Xcode 创建 SwiftUI 项目，集成 TFLite Swift 包
2. 实现 AVAudioEngine 录音（2000Hz, 单声道）
3. 用 vDSP 实现带通滤波器（25-400Hz）
4. 实现 mel 频谱计算
5. 加载两个 INT8 全整型 TFLite 模型
6. 实现双阶段推理流水线
7. 简单 UI：录音按钮 + 结果显示
8. 真机测试

---

## 二、云平台

### 核心组件：StethoLM

| 项目 | 详情 |
|------|------|
| 全称 | StethoLM: Audio Language Model for Cardiopulmonary Analysis |
| 架构 | COLA（EfficientNet 音频编码器）+ MedGemma-4B-IT |
| 参数量 | 4B（LLM 骨干） |
| 模型大小 | ~713MB（LoRA adapter + 音频编码器，不含 base LLM） |
| 训练数据 | StethoBench（7 数据集，16,125 条录音，77K 指令对） |
| 能力 | 二分类 / 检测 / 报告生成 / 推理 / 鉴别诊断 / 比较 / 定位 |
| 许可证 | 研究用途 |
| GitHub | https://github.com/askyishan/StethoLM |
| HuggingFace | https://huggingface.co/askyishan/StethoLM |

### 云端流水线

```
手机上传 Abnormal WAV
        │
        ▼
   [API Gateway] 接收请求
        │
        ▼
   [预处理] 重采样 → 滤波 → 分段（与边缘一致）
        │
        ▼
   [StethoLM] 多任务推理
   ├── 二分类：Normal / Abnormal
   ├── 特征检测：杂音类型、强度、时相
   ├── 鉴别诊断：ASD / VSD / MR / MS / ...
   └── 报告生成：结构化文本
        │
        ▼
   [结果存储 + 推送] 存入数据库，通知用户
```

### 最小可行云平台（论文概念验证）

不需要部署完整的 GPU 服务器：

```
方案 A：Hugging Face Spaces (免费 GPU)
    - 部署 StethoLM 推理 endpoint
    - iOS 端 HTTP POST 音频文件
    - 返回 JSON 诊断结果

方案 B：GitHub 仓库作为 "云存储"
    - Abnormal 样本通过 git push 上传到 FYP_cloud 仓库
    - 周期性手动跑 StethoLM 评估
    - 概念验证足够，不需要实时
```

### 数据集持续进化

```
部署后累积的 Abnormal 样本：
  ├── 自动标注（StethoLM 输出作为伪标签）
  ├── 人工复核（医学合作者抽查）
  └── 加入训练集 → 重训边缘 CNN → OTA 推送更新
```

---

## 三、与当前项目的关系

### 已完成的

| 组件 | 状态 |
|------|------|
| SQA 模型 (M-Score 0.815, Se 0.827) | ✅ 训练完成 |
| 诊断模型 (M-Score 0.890, Se 0.949) | ✅ 训练完成 |
| FP32 TFLite 导出 | ✅ |
| INT8 动态量化 TFLite | ✅ |
| INT8 全整型量化 TFLite | ✅ |
| PC 端精度评估 | ✅ |
| Pi 端评估脚本（含预热和统计） | ✅ |

### 待完成

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| P0 | Pi 上跑 `evaluate_tflite_on_pi.py --mode all` | 1h |
| P1 | 论文 Method / Experiment 章节 | 3-5 天 |
| P2 | iOS App 开发 | 1-2 周 |
| P3 | StethoLM 云端部署（Hugging Face Spaces） | 2-3 天 |

---

## 四、可行性评估

### 手机端

| 关注点 | 评估 |
|--------|------|
| 模型体积 | 290KB（两个模型合计），完全无感 |
| 推理延迟 | <0.5ms/窗口（Neural Engine），完全不卡 UI |
| 内存占用 | <20MB（含 TFLite runtime） |
| 预处理延迟 | mel 计算是主要开销，需 Profile 实测 |
| 隐私 | 全部本地计算，仅 Abnormal 上传 |
| 开发难度 | 中等（Swift 音频处理需要学习） |

### 云平台

| 关注点 | 评估 |
|--------|------|
| StethoLM 可用性 | 已开源，有完整推理代码 |
| 部署难度 | 低（Hugging Face Spaces 免费 GPU） |
| 推理成本 | 免费额度内足够论文 demo |
| 延迟 | 3-10s（GPU 推理 + 网络传输），可接受 |

---

> 下一步：Pi 端评估 → 论文写作 → 手机端开发
