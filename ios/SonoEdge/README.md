# SonoEdge — iOS 心音筛查 App

基于 `FYP_raspberry_pi` 仓库的 Pi 端推理流水线，移植到 iOS (SwiftUI + TensorFlowLite)。

**连接方式**: 通过 CoreBluetooth 直连 ESP32 电子听诊器 (service UUID `4fafc201-...`), 接收 int16 PCM @ 2000Hz, 与 Pi 端 BLE 采集逻辑完全一致。
**流式推理**: 音频持续来电 → 每 20s 一块 → 块内 19 个窗口逐窗推理 + 实时 UI 更新 (进度条 + 逐窗分数), 完全对齐 Pi 端 `run_inference()` 的 `on_window` 机制。带 `maxsize=1` 队列反压, 推理积压时丢弃旧块保留最新。

## 项目结构

```
SonoEdge/
├── Package.swift                          # SPM 依赖 (TensorFlowLiteSwift 2.16+)
├── Resources/Info.plist                   # 麦克风权限, bundle 信息
└── Sources/
    ├── App/
    │   ├── SonoEdgeApp.swift           # @main 入口
    │   └── ContentView.swift              # UI (连接 ESP32, 流式推理, 窗口详情)
    ├── Audio/
    │   ├── BLERecorder.swift              # CoreBluetooth 对接 ESP32, maxsize=1 反压队列
    │   └── AudioProcessor.swift           # 零相位带通滤波 + librosa 兼容 Mel 频谱
    ├── Inference/
    │   ├── TFLiteEngine.swift             # INT8 全整型 TFLite 推理封装
    │   └── InferencePipeline.swift        # 流式: SQA 门控 → 诊断 → 加权平均 + 逐窗进度回调
    └── Models/
        ├── heart_quality_int8full.tflite   # SQA 质量评估模型 (145KB)
        └── heart_model_int8full.tflite     # 诊断模型 (145KB)
```

## 运行时流程

```
ESP32 电子听诊器 (BLE)
    │  int16 PCM @ 2000Hz, notification
    ▼
CoreBluetooth → BLERecorder (accumulate → 20s chunk → maxsize=1 queue)
    │                                   ▲ 推理积压 → 丢弃旧块
    ▼
────── per chunk ──────
    │
ButterworthBandpass (order-5, 25–400Hz, zero-phase filtfilt)
    │
    ▼
∑── per window (2s, 50% overlap, 19 windows) ──□
    │                                           ▲ 逐窗进度回调 → UI 实时更新
    peak normalize → LogMel (64×64, librosa-compatible)
    │
    ├── SQA model (int8full) → softmax[0] = P(Good)
    │       │
    │       ├── < 0.65 → skip (UI: ✗)
    │       └── >= 0.65 → pass (UI: ✓)
    │
    └── Diag model (int8full) → softmax[0] = P(Normal)
            │
            ▼
    running weighted avg (所有已处理有效窗口) → stream → UI 进度条
            │
            ▼  (全部 19 窗完成后)
    final label: Normal / Abnormal / LowQuality
```

## 与 Pi 端 (FYP_raspberry_pi) 的对齐

每个关键模块都以 Pi 代码为基线一一对齐：

| 组件 | Pi 端 (`main_pi.py`) | iOS 端 (`SonoEdge`) | 对齐方式 |
|------|----------------------|------------------------|---------|
| 音频输入 | ESP32 BLE, int16 字节流 | CoreBluetooth 直连同一 ESP32 | 同硬件, 同协议 |
| 流式队列 | `asyncio.Queue(maxsize=1)` + drop oldest (line 105-113) | `pendingChunk` + `queueLock`, 丢弃旧块 | 一致 |
| 逐窗进度 | `on_window` callback 每 2 窗口更新 OLED (line 181-184) | `WindowProgressCallback` 每窗口更新 UI | 更细粒度 (每窗) |
| int16→float | `np.frombuffer(raw, int16) / 32768.0` | `Float(Int16(ptr)) / 32768.0` | 等价 |
| 带通滤波 | `scipy.signal.butter(5, [25,400], 'band')` + `sosfiltfilt` | scipy SOS 系数硬编码, `sosfilt` + forward-backward | 同系数, 同算法 |
| 窗口长度 | `SEG_DURATION=2.0`, 4000 samples | `kSegSamples = 4000` | 一致 |
| 重叠 | `OVERLAP=0.5`, hop=2000 samples | `kHopSamples = 2000` | 一致 |
| 归一化 | `max_val = np.max(np.abs(window))` per-window | `MelSpectrogram.compute` 内 `window.map(abs).max()` | 一致 |
| STFT | `librosa.stft` (center=True, hann, n_fft=256, hop=128) | `vDSP.FFT` + hann window + n_fft//2 center pad | 参数一致 |
| Mel filterbank | `librosa.feature.melspectrogram` (64 bands, fmin=25, fmax=400, norm=slaney) | 三角滤波器矩阵 (Hz→Mel→Hz→bin) + Slaney 归一化, static 缓存 | 等价 |
| Power to dB | `librosa.power_to_db` = `10*log10(S + 1e-6)`, top_db=80 | `10.0 * log10(max(val, 1e-6))` + top_db=80 clamp | 一致 |
| `fix_length` | `librosa.util.fix_length(size=64, axis=1)` | 截断或零填充到 64 帧 | 一致 |
| 模型输入 | float32 `(1,1,64,64)` | 量化为 INT8 → `(1,1,64,64)` | 量化公式: `round(f/scale + zp)` |
| SQA softmax 索引 | `q_probs[0]` = P(Good) | `sqaProbs[0]` | 一致 |
| SQA 阈值 | `SQA_THRESHOLD = 0.65` | `sqaThreshold = 0.65` | 一致 |
| Diag softmax 索引 | `d_probs[0]` = P(Normal) | `diagProbs[0]` | 一致 |
| Diag 阈值 | `DIAG_THRESHOLD = 0.5` | `diagThreshold = 0.5` | 一致 |
| 加权平均 | `sum(sqa * normal) / sum(sqa)` | `zip(weights, normals).reduce...` | 一致 |

参考文件:
- `FYP_raspberry_pi/main_pi.py` — 主推理循环
- `FYP_raspberry_pi/src/preprocess/filters.py` — `apply_bandpass`
- `FYP_raspberry_pi/src/preprocess/mel.py` — `logmel_fixed_size`
- `FYP_raspberry_pi/src/preprocess/segment.py` — `segment_audio`
- `FYP_raspberry_pi/config.yaml` — 所有超参数

## 模型说明

使用 **INT8 全整型量化** TFLite 模型（非 Pi 端当前的动态量化 `.quant.tflite`）:

- `heart_quality_int8full.tflite` — SQA 模型, 输入 INT8, 输出 INT8
- `heart_model_int8full.tflite` — 诊断模型, 输入 INT8, 输出 INT8

选择全整型的原因: iOS Neural Engine / CPU 对 INT8 推理有更好的硬件加速支持。

模型来源: `FypProj/scripts/convert_to_tflite.py` 生成。

## 构建

**前置条件**: Xcode 15+, macOS 14+, 有效的 Apple Developer account (真机测试需要)

```bash
cd ios/SonoEdge
open Package.swift
```

Xcode 会自动解析 SPM 依赖, 之后:
1. 在 scheme 中选择目标 (iPhone 真机或模拟器)
2. `Product → Run` 或 `Cmd+R`

## 已知限制 & 待验证

1. ~~**Butterworth 滤波器系数**~~: 已改为 scipy `output='sos'` 硬编码, 不再运行时转换。

2. ~~**Mel 频谱精度**~~: 已对齐 librosa 0.11.0 — 包括 Slaney 归一化 (`norm='slaney'`)、`top_db=80` 裁剪、零填充 STFT (`pad_mode='constant'`)。Mel filterbank 已缓存为 `static let`。仍建议端到端验证 (同一段音频, 对比 iOS 和 Pi 端生成的 mel tensor, 允许浮点误差 < 1e-5)。

3. **INT8 全整型 vs 动态量化**: Pi 端当前使用 `heart_*_quant.tflite` (动态量化, float32 in/out)。iOS 使用 `heart_*_int8full.tflite` (全整型, int8 in/out)。两个模型的预量化精度已评估, 但 iOS 端对 INT8 输入的量化和 INT8 输出的反量化是否正确需要验证。

4. **录音采样率**: AVAudioEngine 原生支持 2000Hz, 但默认硬件可能不完全匹配。建议真机测试时录制一段已知频率的正弦波验证。

5. ~~**实时性能**~~: `InferencePipeline.run()` 已改为 `async`, 在后台线程执行推理, UI 更新通过 `await MainActor.run` 回到主线程, 流式进度可正常推进。

## 审查重点

如果你在 review 这些代码, 建议关注:

1. **数值正确性**: `AudioProcessor.swift` 中的 STFT、mel filterbank、dB 转换是否与 librosa 的输出一致。
2. **量化逻辑**: `TFLiteEngine.swift` 的 `quantizeInput` / `dequantizeOutput` 公式是否与 TensorFlow Lite 的量化规范一致。
3. **索引约定**: SQA 模型用 `probs[0]` 作为 Good 概率 (>= 0.65 通过), 诊断模型用 `probs[0]` 作为 Normal 概率 — 与训练时的标签编码一致, 已与 Pi 端对齐。
4. **内存安全**: `BLERecorder.swift` 中的 `withUnsafeBytes` 和 `AudioProcessor.swift` 中的 `withUnsafeBytes` / `withUnsafeMutableBufferPointer` 调用。
5. **线程模型**: 推理在后台线程执行, UI 更新通过 `MainActor` async 回调; BLE delegate 在主队列 (`queue: .main`), 确保 `@Published` 属性安全更新。
