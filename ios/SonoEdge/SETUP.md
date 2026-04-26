# SonoEdge 环境搭建指南

## 前提条件

- macOS 14+ (Sonoma 以上)
- Xcode 15+ (带 Swift 5.9)
- 一台运行 iOS 的 iPhone（用于真机测试蓝牙 BLE）

## 快速开始

### 1. 拷贝项目

将 `ios/SonoEdge/` 文件夹拷贝到 Mac 上。

### 2. 打开项目

在终端中执行：

```bash
cd ios/SonoEdge
open Package.swift
```

Xcode 会自动打开项目，并解析 TensorFlowLiteSwift 依赖（首次下载可能需要几分钟）。

### 3. 配置 Xcode

- 进入 Project Navigator → 选中 SonoEdge target
- Signing & Capabilities → 选择你的 Developer Team
- 添加 Capability: Background Modes → 勾选 **Uses Bluetooth LE accessories**

### 4. 运行

连接 iPhone → `Cmd+R` 运行。

---

## 调试说明

**必须用真机**: CoreBluetooth BLE 在 iOS 模拟器上不可用，所有测试都必须连接 iPhone 真机。

### 调试流程

1. iPhone 用数据线连接 Mac
2. 确保 iPhone 已开启蓝牙
3. Xcode 中选择你的 iPhone 作为 Run Destination
4. `Cmd+R` 编译运行
5. Xcode 控制台 (Debug Area) 会输出:
   - BLE 扫描、连接、订阅状态
   - 每个窗口的 SQA 分数和诊断结果
   - 每块推理总耗时

### 分享给他人

如果只是给朋友试用（无付费开发者账号）:
- 朋友的 iPhone 用数据线连你的 Mac
- Xcode 选他的 iPhone → `Cmd+R` 直接安装
- 注意: 免费 Apple ID 签名的 app **7 天后过期**，需要重新安装

---

## 加速依赖下载

如果 TensorFlowLite Swift 包下载较慢，可以提前克隆缓存：

```bash
git clone https://github.com/tensorflow/tensorflow.git --depth=1 /tmp/tf
```

或者使用 Xcode 菜单：**File → Packages → Resolve Package Versions**。
