import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import litert_torch as ai_edge_torch
import tensorflow as tf
from pathlib import Path

# 1. 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.lightweight_cnn import LightweightCNN
from src.train.dataset.dataset_mel import HeartSoundMelDataset

N_CALIB_SAMPLES = 200  # 校准样本数


def collect_calibration_npy(metadata_path, cfg, n=N_CALIB_SAMPLES, seed=42):
    """采样 mel 切片，返回 numpy array (N, 1, 64, 64)"""
    dataset = HeartSoundMelDataset(
        metadata_path=metadata_path,
        sr=cfg["data"]["sample_rate"],
        segment_sec=cfg["data"]["segment_length"],
        mel_cfg=cfg["mel"]
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), min(n, len(dataset)), replace=False)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    samples = [mel.numpy().astype(np.float32) for mel, _ in loader]
    calib = np.concatenate(samples, axis=0)  # (N, 1, 64, 64)
    print(f"  校准数据形状: {calib.shape}")
    return calib


def convert_model(task_name, pth_path, output_prefix, metadata_path, cfg):
    """
    转换单个模型（全部通过 litert_torch 原生路径）：
      1. FP32（已有则跳过）
      2. 动态范围量化 INT8（已有则跳过）
      3. 全整型量化 INT8（via litert_torch _ai_edge_converter_flags）
    """
    print(f"\n{'='*50}")
    print(f"📦 正在处理模型任务: {task_name.upper()}")
    print(f"源文件: {pth_path}")
    print("="*50)

    if not os.path.exists(pth_path):
        print(f"❌ 错误: 找不到权重文件 {pth_path}，跳过此任务。")
        return

    input_shape = (1, 1, 64, 64)
    sample_input = torch.randn(input_shape)

    model = LightweightCNN(num_classes=2, in_channels=1)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    # ─── 转换 1: FP32（已有则跳过） ───
    fp32_output = f"{output_prefix}_fp32.tflite"
    if not os.path.exists(fp32_output):
        print(f"🚀 正在生成 FP32 模型: {os.path.basename(fp32_output)}...")
        edge_model_fp32 = ai_edge_torch.convert(model, (sample_input,))
        edge_model_fp32.export(fp32_output)
    else:
        print(f"⏭️  FP32 已存在，跳过: {os.path.basename(fp32_output)}")

    # ─── 转换 2: 动态范围量化 INT8（已有则跳过） ───
    quant_output = f"{output_prefix}_quant.tflite"
    if not os.path.exists(quant_output):
        print(f"🚀 正在生成动态量化模型: {os.path.basename(quant_output)}...")
        try:
            edge_model_quant = ai_edge_torch.convert(
                model, (sample_input,),
                _ai_edge_converter_flags={"optimizations": [tf.lite.Optimize.DEFAULT]}
            )
            edge_model_quant.export(quant_output)
            print(f"✅ 动态量化完成")
        except Exception as e:
            print(f"❌ 动态量化失败: {e}")
    else:
        print(f"⏭️  动态量化已存在，跳过: {os.path.basename(quant_output)}")

    # ─── 转换 3: 全整型量化 INT8（已有则跳过） ───
    int8full_output = f"{output_prefix}_int8full.tflite"
    if os.path.exists(int8full_output):
        os.remove(int8full_output)  # 删除旧文件，确保重新生成

    print(f"🚀 正在生成全整型量化模型: {os.path.basename(int8full_output)}...")

    try:
        # 1. 收集校准数据
        calib = collect_calibration_npy(metadata_path, cfg)  # (N, 1, 64, 64)
        if calib is None or len(calib) == 0:
            raise RuntimeError("校准数据为空")

        # 2. 代表数据集生成器（喂给 TFLite converter 做校准）
        def rep_dataset():
            for i in range(len(calib)):
                sample = calib[i:i+1].astype(np.float32)  # (1, 1, 64, 64)
                yield [sample]

        # 3. 通过 _ai_edge_converter_flags 传递全整型量化参数
        #    litert_torch 会将这些 flags 透传给底层的 TFLiteConverter
        edge_model_int8 = ai_edge_torch.convert(
            model, (sample_input,),
            _ai_edge_converter_flags={
                "optimizations": [tf.lite.Optimize.DEFAULT],
                "representative_dataset": rep_dataset,
                "target_spec": {
                    "supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                },
                "inference_input_type": tf.int8,
                "inference_output_type": tf.int8,
            }
        )
        edge_model_int8.export(int8full_output)

        size_mb = os.path.getsize(int8full_output) / (1024 * 1024)
        print(f"✅ 全整型量化完成！大小: {size_mb:.2f}MB → {os.path.basename(int8full_output)}")

    except Exception as e:
        print(f"❌ 全整型量化失败: {e}")
        import traceback; traceback.print_exc()


def main():
    cfg_path = Path(PROJECT_ROOT) / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tasks = [
        {
            "name":     "diagnosis",
            "pth":      os.path.join(PROJECT_ROOT, "checkpoints/best_model.pth"),
            "prefix":   os.path.join(PROJECT_ROOT, "heart_model"),
            "metadata": os.path.join(PROJECT_ROOT, "data/metadata_physionet.csv"),
        },
        {
            "name":     "quality",
            "pth":      os.path.join(PROJECT_ROOT, "checkpoints/best_model_sqa.pth"),
            "prefix":   os.path.join(PROJECT_ROOT, "heart_quality"),
            "metadata": os.path.join(PROJECT_ROOT, "data/metadata_quality_reversed.csv"),
        },
    ]

    for t in tasks:
        convert_model(t["name"], t["pth"], t["prefix"], t["metadata"], cfg)

    print("\n✨ 所有转换任务已完成！")
    print("新文件：heart_model_int8full.tflite  /  heart_quality_int8full.tflite")


if __name__ == "__main__":
    main()
