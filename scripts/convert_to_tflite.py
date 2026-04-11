import os
import sys
import shutil
import tempfile
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
    转换单个模型：
      1. FP32（已有则跳过）
      2. 动态范围量化 INT8（已有则跳过）
      3. 全整数量化 INT8 via PyTorch → ONNX → onnx2tf（输出 _int8full.tflite）
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

    # --- 转换 1: FP32（已有则跳过） ---
    fp32_output = f"{output_prefix}_fp32.tflite"
    if not os.path.exists(fp32_output):
        print(f"🚀 正在生成 FP32 模型: {os.path.basename(fp32_output)}...")
        edge_model_fp32 = ai_edge_torch.convert(model, (sample_input,))
        edge_model_fp32.export(fp32_output)
    else:
        print(f"⏭️  FP32 已存在，跳过: {os.path.basename(fp32_output)}")

    # --- 转换 2: 动态范围量化 INT8（已有则跳过） ---
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

    # --- 转换 3: 全整数量化 via ONNX → onnx2tf → TFLite ---
    int8full_output = f"{output_prefix}_int8full.tflite"
    print(f"🚀 正在生成全整数量化模型: {os.path.basename(int8full_output)}...")

    try:
        import onnx
        import onnx2tf
        from src.model.lightweight_cnn import CoordAtt  # 导入原始类

        # ── Patch：导出专用的 CoordAtt，expand_as 消除歧义形状 ──────────────
        class _CoordAttExport(CoordAtt):
            """ONNX 导出专用：a_h/a_w 显式 expand，形状统一，onnx2tf 不再困惑"""
            def forward(self, x):
                identity = x
                n, c, h, w = x.size()
                x_h = self.pool_h(x)
                x_w = self.pool_w(x).permute(0, 1, 3, 2)
                y = torch.cat([x_h, x_w], dim=2)
                y = self.act(self.bn1(self.conv1(y)))
                x_h, x_w = torch.split(y, [h, w], dim=2)
                x_w = x_w.permute(0, 1, 3, 2)
                a_h = torch.sigmoid(self.conv_h(x_h)).expand_as(identity)
                a_w = torch.sigmoid(self.conv_w(x_w)).expand_as(identity)
                return identity * a_h * a_w

        # 替换模型内所有 CoordAtt 模块为导出版
        def _patch_coordatt(module):
            for name, child in module.named_children():
                if type(child) is CoordAtt:
                    new_child = _CoordAttExport.__new__(_CoordAttExport)
                    new_child.__dict__ = child.__dict__
                    setattr(module, name, new_child)
                else:
                    _patch_coordatt(child)

        _patch_coordatt(model)
        # ─────────────────────────────────────────────────────────────────────

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path      = os.path.join(tmpdir, "model.onnx")
            calib_path     = os.path.join(tmpdir, "calib.npy")
            saved_model_dir = os.path.join(tmpdir, "saved_model")

            # 1. PyTorch (patched) → ONNX
            print("  [1/3] PyTorch(patched) → ONNX...")
            torch.onnx.export(
                model, sample_input, onnx_path,
                input_names=["input"], output_names=["output"],
                opset_version=18, dynamic_axes=None,
            )

            # 2. 校准数据
            print("  [2/3] 收集校准数据...")
            calib = collect_calibration_npy(metadata_path, cfg)
            np.save(calib_path, calib)

            # 3. onnx2tf → SavedModel
            print("  [3a/3] onnx2tf → SavedModel...")
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=saved_model_dir,
                output_integer_quantized_tflite=False,
                not_use_onnxsim=True,
                disable_strict_mode=True,
                verbosity="warn",
            )

            # 4. SavedModel → TFLite 全整数量化
            print("  [3b/3] TFLiteConverter 全整数量化...")
            calib_data = np.load(calib_path)  # (N,1,64,64)

            # 找 SavedModel 输入形状（onnx2tf 转成 NHWC）
            import tensorflow as tf
            saved = tf.saved_model.load(saved_model_dir)
            infer = saved.signatures["serving_default"]
            input_key = list(infer.structured_input_signature[1].keys())[0]
            input_shape = infer.structured_input_signature[1][input_key].shape.as_list()
            print(f"  SavedModel 输入形状: {input_shape}  key: {input_key}")

            def rep_dataset():
                for i in range(len(calib_data)):
                    s = calib_data[i]  # (1, 64, 64)
                    if len(input_shape) == 4 and input_shape[-1] == 1:
                        # NHWC: (1, 64, 64, 1)
                        s = np.transpose(s, (1, 2, 0))[np.newaxis].astype(np.float32)
                    else:
                        s = s[np.newaxis].astype(np.float32)
                    yield [s]

            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type  = tf.float32
            converter.inference_output_type = tf.float32
            tflite_bytes = converter.convert()

            with open(int8full_output, "wb") as f:
                f.write(tflite_bytes)
            size_mb = os.path.getsize(int8full_output) / (1024 * 1024)
            print(f"✅ 全整数量化完成！大小: {size_mb:.2f}MB → {os.path.basename(int8full_output)}")

    except Exception as e:
        print(f"❌ 全整数量化失败: {e}")
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