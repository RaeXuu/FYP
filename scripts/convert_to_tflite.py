import os
import sys
import torch
import ai_edge_torch
import tensorflow as tf

# 1. 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.lightweight_cnn import LightweightCNN 

def convert_model(task_name, pth_path, output_prefix):
    """
    通用转换函数，处理单个模型的 FP32 和 INT8 量化转换
    """
    print(f"\n" + "="*50)
    print(f"📦 正在处理模型任务: {task_name.upper()}")
    print(f"源文件: {pth_path}")
    print("="*50)

    if not os.path.exists(pth_path):
        print(f"❌ 错误: 找不到权重文件 {pth_path}，跳过此任务。")
        return

    # 输入形状：(batch=1, channel=1, n_mels=64, time=64)
    input_shape = (1, 1, 64, 64)
    sample_input = torch.randn(input_shape)

    # 实例化模型 (诊断和质量现在都是二分类)
    model = LightweightCNN(num_classes=2, in_channels=1)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    # --- 转换 1: FP32 版本 ---
    fp32_output = f"{output_prefix}_fp32.tflite"
    print(f"🚀 正在生成 FP32 模型: {os.path.basename(fp32_output)}...")
    edge_model_fp32 = ai_edge_torch.convert(model, (sample_input,))
    edge_model_fp32.export(fp32_output)

    # --- 转换 2: 动态范围量化版本 (INT8) ---
    quant_output = f"{output_prefix}_quant.tflite"
    print(f"🚀 正在生成量化模型: {os.path.basename(quant_output)}...")
    
    tflite_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
    
    try:
        edge_model_quant = ai_edge_torch.convert(
            model, 
            (sample_input,), 
            _ai_edge_converter_flags=tflite_flags
        )
        edge_model_quant.export(quant_output)
        print(f"✅ 转换成功！")
    except Exception as e:
        print(f"❌ 量化转换失败: {e}")

def main():
    # 定义任务清单：[任务名, 权重路径, 输出前缀]
    tasks = [
        ["diagnosis", 
        os.path.join(PROJECT_ROOT, "checkpoints/best_model.pth"), 
        os.path.join(PROJECT_ROOT, "heart_model")],
        
        ["quality",
        os.path.join(PROJECT_ROOT, "checkpoints/best_model_sqa.pth"),
        os.path.join(PROJECT_ROOT, "heart_quality")]
    ]

    for task_name, pth_path, output_prefix in tasks:
        convert_model(task_name, pth_path, output_prefix)
    
    print("\n✨ 所有转换任务已完成！")

if __name__ == "__main__":
    main()