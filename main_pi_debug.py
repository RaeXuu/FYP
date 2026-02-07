import os
import sys
import numpy as np
import tflite_runtime.interpreter as tflite
import yaml

# ==========================================
# 1. 路径修复 (必须放在所有 src 导入之前)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 打印一下路径，方便出问题时排查
print(f"📂 当前运行目录: {CURRENT_DIR}")

# ==========================================
# 2. 动态导入预处理函数
# ==========================================
try:
    # 尝试从 preprocess_inference 导入
    from src.preprocess.preprocess_inference import preprocess_wav_for_pi
    print("✅ 成功从 preprocess_inference 导入函数")
except ImportError:
    try:
        # 如果失败，尝试从 preprocess_pipeline 导入
        from src.preprocess.preprocess_pipeline import preprocess_wav_for_pi
        print("✅ 成功从 preprocess_pipeline 导入函数")
    except ImportError as e:
        print(f"❌ 导入失败！请检查 src/preprocess/ 下是否存在对应的文件。")
        print(f"错误信息: {e}")
        sys.exit(1)

# ==========================================
# 3. 配置与模型设置
# ==========================================
QUALITY_MODEL_PATH = "heart_quality_quant.tflite"
DIAG_MODEL_PATH = "heart_model_quant.tflite"
# 确保这个文件路径在你电脑上是存在的
TEST_WAV = "data/raw/DataSet1/set_a/normal__103_1305031931979_B.wav"

def main():
    # A. 加载 config
    with open(os.path.join(CURRENT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # B. 加载模型
    q_interpreter = tflite.Interpreter(model_path=QUALITY_MODEL_PATH)
    d_interpreter = tflite.Interpreter(model_path=DIAG_MODEL_PATH)
    q_interpreter.allocate_tensors()
    d_interpreter.allocate_tensors()

    q_in_idx = q_interpreter.get_input_details()[0]['index']
    q_out_idx = q_interpreter.get_output_details()[0]['index']
    d_in_idx = d_interpreter.get_input_details()[0]['index']
    d_out_idx = d_interpreter.get_output_details()[0]['index']

    # C. 预处理与推理
    if not os.path.exists(TEST_WAV):
        print(f"❌ 找不到音频文件: {TEST_WAV}")
        return

    print(f"🎧 正在处理: {os.path.basename(TEST_WAV)}")
    tensors = preprocess_wav_for_pi(TEST_WAV, config)

    for i, input_tensor in enumerate(tensors):
        # 第一级：质量
        q_interpreter.set_tensor(q_in_idx, input_tensor)
        q_interpreter.invoke()
        q_pred = np.argmax(q_interpreter.get_tensor(q_out_idx))

        if q_pred == 0:
            print(f"  Segment {i+1}: ⚠️ 噪声拦截 (Poor Quality)")
        else:
            # 第二级：诊断
            d_interpreter.set_tensor(d_in_idx, input_tensor)
            d_interpreter.invoke()
            d_output = d_interpreter.get_tensor(d_out_idx)
            d_pred = np.argmax(d_output)
            res = "Abnormal" if d_pred == 1 else "Normal"
            print(f"  Segment {i+1}: ✨ {res} (置信度: {np.max(d_output):.2f})")

if __name__ == "__main__":
    main()