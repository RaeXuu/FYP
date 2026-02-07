import numpy as np
import tflite_runtime.interpreter as tflite
import bluetooth
import os
import sys

# 1. 确保根目录在路径中，以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. 导入你的预处理逻辑
# 根据 tree 结构，预处理流水线位于 src/preprocess/preprocess_pipeline.py
from src.preprocess.preprocess_pipeline import run_mel_extraction

# =========================
# 配置参数
# =========================
SAMPLE_RATE = 2000
WINDOW_SIZE = SAMPLE_RATE * 2 # 2秒窗口
CHUNK_SIZE = 1024

# 根据 tree 结果，模型就在当前目录下
QUALITY_MODEL_PATH = "heart_quality_quant.tflite"
DIAG_MODEL_PATH = "heart_model_quant.tflite"

def main():
    # 3. 初始化双级 TFLite 解释器
    print("🚀 正在加载双级模型...")
    q_interpreter = tflite.Interpreter(model_path=QUALITY_MODEL_PATH)
    d_interpreter = tflite.Interpreter(model_path=DIAG_MODEL_PATH)

    q_interpreter.allocate_tensors()
    d_interpreter.allocate_tensors()

    # 获取输入/输出详情
    q_input_idx = q_interpreter.get_input_details()[0]['index']
    q_output_idx = q_interpreter.get_output_details()[0]['index']
    d_input_idx = d_interpreter.get_input_details()[0]['index']
    d_output_idx = d_interpreter.get_output_details()[0]['index']

    # 4. 建立蓝牙连接
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    print("📡 等待 ESP32 连接...")
    client_sock, address = server_sock.accept()
    print(f"✅ 已连接到: {address}")

    pcm_buffer = bytearray()

    try:
        while True:
            # 5. 持续接收数据
            data = client_sock.recv(CHUNK_SIZE)
            if not data: break
            pcm_buffer.extend(data)

            # 6. 当满足 2 秒数据量时启动双级推理
            if len(pcm_buffer) >= WINDOW_SIZE * 2:
                # 原始音频处理
                audio_np = np.frombuffer(pcm_buffer[:WINDOW_SIZE*2], dtype=np.int16).astype(np.float32) / 32768.0
                
                # 特征提取：生成 (1, 1, 32, 64) 的输入
                input_tensor = run_mel_extraction(audio_np)

                # --- 第一级：质量关卡 ---
                q_interpreter.set_tensor(q_input_idx, input_tensor)
                q_interpreter.invoke()
                q_pred = np.argmax(q_interpreter.get_tensor(q_output_idx))

                if q_pred == 0: # 0 代表 Poor_Quality
                    print("⚠️ 信号质量差，跳过诊断。")
                    client_sock.send("ERR:POOR_QUALITY")
                else:
                    # --- 第二级：疾病诊断 ---
                    d_interpreter.set_tensor(d_input_idx, input_tensor)
                    d_interpreter.invoke()
                    d_output = d_interpreter.get_tensor(d_output_idx)
                    d_pred = np.argmax(d_output)
                    
                    result = "Abnormal" if d_pred == 1 else "Normal"
                    conf = np.max(d_output)
                    print(f"🏥 诊断结果: {result} (置信度: {conf:.2f})")
                    client_sock.send(f"RES:{result}")

                # 7. 滑动窗口：移除前 0.5 秒数据
                overlap_size = int(SAMPLE_RATE * 0.5 * 2)
                pcm_buffer = pcm_buffer[overlap_size:]

    finally:
        client_sock.close()
        server_sock.close()

if __name__ == "__main__":
    main()