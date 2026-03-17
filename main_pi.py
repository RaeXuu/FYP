import os
import sys
import numpy as np
import tflite_runtime.interpreter as tflite
import yaml
import bluetooth
import time

# ==========================================
# 1. 环境初始化
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.preprocess_pipeline import preprocess_wav_for_pi

def softmax(x):
    """Logits 转换为概率"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# 2. 模型加载
# ==========================================
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

q_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite"))
d_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_model_quant.tflite"))
q_interp.allocate_tensors()
d_interp.allocate_tensors()

# ==========================================
# 3. 蓝牙监控接收逻辑
# ==========================================
def main():
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    
    print("📡 正在等待手机连接...")
    client_sock, address = server_sock.accept()
    print(f"✅ 已连接手机: {address}")

    received_file = os.path.join(PROJECT_ROOT, "received_test.wav")
    total_bytes = 0
    start_time = time.time()

    with open(received_file, "wb") as f:
        print("\n📥 正在接收数据流 (实时监控中)...")
        print("-" * 50)
        try:
            while True:
                # 设定超时，如果 3 秒没收到新数据就认为文件发完了
                client_sock.settimeout(3.0) 
                data = client_sock.recv(4096)
                if not data: break
                
                # 写入文件并统计
                f.write(data)
                total_bytes += len(data)
                
                # --- 实时显示部分 ---
                # 1. 检查前几个字节，判断格式
                sample_hex = data[:8].hex().upper()
                try:
                    sample_text = data[:8].decode('utf-8')
                except:
                    sample_text = "[Binary]"

                # 2. 打印进度条
                elapsed = time.time() - start_time
                speed = (total_bytes / 1024) / elapsed if elapsed > 0 else 0
                print(f"\r📦 已接收: {total_bytes/1024:>7.2f} KB | 速度: {speed:>5.1f} KB/s | 采样: {sample_hex} ({sample_text})", end="")

        except bluetooth.btcommon.BluetoothError:
            print("\n\n⏱️ 接收超时或手机断开，开始处理文件...")
        except Exception as e:
            print(f"\n\n⚠️ 出错: {e}")

    client_sock.close()
    server_sock.close()
    
    # 验证文件是否有效 (WAV 头部至少 44 字节)
    if total_bytes < 44:
        print("\n❌ 错误：接收到的数据太少，不像是有效的音频文件。")
        return

    print(f"\n\n💾 接收完毕。文件大小: {total_bytes} 字节")
    print("-" * 50)

    # 4. 执行推理
    try:
        tensors = preprocess_wav_for_pi(received_file, config)
        print(f"🧩 预处理成功：切分出 {len(tensors)} 个片段")
        
        # 择优推理逻辑
        best_score = -1
        best_tensor = None

        for tensor in tensors:
            q_interp.set_tensor(q_interp.get_input_details()[0]['index'], tensor)
            q_interp.invoke()
            q_probs = softmax(q_interp.get_tensor(q_interp.get_output_details()[0]['index'])[0])
            
            if q_probs[1] > best_score: # 1 为 Good Quality
                best_score = q_probs[1]
                best_tensor = tensor

        if best_tensor is not None and best_score > 0.8:
            d_interp.set_tensor(d_interp.get_input_details()[0]['index'], best_tensor)
            d_interp.invoke()
            d_probs = softmax(d_interp.get_tensor(d_interp.get_output_details()[0]['index'])[0])
            
            label = "Normal" if np.argmax(d_probs) == 0 else "Abnormal"
            print(f"✨ 黄金片段结果: {label} | 置信度: {np.max(d_probs):.2%}")
        else:
            print("⚠️ 未发现高质量心音片段。")

    except Exception as e:
        print(f"❌ 推理失败: {e}")

if __name__ == "__main__":
    main()