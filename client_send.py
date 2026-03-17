import bluetooth  # 确保已安装 pybluez2 
import os
import time

# ==========================================
# 配置参数
# ==========================================
# 1. 填入你树莓派的蓝牙 MAC 地址
TARGET_MAC = "AA:BB:CC:DD:EE:FF" 

# 2. 选一个你电脑上存在的 wav 文件进行测试
FILE_TO_SEND = "data/raw/DataSet1/set_a/Aunlabelledtest__201101051105.wav"

def main():
    if not os.path.exists(FILE_TO_SEND):
        print(f"❌ 错误：找不到文件 {FILE_TO_SEND}")
        return

    print(f"📡 正在尝试连接树莓派 ({TARGET_MAC})...")
    
    try:
        # 创建 RFCOMM 蓝牙 Socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        # 默认通道通常为 1
        sock.connect((TARGET_MAC, 1))
        print("✅ 连接成功！")

        # 以二进制只读模式打开文件
        with open(FILE_TO_SEND, "rb") as f:
            data = f.read()
            file_size = len(data)
            
            print(f"🚀 开始发送音频文件 ({file_size / 1024:.2f} KB)...")
            
            # 为了防止树莓派接收缓冲区溢出，我们分块发送
            chunk_size = 1024
            sent_bytes = 0
            
            for i in range(0, file_size, chunk_size):
                chunk = data[i : i + chunk_size]
                sock.send(chunk)
                sent_bytes += len(chunk)
                # 打印发送进度
                print(f"\r📤 已发送: {sent_bytes/1024:>7.2f} KB / {file_size/1024:.2f} KB", end="")
                # 稍微停顿一下，确保传输稳定性
                time.sleep(0.01)

        print("\n\n🎉 发送完毕！正在关闭连接...")
        sock.close()
        print("🏁 任务完成。请查看树莓派端的推理结果。")

    except Exception as e:
        print(f"\n❌ 连接或发送失败: {e}")
        print("💡 请检查：1. 树莓派是否已运行 main_pi.py  2. 蓝牙是否已配对")

if __name__ == "__main__":
    main()