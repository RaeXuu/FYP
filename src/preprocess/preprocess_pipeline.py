import os
import sys
import numpy as np

# 确保能找到项目模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size

def preprocess_wav_for_pi(wav_path, config):
    """
    完全复刻训练时的预处理流程，用于真机推理。
    返回: list of numpy arrays, 每个形状为 (1, 1, 32, 64)
    """
    sr = config["data"]["sample_rate"]
    mel_cfg = config["mel"]
    
    # 1. 加载音频
    y, _ = load_wav(wav_path, target_sr=sr)
    
    # 2. 带通滤波 (25Hz - 400Hz)
    # 这是去除心音中高频环境噪音的关键步骤
    y_filtered = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)
    
    # 3. 切片处理
    # 将接收到的音频切成若干个 2 秒的片段
    segments = segment_audio(y_filtered, sr=sr)
    
    processed_segments = []
    
    for seg in segments:
        # 4. 转换为 Log-Mel 频谱
        # target_shape 必须与训练时完全一致 (32, 64)
        mel = logmel_fixed_size(
            y=seg,
            sr=sr,
            mel_cfg=mel_cfg,
            target_shape=(mel_cfg["n_mels"], 64), 
        )
        
        # 5. 调整维度适配 TFLite 输入
        # (32, 64) -> (1, 1, 32, 64)  即 [Batch, Channel, Height, Width]
        mel_tensor = mel[np.newaxis, np.newaxis, ...]
        processed_segments.append(mel_tensor.astype(np.float32))
        
    return processed_segments

if __name__ == "__main__":
    # 快速自测逻辑
    import yaml
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        conf = yaml.safe_load(f)
        
    test_wav = "data/raw/test.wav" # 找一个本地 wav 测试
    if os.path.exists(test_wav):
        tensors = preprocess_wav_for_pi(test_wav, conf)
        print(f"✅ 预处理完成，生成了 {len(tensors)} 个切片")
        print(f"✅ 单个切片形状: {tensors[0].shape}")