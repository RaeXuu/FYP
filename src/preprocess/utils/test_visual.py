import os, sys
import yaml
import pandas as pd

# 自动定位项目根目录
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size
from src.preprocess.bicoherence_2d import bicoherence_2d
from src.preprocess.utils.visualize import *

# ---- 📥 加载 Config ----
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_cfg = config["data"]
mel_cfg = config["mel"]

# ---- 📂 加载数据 ----
df = pd.read_csv("/home/agiuser/FypProj/data/metadata_physionet.csv")   
# 取一个样本做可视化测试
path = df.iloc[0]["filepath"]

# ---- 🛠 预处理流水线 ----

# Step 1: Load (自动使用 yaml 里的 sample_rate)
y, sr = load_wav(path, target_sr=data_cfg["sample_rate"])

# Step 2: Bandpass (使用 yaml 里的 low 和 high)
y_filt = apply_bandpass(
    y, 
    fs=sr, 
    lowcut=data_cfg["bandpass"]["low"], 
    highcut=data_cfg["bandpass"]["high"]
)

# Step 3: Segment (这里建议确保你的 segment_audio 函数也支持从 yaml 传参)
# 如果该函数内部没读 yaml，可以手动传：segment_length=data_cfg["segment_length"]
segments = segment_audio(y_filt, sr=sr)

# Step 4: Mel (传入完整的 mel_cfg)
# 注意：如果你的 logmel_fixed_size 还需要 target_shape，
# 而 YAML 里没写，我们先手动指定一个，或者你可以加到 YAML 的 data 栏目下
target_shape = (mel_cfg["n_mels"], 64)

mel = logmel_fixed_size(
    y=segments[0], 
    sr=sr, 
    mel_cfg=mel_cfg, 
    target_shape=target_shape
)

# Step 5: Bispectrum (双谱分析)
bic = bicoherence_2d(segments[0], fs=sr, out_size=64)

# ---- 📊 可视化 ----
print(f"正在展示文件: {os.path.basename(path)}")
plot_waveform(y, sr, "Raw WAV")
plot_compare_waveforms(y, y_filt, sr)
plot_segments(segments, sr)
plot_mel(mel, sr)
plot_bicoherence(bic, title="Bicoherence")
