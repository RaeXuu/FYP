import pandas as pd
from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size
from src.preprocess.utils.visualize import *

df = pd.read_csv("/mnt/d/FypProj/data/metadata1.csv")
path = df.iloc[0]["filepath"]

# Step1 load
y, sr = load_wav(path)

# Step2 bandpass
y_filt = apply_bandpass(y, fs=sr)

# Step3 segment
segments = segment_audio(y_filt, sr)

# Step4 mel
mel = logmel_fixed_size(segments[0], sr)

# ---- Visualize ----
plot_waveform(y, sr, "Raw WAV")
plot_compare_waveforms(y, y_filt, sr)
plot_segments(segments, sr)
plot_mel(mel, sr)

