import os, sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import pandas as pd
from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size
from preprocess.bicoherence_2d import bicoherence_2d
from src.preprocess.utils.visualize import *

df = pd.read_csv("/mnt/d/FypProj/data/metadata1.csv")
path = df.iloc[68]["filepath"]
# fill in need to substract 2 in advance
# 2-41 artifact
# 42-60 extrahls
# 61-94 murmur
# 95-125 normal



# Step1 load
y, sr = load_wav(path)

# Step2 bandpass
y_filt = apply_bandpass(y, fs=sr)

# Step3 segment
segments = segment_audio(y_filt, sr)

# Step4 mel
mel = logmel_fixed_size(segments[0], sr)

# Step5 bispectrum
bic = bicoherence_2d(segments[0], fs=sr, out_size=64)

# ---- Visualize ----
plot_waveform(y, sr, "Raw WAV")
plot_compare_waveforms(y, y_filt, sr)
plot_segments(segments, sr)
plot_mel(mel, sr)
plot_bicoherence(bic, title="Bicoherence")
