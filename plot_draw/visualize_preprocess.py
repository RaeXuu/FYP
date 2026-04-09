import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt

# ---- 参数（与论文一致） ----
SR = 2000
LOWCUT, HIGHCUT = 25, 400
SEG_LEN = SR * 2          # 2s window
N_MELS, N_FFT, HOP = 64, 256, 128

# ---- 载入一条 Normal 录音（替换路径）----
raw, orig_sr = librosa.load('/home/agiuser/FypProj/data/raw/DataSet2/training-a/a0007.wav', sr=SR, mono=True)

# Step 1: raw waveform（截前 4s 方便展示）
display_len = SR * 4
raw_display = raw[:display_len]
t = np.arange(len(raw_display)) / SR

# Step 2: 带通滤波
b, a = butter(4, [LOWCUT / (SR/2), HIGHCUT / (SR/2)], btype='band')
filtered = filtfilt(b, a, raw)
filtered_display = filtered[:display_len]

# Step 3: 取第一个 2s 分段
segment = filtered[:SEG_LEN]
t_seg = np.arange(SEG_LEN) / SR

# Step 4: Log-Mel 频谱
mel = librosa.feature.melspectrogram(y=segment, sr=SR,
                                      n_fft=N_FFT, hop_length=HOP,
                                      n_mels=N_MELS, fmin=LOWCUT, fmax=HIGHCUT)
log_mel = librosa.power_to_db(mel, ref=np.max)

# ---- 画四格图 ----
fig, axes = plt.subplots(4, 1, figsize=(10, 10))
plt.rcParams['font.family'] = 'serif'

axes[0].plot(t, raw_display, linewidth=0.6)
axes[0].set_title('(a) Raw waveform')
axes[0].set_ylabel('Amplitude'); axes[0].set_xlabel('Time (s)')

axes[1].plot(t, filtered_display, linewidth=0.6, color='steelblue')
axes[1].set_title('(b) After bandpass filter (25–400 Hz)')
axes[1].set_ylabel('Amplitude'); axes[1].set_xlabel('Time (s)')

axes[2].plot(t_seg, segment, linewidth=0.6, color='darkorange')
axes[2].set_title('(c) Segmented window (2 s)')
axes[2].set_ylabel('Amplitude'); axes[2].set_xlabel('Time (s)')

img = axes[3].imshow(log_mel, aspect='auto', origin='lower',
                     extent=[0, segment.shape[0]/SR, 0, N_MELS],
                     cmap='magma')
axes[3].set_title('(d) Log-Mel spectrogram (32×64)')
axes[3].set_ylabel('Mel bin'); axes[3].set_xlabel('Time (s)')
fig.colorbar(img, ax=axes[3], label='dB')

plt.tight_layout()
plt.savefig('fig3_2_preprocessing_steps.png', dpi=300, bbox_inches='tight')
print("fig3_2_preprocessing_steps.png 已保存")