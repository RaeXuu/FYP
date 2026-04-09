import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt

plt.rcParams['font.family'] = 'serif'

SR = 2000
LOWCUT, HIGHCUT = 25, 400
SEG_LEN = SR * 2
N_MELS, N_FFT, HOP = 64, 256, 128

def get_log_mel(path):
    raw, _ = librosa.load(path, sr=SR, mono=True)
    b, a = butter(4, [LOWCUT / (SR/2), HIGHCUT / (SR/2)], btype='band')
    filtered = filtfilt(b, a, raw)
    seg = filtered[:SEG_LEN]
    mel = librosa.feature.melspectrogram(y=seg, sr=SR,
                                          n_fft=N_FFT, hop_length=HOP,
                                          n_mels=N_MELS, fmin=LOWCUT, fmax=HIGHCUT)
    return librosa.power_to_db(mel, ref=np.max)

normal_mel   = get_log_mel('/home/agiuser/FypProj/data/raw/DataSet2/training-a/a0007.wav')
abnormal_mel = get_log_mel('/home/agiuser/FypProj/data/raw/DataSet2/training-a/a0002.wav')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, mel, title in zip(axes,
                           [normal_mel, abnormal_mel],
                           ['(a) Normal', '(b) Abnormal']):
    img = ax.imshow(mel, aspect='auto', origin='lower',
                    extent=[0, SEG_LEN/SR, 0, N_MELS],
                    cmap='magma', vmin=-80, vmax=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel bin', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    fig.colorbar(img, ax=ax, label='dB')

plt.tight_layout()
plt.savefig('fig3_3_mel_comparison.png', dpi=300, bbox_inches='tight')
print("fig3_3_mel_comparison.png 已保存")
