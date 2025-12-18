import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_waveform(y, sr=4000, title="Waveform"):
    """
    绘制音频波形
    """
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_compare_waveforms(y1, y2, sr=4000, labels=("Before", "After")):
    """
    对比：滤波前 vs 滤波后
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y1, sr=sr)
    plt.title(labels[0])

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y2, sr=sr)
    plt.title(labels[1])

    plt.tight_layout()
    plt.show()


def plot_segments(segments, sr=4000):
    """
    展示切片后的多段波形
    """
    n = len(segments)
    plt.figure(figsize=(12, 3*n))
    
    for i, seg in enumerate(segments):
        plt.subplot(n, 1, i+1)
        librosa.display.waveshow(seg, sr=sr)
        plt.title(f"Segment {i+1}")
    
    plt.tight_layout()
    plt.show()


def plot_mel(mel, sr=4000, title="Mel Spectrogram"):
    """
    绘制 Mel 频谱（2D CNN 输入）
    mel shape: (n_mels, time_frames)
    """
    plt.figure(figsize=(5, 4))
    librosa.display.specshow(
        mel,
        sr=sr,
        hop_length=256,
        x_axis='time',
        y_axis='mel',
        cmap='magma'
    )
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# NEW: Bicoherence / Bispectrum
# =========================
def plot_bicoherence(
    bic,
    title="2D Bicoherence",
    cmap="jet",
):
    """
    绘制 2D bicoherence / bispectrum 图

    bic: np.ndarray, shape (H, W)
         一般是 64x64 或 128x128
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(
        bic,
        origin="lower",
        aspect="auto",
        cmap=cmap
    )
    plt.colorbar(label="Bicoherence")
    plt.title(title)
    plt.xlabel("f2 index")
    plt.ylabel("f1 index")
    plt.tight_layout()
    plt.show()