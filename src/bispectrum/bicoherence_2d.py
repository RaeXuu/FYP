# src/bispectrum/bicoherence_2d.py
# Generate 2D bicoherence map for CNN input
# Designed for heart sound signals (paper-style reproduction)

import numpy as np
from scipy.signal import get_window
from skimage.transform import resize
import matplotlib.pyplot as plt

def bicoherence_2d(
    x: np.ndarray,
    fs: int,
    nfft: int = 256,
    seglen: int = 512,
    overlap: float = 0.5,
    fmax_hz: float = 800.0,
    out_size: int = 128,
):
    """
    Compute 2D bicoherence map (upper triangular) for CNN input.

    Returns:
        bic_img: shape (out_size, out_size), float32
    """

    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size < seglen:
        x = np.pad(x, (0, seglen - x.size))

    hop = int(seglen * (1 - overlap))
    window = get_window("hann", seglen, fftbins=True)

    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    fmask = freqs <= fmax_hz
    idx = np.where(fmask)[0]
    F = len(idx)

    B = np.zeros((F, F), dtype=np.complex64)
    P12 = np.zeros((F, F), dtype=np.float32)
    P3 = np.zeros((F, F), dtype=np.float32)
    count = 0

    for start in range(0, len(x) - seglen + 1, hop):
        seg = x[start:start + seglen] * window
        X = np.fft.rfft(seg, nfft)[idx]

        X1 = X[:, None]
        X2 = X[None, :]

        k = np.arange(F)
        k3 = k[:, None] + k[None, :]
        valid = k3 < F

        X3 = np.zeros((F, F), dtype=np.complex64)
        X3[valid] = X[k3[valid]]

        Bij = X1 * X2 * np.conj(X3)
        B += Bij

        P12 += np.abs(X1 * X2) ** 2
        P3 += np.abs(X3) ** 2

        count += 1

    B /= count
    denom = np.sqrt(P12 * P3) + 1e-12
    bic = np.abs(B) / denom

    # only upper-triangular & valid region
    tri = np.triu(np.ones_like(bic, dtype=bool))
    valid = (np.arange(F)[:, None] + np.arange(F)[None, :]) < F
    mask = tri & valid

    bic[~mask] = 0.0

    # log + normalize (论文常见隐含步骤)
    bic = np.log1p(bic)
    bic = (bic - bic.min()) / (bic.max() - bic.min() + 1e-9)

    # resize for CNN
    bic_img = resize(
        bic,
        (out_size, out_size),
        mode="reflect",
        anti_aliasing=True,
    ).astype(np.float32)

    return bic_img


if __name__ == "__main__":
    # quick sanity check
    fs = 4000
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 80 * t) + 0.4 * np.sin(2 * np.pi * 160 * t)
    x += 0.05 * np.random.randn(len(x))

    img = bicoherence_2d(x, fs)
    print("bicoherence map:", img.shape, img.min(), img.max())
    plt.imshow(img, cmap="jet", origin="lower")
    plt.colorbar()
    plt.title("2D Bicoherence")
    plt.show()