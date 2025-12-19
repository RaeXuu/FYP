# src/preprocess/wavelet.py
# Continuous Wavelet Transform (CWT) feature extraction for heart sounds

import numpy as np
import pywt
import torch
import torch.nn.functional as F



def cwt_fixed_size(
    x,
    sr=4000,
    wavelet="morl",
    num_scales=64,
    out_size=64,
):
    """
    Continuous Wavelet Transform -> fixed size 2D feature

    Parameters
    ----------
    x : np.ndarray
        1D waveform segment
    sr : int
        Sampling rate
    wavelet : str
        Mother wavelet (default: morl)
    num_scales : int
        Number of wavelet scales
    out_size : int
        Output size (out_size x out_size)

    Returns
    -------
    feat : np.ndarray
        2D wavelet scalogram (out_size, out_size)
    """

    # 1️⃣ 定义 scales（由高到低频）
    scales = np.linspace(1, num_scales, num_scales)

    # 2️⃣ CWT
    coeffs, freqs = pywt.cwt(
        x,
        scales=scales,
        wavelet=wavelet,
        sampling_period=1.0 / sr,
    )
    # coeffs shape: (num_scales, time)

    # 3️⃣ 幅值（能量）
    scalogram = np.abs(coeffs)

    # 4️⃣ log 压缩（和 mel 类似，稳定训练）
    scalogram = np.log1p(scalogram)

    # 5️⃣ resize 到固定尺寸（CNN 友好）
    scalogram_t = torch.tensor(scalogram).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    scalogram_resized = F.interpolate(
        scalogram_t,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    # 6️⃣ 归一化到 [0, 1]
    min_val = scalogram_resized.min()
    max_val = scalogram_resized.max()
    if max_val > min_val:
        scalogram_resized = (scalogram_resized - min_val) / (max_val - min_val)

    return scalogram_resized
