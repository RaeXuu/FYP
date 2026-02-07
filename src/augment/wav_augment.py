# src/augment/waveform_augment.py
# Waveform-level augmentation for heart sound signals

import numpy as np
import librosa


def random_gain(x, gain_range=(0.8, 1.2)):
    """
    随机幅值缩放（模拟按压力/个体差异）
    """
    gain = np.random.uniform(gain_range[0], gain_range[1])
    return x * gain


def add_gaussian_noise(x, snr_db_range=(20, 35)):
    """
    加高斯噪声（轻量）
    snr_db 越大，噪声越小
    """
    signal_power = np.mean(x ** 2)
    if signal_power <= 1e-12:
        return x

    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    snr = 10 ** (snr_db / 10)

    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def random_time_shift(x, max_shift_ratio=0.1):
    """
    随机时间平移（循环移位）
    max_shift_ratio: 最大平移比例（相对于长度）
    """
    n = len(x)
    max_shift = int(n * max_shift_ratio)
    if max_shift <= 1:
        return x

    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift)

def random_resampling(x, stretch_range=(0.9, 1.1)):
    """
    随机重采样：模拟心率（BPM）的变化。
    这是对抗过拟合最有效的手段之一。
    """
    stretch = np.random.uniform(stretch_range[0], stretch_range[1])
    n_orig = len(x)
    # 使用线性插值进行简易拉伸，然后恢复到固定长度
    x_stretched = np.interp(
        np.linspace(0, n_orig, int(n_orig * stretch)),
        np.arange(n_orig),
        x
    )
    # 保持长度一致（截断或补零）
    if len(x_stretched) > n_orig:
        return x_stretched[:n_orig]
    else:
        return np.pad(x_stretched, (0, n_orig - len(x_stretched)), 'constant')


def waveform_augment(
    x,
    prob_gain=0.5,
    prob_noise=0.5,
    prob_shift=0.5,
    prob_resample=0.3, # 新增：模拟心率波动
    prob_flip=0.5      # 新增：极性反转
):
    """
    组合增广（按概率触发）
    """
    if np.random.rand() < prob_gain:
        x = random_gain(x)

    if np.random.rand() < prob_noise:
        x = add_gaussian_noise(x)

    if np.random.rand() < prob_shift:
        x = random_time_shift(x)

    if np.random.rand() < prob_resample:
        x = random_resampling(x)
        
    if np.random.rand() < prob_flip:
        x = x * -1.0

    return x
