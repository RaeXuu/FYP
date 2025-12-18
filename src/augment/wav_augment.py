# src/augment/waveform_augment.py
# Waveform-level augmentation for heart sound signals

import numpy as np


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


def waveform_augment(
    x,
    prob_gain=0.5,
    prob_noise=0.5,
    prob_shift=0.5,
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

    return x
