"""Spectrum and MFCC features for experiment 2."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .config import DEFAULT_CONFIG
from .preprocess import apply_window, frame_signal, pre_emphasis


def magnitude_spectrum(frames: np.ndarray, n_fft: int = 512) -> np.ndarray:
    return np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float32)


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_filters: int) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(sample_rate / 2), n_filters + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_filters):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for value in range(left, min(center, bank.shape[1])):
            bank[index, value] = (value - left) / (center - left)
        for value in range(center, min(right, bank.shape[1])):
            bank[index, value] = (right - value) / (right - center)
    return bank


def extract_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_filters: int = 26,
    n_fft: int = 512,
) -> np.ndarray:
    emphasized = pre_emphasis(samples, DEFAULT_CONFIG.pre_emphasis)
    frames = apply_window(
        frame_signal(
            emphasized,
            sample_rate,
            DEFAULT_CONFIG.frame_ms,
            DEFAULT_CONFIG.hop_ms,
        ),
        DEFAULT_CONFIG.window,
    )
    power = magnitude_spectrum(frames, n_fft=n_fft) ** 2 / n_fft
    energies = np.maximum(power @ _mel_filterbank(sample_rate, n_fft, n_filters).T, 1e-10)
    coefficients = dct(np.log(energies), type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coefficients.astype(np.float32)
