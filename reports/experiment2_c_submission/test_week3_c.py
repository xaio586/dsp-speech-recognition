# -*- coding: utf-8 -*-
# 成员C单元测试：
# 重点验证自行实现的 DFT/FFT 与 numpy FFT 一致，
# 并验证 STFT 输出形状、补零逻辑和基本物理性质。

import numpy as np
import pytest

import week3_c as w3


def test_fft_matches_dft_random():
    rng = np.random.default_rng(0)

    for N in [1, 2, 4, 8, 16, 64, 128, 512]:
        x = rng.normal(size=N) + 1j * rng.normal(size=N)

        X_fft = w3.fft_radix2(x)
        X_dft = w3.dft_naive(x)

        np.testing.assert_allclose(X_fft, X_dft, atol=1e-9, rtol=1e-9)


def test_fft_matches_numpy_fft():
    rng = np.random.default_rng(1)

    for N in [2, 4, 8, 16, 64, 128, 512]:
        x = rng.normal(size=N) + 1j * rng.normal(size=N)

        X_fft = w3.fft_radix2(x)
        X_np = np.fft.fft(x)

        np.testing.assert_allclose(X_fft, X_np, atol=1e-8, rtol=1e-8)


def test_sine_peak_frequency():
    fs = 16000
    N = 512
    f0 = 1000.0

    n = np.arange(N)
    x = np.sin(2.0 * np.pi * f0 * n / fs)

    X = w3.fft_radix2(x)
    k = int(np.argmax(np.abs(X[: N // 2 + 1])))
    f_est = k * fs / N

    assert abs(f_est - f0) <= fs / N + 1e-9


def test_parseval_energy():
    rng = np.random.default_rng(2)
    N = 512
    x = rng.normal(size=N)

    X = w3.fft_radix2(x)

    energy_time = np.sum(np.abs(x) ** 2)
    energy_freq = np.sum(np.abs(X) ** 2) / N

    assert np.isclose(energy_time, energy_freq, rtol=1e-10, atol=1e-10)


def test_stft_output_shape():
    rng = np.random.default_rng(3)
    fs = 16000
    signal = rng.normal(size=16000)

    frame_len = 400
    hop_len = 160
    n_fft = 512

    X_complex, power, freqs, times, frames_w = w3.stft_from_signal(
        signal=signal,
        fs=fs,
        frame_len=frame_len,
        hop_len=hop_len,
        n_fft=n_fft,
        window="hamming",
    )

    n_frames = 1 + int(np.ceil((signal.size - frame_len) / hop_len))
    n_one_side = n_fft // 2 + 1

    assert X_complex.shape == (n_frames, n_one_side)
    assert power.shape == X_complex.shape
    assert freqs.shape == (n_one_side,)
    assert times.shape == (n_frames,)
    assert frames_w.shape == (n_frames, frame_len)


def test_short_frame_padding():
    signal = np.ones(100, dtype=np.float64)

    X_complex, power, freqs, times, frames_w = w3.stft_from_signal(
        signal=signal,
        fs=16000,
        frame_len=400,
        hop_len=160,
        n_fft=512,
        window="hamming",
    )

    assert X_complex.shape == (1, 257)
    assert power.shape == (1, 257)
    assert frames_w.shape == (1, 400)


def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        w3.fft_radix2(np.ones(3))


def test_stft_rejects_n_fft_less_than_frame_len():
    with pytest.raises(ValueError):
        w3.stft_from_signal(
            np.ones(1000),
            fs=16000,
            frame_len=400,
            hop_len=160,
            n_fft=256,
        )


def test_validate_fft_thresholds():
    metrics = w3.validate_fft(fs=16000, n_fft=512)

    assert metrics["dft_vs_fft_N512_max_abs_err"] < 1e-8
    assert metrics["fft_vs_numpy_N512_max_abs_err"] < 1e-8
    assert metrics["sine_peak_abs_err_hz"] <= 16000 / 512 + 1e-9
    assert metrics["parseval_rel_err"] < 1e-10
