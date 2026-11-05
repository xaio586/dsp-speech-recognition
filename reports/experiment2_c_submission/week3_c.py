# -*- coding: utf-8 -*-
"""
数字信号处理实验 第3周 成员C独立代码
负责：
1. 自行实现 DFT / 基2 FFT
2. 实现 STFT 频谱计算
3. 绘制波形、单帧频谱、语谱图、DFT-FFT误差图
4. 输出 FFT 正确性对照指标

说明：
frame_signal / get_window 仅为 C 独立自测 STFT 的辅助函数，
不代替 B 的预加重、分帧、加窗模块。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile


# ============================================================
# 一、DFT / FFT 核心实现
# ============================================================

def dft_naive(x: np.ndarray) -> np.ndarray:
    """
    直接按 DFT 公式实现，仅用于小规模正确性对照。
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = x.size
    if N == 0:
        raise ValueError("DFT input length must be > 0")

    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    return (W @ x).astype(np.complex128)


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """
    自行实现基2 DIT FFT。
    输入长度必须是 2 的整数次幂。
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = x.size

    if N == 0:
        raise ValueError("FFT input length must be > 0")
    if (N & (N - 1)) != 0:
        raise ValueError("FFT input length must be a power of two")

    y = x.copy()

    # 1) 比特反转重排
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            y[i], y[j] = y[j], y[i]

    # 2) 蝶形运算
    length = 2
    while length <= N:
        half = length // 2
        wlen = np.exp(-2j * np.pi / length)

        for start in range(0, N, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = y[start + k]
                v = y[start + k + half] * w
                y[start + k] = u + v
                y[start + k + half] = u - v
                w *= wlen

        length <<= 1

    return y


# ============================================================
# 二、STFT 辅助函数与频谱计算
# ============================================================

def frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    辅助分帧函数，仅用于 C 独立运行 STFT。
    返回 shape = (n_frames, frame_len)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()

    if x.size == 0:
        raise ValueError("signal is empty")
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")

    if x.size <= frame_len:
        n_frames = 1
    else:
        n_frames = 1 + int(np.ceil((x.size - frame_len) / hop_len))
    required_length = (n_frames - 1) * hop_len + frame_len
    if required_length > x.size:
        x = np.pad(x, (0, required_length - x.size), mode="constant")
    frames = np.empty((n_frames, frame_len), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_len
        frames[i, :] = x[start:start + frame_len]

    return frames


def get_window(name: str, N: int) -> np.ndarray:
    """
    辅助窗函数，仅用于 C 独立运行 STFT。
    """
    name = name.lower()
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(N, dtype=np.float64)
    if name == "hamming":
        return np.hamming(N)
    if name in ("hanning", "hann"):
        return np.hanning(N)
    raise ValueError(f"Unsupported window: {name}")


def stft_from_signal(
    signal: np.ndarray,
    fs: int = 16000,
    frame_len: int = 400,
    hop_len: int = 160,
    n_fft: int = 512,
    window: str = "hamming",
    pre_emphasis_coef: float = 0.97,
):
    """
    对整段语音做 STFT。

    返回：
        X_complex: 单边复数谱，shape=(n_frames, n_fft//2+1)
        power:     单边功率谱，shape=(n_frames, n_fft//2+1)
        freqs:     频率轴，shape=(n_fft//2+1,)
        times:     每帧中心时间，shape=(n_frames,)
        frames_w:  加窗后未补零的帧，shape=(n_frames, frame_len)
    """
    if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:
        raise ValueError("n_fft must be a power of two")
    if n_fft < frame_len:
        raise ValueError("n_fft must be >= frame_len")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0.0 <= pre_emphasis_coef < 1.0:
        raise ValueError("pre_emphasis_coef must be in [0, 1)")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    emphasized = signal.copy()
    if emphasized.size > 1:
        emphasized[1:] = signal[1:] - pre_emphasis_coef * signal[:-1]

    frames = frame_signal(emphasized, frame_len, hop_len)
    win = get_window(window, frame_len)
    frames_w = frames * win[None, :]

    if n_fft > frame_len:
        frames_padded = np.pad(
            frames_w,
            ((0, 0), (0, n_fft - frame_len)),
            mode="constant",
        )
    else:
        frames_padded = frames_w

    X_full = np.empty((frames_padded.shape[0], n_fft), dtype=np.complex128)
    for i in range(frames_padded.shape[0]):
        X_full[i, :] = fft_radix2(frames_padded[i, :])

    n_one_side = n_fft // 2 + 1
    X_complex = X_full[:, :n_one_side]
    power = (np.abs(X_complex) ** 2 / n_fft).astype(np.float64)

    # 自行生成频率轴，不调用 np.fft.rfftfreq
    freqs = np.arange(n_one_side, dtype=np.float64) * fs / n_fft

    times = (
        np.arange(frames.shape[0], dtype=np.float64) * hop_len
        + frame_len / 2.0
    ) / fs

    return X_complex, power, freqs, times, frames_w


# ============================================================
# 三、读取语音与绘图
# ============================================================

def load_wav(path: str | Path):
    """
    读取 wav，多声道取平均，并归一化幅值。
    """
    fs, data = wavfile.read(str(path))
    data = data.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak > 0:
        data = data / peak

    return int(fs), data


def plot_results(
    signal: np.ndarray,
    fs: int,
    X_complex: np.ndarray,
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    frames_w: np.ndarray,
    outdir: str | Path,
    n_fft: int,
    frame_len: int,
    hop_len: int,
):
    """
    输出：
        week3_waveform.png
        week3_frame_spectrum.png
        week3_spectrogram.png
        week3_dft_vs_fft_error.png
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 原始语音波形
    t = np.arange(signal.size, dtype=np.float64) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, signal, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(outdir / "week3_waveform.png", dpi=150)
    plt.close()

    # 2. 选择能量最大的一帧画单帧频谱
    frame_idx = int(np.argmax(np.sum(power, axis=1)))
    mag_db = 20.0 * np.log10(np.abs(X_complex[frame_idx, :]) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Single-frame spectrum (frame {frame_idx})")
    plt.xlim(0, fs / 2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "week3_frame_spectrum.png", dpi=150)
    plt.close()

    # 3. 语谱图
    S_db = 10.0 * np.log10(power.T + 1e-12)  # power uses 10*log10

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        f"Pre-emphasized power spectrogram (frame_len={frame_len}, hop={hop_len}, n_fft={n_fft})"
    )
    plt.ylim(0, min(fs / 2, 8000))
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(outdir / "week3_spectrogram.png", dpi=150)
    plt.close()

    # 4. DFT 与 FFT 误差图：对同一帧分别调用 dft_naive 和 fft_radix2
    x = frames_w[frame_idx, :]

    if x.size < n_fft:
        x_pad = np.pad(x, (0, n_fft - x.size), mode="constant")
    else:
        x_pad = x[:n_fft]

    X_fft = fft_radix2(x_pad)
    X_dft = dft_naive(x_pad)
    err = np.abs(X_fft - X_dft)

    plt.figure(figsize=(10, 3))
    plt.plot(err)
    plt.xlabel("FFT bin")
    plt.ylabel("Absolute error")
    plt.title("DFT vs self-implemented FFT absolute error")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "week3_dft_vs_fft_error.png", dpi=150)
    plt.close()

    return frame_idx, float(np.max(err))


# ============================================================
# 四、正确性对照指标
# ============================================================

def validate_fft(fs: int = 16000, n_fft: int = 512):
    """
    返回 JSON 可序列化的对照指标。
    """
    rng = np.random.default_rng(2024)
    out = {}

    # 1) DFT vs 自实现 FFT vs numpy FFT
    for N in (64, 128, 512):
        x = rng.normal(size=N) + 1j * rng.normal(size=N)

        X_fft = fft_radix2(x)
        X_dft = dft_naive(x)
        X_np = np.fft.fft(x)

        out[f"dft_vs_fft_N{N}_max_abs_err"] = float(
            np.max(np.abs(X_fft - X_dft))
        )
        out[f"fft_vs_numpy_N{N}_max_abs_err"] = float(
            np.max(np.abs(X_fft - X_np))
        )

    # 2) 正弦峰值频率对照
    f0 = 1000.0
    n = np.arange(n_fft)
    x_sine = np.sin(2.0 * np.pi * f0 * n / fs)

    X_sine = fft_radix2(x_sine)
    k_peak = int(np.argmax(np.abs(X_sine[: n_fft // 2 + 1])))
    f_est = k_peak * fs / n_fft

    out["sine_f0_hz"] = float(f0)
    out["sine_f_est_hz"] = float(f_est)
    out["sine_freq_bin_hz"] = float(fs / n_fft)
    out["sine_peak_abs_err_hz"] = float(abs(f_est - f0))

    # 3) Parseval 能量对照
    x_rand = rng.normal(size=n_fft)
    X_rand = fft_radix2(x_rand)

    energy_time = float(np.sum(np.abs(x_rand) ** 2))
    energy_freq = float(np.sum(np.abs(X_rand) ** 2) / n_fft)

    out["parseval_time_energy"] = energy_time
    out["parseval_freq_energy"] = energy_freq
    out["parseval_rel_err"] = float(
        abs(energy_time - energy_freq) / max(energy_time, 1e-12)
    )

    return out


# ============================================================
# 五、命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="输入 wav 文件路径")
    parser.add_argument("--out", default="outputs/week3_C", help="输出目录")
    parser.add_argument("--fs", type=int, default=None, help="覆盖采样率")
    parser.add_argument("--frame-len", type=int, default=400)
    parser.add_argument("--hop-len", type=int, default=160)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument(
        "--window",
        default="hamming",
        choices=["rectangular", "hamming", "hanning", "hann"],
    )
    args = parser.parse_args()

    fs, signal = load_wav(args.wav)
    if args.fs is not None:
        fs = args.fs

    X_complex, power, freqs, times, frames_w = stft_from_signal(
        signal=signal,
        fs=fs,
        frame_len=args.frame_len,
        hop_len=args.hop_len,
        n_fft=args.n_fft,
        window=args.window,
    )

    outdir = Path(args.out)
    frame_idx, max_err = plot_results(
        signal=signal,
        fs=fs,
        X_complex=X_complex,
        power=power,
        freqs=freqs,
        times=times,
        frames_w=frames_w,
        outdir=outdir,
        n_fft=args.n_fft,
        frame_len=args.frame_len,
        hop_len=args.hop_len,
    )

    metrics = validate_fft(fs=fs, n_fft=args.n_fft)
    metrics.update(
        {
            "input_wav": str(Path(args.wav).resolve()),
            "sample_rate": int(fs),
            "frame_len": int(args.frame_len),
            "hop_len": int(args.hop_len),
            "n_fft": int(args.n_fft),
            "window": args.window,
            "pre_emphasis_coefficient": 0.97,
            "power_normalization": "abs(FFT)**2 / n_fft",
            "signal_len_samples": int(signal.size),
            "n_frames": int(X_complex.shape[0]),
            "n_freq_bins_one_side": int(X_complex.shape[1]),
            "selected_frame": int(frame_idx),
            "selected_frame_dft_vs_fft_max_abs_err": float(max_err),
        }
    )

    with open(outdir / "fft_validation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output dir: {outdir.resolve()}")
    print(f"X_complex shape: {X_complex.shape}")
    print(f"power shape:     {power.shape}")
    print(f"freqs shape:     {freqs.shape}")
    print(f"times shape:     {times.shape}")
    print(f"selected frame:  {frame_idx}")
    print(f"DFT vs FFT max abs error: {max_err:.3e}")
    print(f"Validation json: {(outdir / 'fft_validation.json').resolve()}")


if __name__ == "__main__":
    main()
