"""Run the shared B preprocessing, D Mel filter bank, and E MFCC pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "reports" / "experiment2_c_submission") not in sys.path:
    sys.path.insert(0, str(ROOT / "reports" / "experiment2_c_submission"))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from preprocess_b import N_FFT, preprocess_waveform  # noqa: E402
from week3_c import load_wav, stft_from_signal  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "experiment2_mfcc"


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = 16000, n_fft: int = 512, n_filters: int = 26,
    low_hz: float = 0.0, high_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build triangular filters on the one-sided FFT frequency-bin grid."""
    high_hz = sample_rate / 2 if high_hz is None else high_hz
    if not 0 <= low_hz < high_hz <= sample_rate / 2:
        raise ValueError("Mel frequency limits must lie within [0, Nyquist]")
    points = mel_to_hz(np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_filters + 2))
    bins = np.floor(points * n_fft / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)
    for m, (left, center, right) in enumerate(zip(bins[:-2], bins[1:-1], bins[2:])):
        if center <= left or right <= center:
            raise ValueError(f"Degenerate Mel filter {m}; change filter count or frequency limits")
        bank[m, left:center] = (np.arange(left, center) - left) / (center - left)
        bank[m, center:right + 1] = (right - np.arange(center, right + 1)) / (right - center)
    return bank, points[1:-1]


def mfcc_from_power(
    power: np.ndarray, n_filters: int = 26, n_mfcc: int = 13, n_fft: int = N_FFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] != n_fft // 2 + 1:
        raise ValueError(f"power must have shape (frames, {n_fft // 2 + 1})")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("power must contain finite, non-negative values")
    bank, centers = mel_filterbank(DEFAULT_CONFIG.sample_rate, n_fft, n_filters)
    mel_energy = power @ bank.T
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mel_energy.astype(np.float32), mfcc.astype(np.float32), centers


def process_wav(wav_path: Path, output_root: Path) -> dict[str, object]:
    fs, samples = load_wav(wav_path)
    samples = np.asarray(samples, dtype=np.float32)
    if fs != DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"{wav_path}: expected 16 kHz WAV, received {fs} Hz")
    power, quality = preprocess_waveform(samples, fs)
    mel, mfcc, centers = mfcc_from_power(power)
    relative = wav_path.relative_to(ROOT / "data" / "processed")
    target = output_root / relative.with_suffix("")
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "power.npy", power, allow_pickle=False)
    np.save(target / "log_mel.npy", np.log(np.maximum(mel, 1e-10)), allow_pickle=False)
    np.save(target / "mfcc.npy", mfcc, allow_pickle=False)
    return {
        "input_file": (Path("data/processed") / relative).as_posix(),
        "output_directory": target.relative_to(ROOT).as_posix(),
        "sample_rate_hz": fs,
        "power_shape": list(power.shape),
        "log_mel_shape": list(mel.shape),
        "mfcc_shape": list(mfcc.shape),
        "mel_center_hz": centers.tolist(),
        **quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="process one WAV; defaults to all project WAVs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    wavs = [args.wav.resolve()] if args.wav else sorted((ROOT / "data" / "processed").rglob("*.wav"))
    if not wavs:
        raise SystemExit("No WAV files found under data/processed")
    results = [process_wav(wav, args.output.resolve()) for wav in wavs]
    manifest = {
        "input_count": len(wavs),
        "sample_rate_hz": DEFAULT_CONFIG.sample_rate,
        "pre_emphasis": DEFAULT_CONFIG.pre_emphasis,
        "frame_length_ms": DEFAULT_CONFIG.frame_ms,
        "frame_hop_ms": DEFAULT_CONFIG.hop_ms,
        "window": DEFAULT_CONFIG.window,
        "n_fft": N_FFT,
        "n_mel_filters": 26,
        "n_mfcc": 13,
        "power_definition": "abs(rfft(windowed_frame))**2 / n_fft",
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Processed {len(results)} WAV file(s); results: {args.output}")


if __name__ == "__main__":
    main()
