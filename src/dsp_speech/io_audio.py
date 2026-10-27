"""WAV loading and normalization helpers used by the experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioData:
    """A mono floating-point audio signal and its sampling rate."""

    samples: np.ndarray
    sample_rate: int
    original_channels: int
    original_dtype: str

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def _to_float32(samples: np.ndarray) -> np.ndarray:
    """Convert common PCM arrays to float32 in approximately [-1, 1]."""
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if np.issubdtype(samples.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        return samples.astype(np.float32) / scale
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        midpoint = (info.max + 1) / 2.0
        return (samples.astype(np.float32) - midpoint) / midpoint
    raise TypeError(f"Unsupported WAV sample type: {samples.dtype}")


def load_wav(path: str | Path, target_sample_rate: int | None = None) -> AudioData:
    """Load a WAV file, mix it to mono, and optionally resample it."""
    audio_path = Path(path)
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("load_wav only accepts real .wav files; convert M4A before loading")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    sample_rate, raw = wavfile.read(audio_path)
    original_dtype = str(raw.dtype)
    original_channels = 1 if raw.ndim == 1 else raw.shape[1]
    samples = _to_float32(raw)

    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono or multi-channel audio, got shape {samples.shape}")

    if target_sample_rate is not None and target_sample_rate != sample_rate:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        divisor = int(np.gcd(sample_rate, target_sample_rate))
        samples = resample_poly(
            samples,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
        sample_rate = target_sample_rate

    return AudioData(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=int(sample_rate),
        original_channels=original_channels,
        original_dtype=original_dtype,
    )

