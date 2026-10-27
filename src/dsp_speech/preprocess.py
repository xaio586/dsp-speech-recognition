"""Preprocessing and short-time analysis for experiment 1."""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, ExperimentConfig


def remove_dc(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    return values - np.mean(values, dtype=np.float64)


def normalize_amplitude(
    samples: np.ndarray, peak: float = DEFAULT_CONFIG.normalize_peak
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if not 0.0 < peak <= 1.0:
        raise ValueError("peak must be in (0, 1]")
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum == 0.0:
        return values.copy()
    return (values * (peak / maximum)).astype(np.float32)


def pre_emphasis(
    samples: np.ndarray, coefficient: float = DEFAULT_CONFIG.pre_emphasis
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if not 0.0 <= coefficient < 1.0:
        raise ValueError("coefficient must be in [0, 1)")
    if values.size == 0:
        return values.copy()
    output = np.empty_like(values)
    output[0] = values[0]
    output[1:] = values[1:] - coefficient * values[:-1]
    return output


def frame_signal(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = DEFAULT_CONFIG.frame_ms,
    hop_ms: float = DEFAULT_CONFIG.hop_ms,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono signal")
    if sample_rate <= 0 or frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("sample_rate, frame_ms, and hop_ms must be positive")
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    hop_length = max(1, round(sample_rate * hop_ms / 1000))
    if values.size <= frame_length:
        frame_count = 1
    else:
        frame_count = 1 + int(np.ceil((values.size - frame_length) / hop_length))
    total = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(values, (0, max(0, total - values.size)))
    starts = np.arange(frame_count)[:, None] * hop_length
    offsets = np.arange(frame_length)[None, :]
    return np.ascontiguousarray(padded[starts + offsets], dtype=np.float32)


def apply_window(
    frames: np.ndarray, window: str = DEFAULT_CONFIG.window
) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("frames must have shape (frame_count, frame_length)")
    functions = {
        "hamming": np.hamming,
        "hann": np.hanning,
        "hanning": np.hanning,
        "rect": np.ones,
        "rectangular": np.ones,
    }
    if window not in functions:
        raise ValueError(f"unsupported window: {window}")
    weights = functions[window](values.shape[1]).astype(np.float32)
    return values * weights


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    return np.sum(values * values, axis=1, dtype=np.float64).astype(np.float32)


def short_time_magnitude(frames: np.ndarray) -> np.ndarray:
    """Return the sum of absolute sample amplitudes in each frame."""
    return np.sum(np.abs(np.asarray(frames, dtype=np.float32)), axis=1).astype(np.float32)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    signs = np.where(values >= 0, 1, -1)
    return (np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1) / values.shape[1]).astype(
        np.float32
    )


def detect_endpoint(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = DEFAULT_CONFIG.frame_ms,
    hop_ms: float = DEFAULT_CONFIG.hop_ms,
    *,
    window: str = DEFAULT_CONFIG.window,
    min_speech_ms: float = DEFAULT_CONFIG.min_speech_ms,
    min_silence_ms: float = DEFAULT_CONFIG.min_silence_ms,
    zcr_consecutive_frames: int = DEFAULT_CONFIG.zcr_consecutive_frames,
    high_energy_ratio: float = DEFAULT_CONFIG.high_energy_ratio,
    low_energy_ratio: float = DEFAULT_CONFIG.low_energy_ratio,
    min_zcr_threshold: float = DEFAULT_CONFIG.min_zcr_threshold,
    zcr_noise_multiplier: float = DEFAULT_CONFIG.zcr_noise_multiplier,
) -> tuple[int, int]:
    """Return the outer sample endpoints using D's adaptive energy/ZCR stages."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono signal")
    if values.size == 0:
        return 0, 0
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    cleaned = remove_dc(values)
    frames = frame_signal(cleaned, sample_rate, frame_ms, hop_ms)
    windowed = apply_window(frames, window)
    energy = short_time_energy(windowed)
    zcr = zero_crossing_rate(frames)
    if not len(energy) or float(np.max(energy)) <= 1e-12:
        return 0, 0

    ms_per_frame = hop_ms
    min_speech_frames = max(1, int(np.ceil(min_speech_ms / ms_per_frame)))
    min_silence_frames = max(1, int(np.ceil(min_silence_ms / ms_per_frame)))
    noise_count = min(20, max(1, len(energy) // 5))
    noise_energy = energy[:noise_count]
    noise_zcr = zcr[:noise_count]
    energy_mean = float(np.mean(noise_energy))
    energy_std = float(np.std(noise_energy))
    zcr_baseline = float(np.median(noise_zcr))
    peak_energy = float(np.max(energy))

    low = energy_mean + 2.0 * energy_std + low_energy_ratio * peak_energy
    low = min(low, peak_energy * 0.5)
    high = min(
        peak_energy,
        max(low, energy_mean + 2.0 * energy_std + high_energy_ratio * peak_energy),
    )
    zcr_threshold = max(min_zcr_threshold, zcr_baseline * zcr_noise_multiplier)
    core = np.flatnonzero(energy >= high)
    if not core.size:
        return 0, 0

    # Find continuous high-energy regions, allowing short internal pauses.
    runs: list[list[int]] = []
    start = previous = int(core[0])
    for frame_index in core[1:]:
        current = int(frame_index)
        if current - previous > min_silence_frames:
            runs.append([start, previous])
            start = current
        previous = current
    runs.append([start, previous])

    # Extend each core to the lower-energy boundary or nearby unvoiced frames.
    expanded: list[list[int]] = []
    for start, end in runs:
        left = start
        while left > 0:
            candidate = left - 1
            if energy[candidate] >= low:
                left -= 1
                continue
            run_start = candidate
            while run_start > 0 and zcr[run_start - 1] >= zcr_threshold:
                run_start -= 1
            if candidate - run_start + 1 >= zcr_consecutive_frames:
                left = run_start
                continue
            break
        right = end
        while right + 1 < len(energy):
            candidate = right + 1
            if energy[candidate] >= low:
                right += 1
                continue
            run_end = candidate
            while run_end + 1 < len(energy) and zcr[run_end + 1] >= zcr_threshold:
                run_end += 1
            if run_end - candidate + 1 >= zcr_consecutive_frames:
                right = run_end
                continue
            break
        expanded.append([left, right])

    merged: list[list[int]] = []
    for start, end in expanded:
        if merged and start - merged[-1][1] <= min_silence_frames:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    valid = [segment for segment in merged if segment[1] - segment[0] + 1 >= min_speech_frames]
    if not valid:
        return 0, 0

    hop_length = max(1, round(sample_rate * hop_ms / 1000))
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    first, last = valid[0][0], valid[-1][1]
    return first * hop_length, min(len(cleaned), last * hop_length + frame_length)


def analyze_time_domain(
    samples: np.ndarray,
    sample_rate: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """Run the shared experiment 1 path and return features plus sample endpoints."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono signal")
    cleaned = normalize_amplitude(remove_dc(values), peak=config.normalize_peak)
    frames = frame_signal(cleaned, sample_rate, config.frame_ms, config.hop_ms)
    windowed = apply_window(frames, config.window)
    energy = short_time_energy(windowed)
    magnitude = short_time_magnitude(windowed)
    zcr = zero_crossing_rate(frames)
    endpoints = detect_endpoint(
        cleaned,
        sample_rate,
        config.frame_ms,
        config.hop_ms,
        window=config.window,
        min_speech_ms=config.min_speech_ms,
        min_silence_ms=config.min_silence_ms,
        zcr_consecutive_frames=config.zcr_consecutive_frames,
        high_energy_ratio=config.high_energy_ratio,
        low_energy_ratio=config.low_energy_ratio,
        min_zcr_threshold=config.min_zcr_threshold,
        zcr_noise_multiplier=config.zcr_noise_multiplier,
    )
    return {
        "samples": cleaned,
        "frames": frames,
        "windowed_frames": windowed,
        "energy": energy,
        "magnitude": magnitude,
        "zcr": zcr,
        "endpoints": endpoints,
    }
