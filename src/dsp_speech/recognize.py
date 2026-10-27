"""Template loading and live-utterance preparation for experiment 3."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .config import DEFAULT_CONFIG
from .dtw import dtw_distance
from .features import extract_mfcc
from .io_audio import load_wav
from .preprocess import detect_endpoint, normalize_amplitude, remove_dc


def _fallback_endpoint(samples: np.ndarray, sample_rate: int) -> tuple[int, int]:
    """Find a speech region from frame RMS when the shared detector is inconclusive."""
    frame_length = max(1, round(sample_rate * 0.025))
    hop_length = max(1, round(sample_rate * 0.010))
    if samples.size < frame_length:
        return 0, int(samples.size)
    frame_count = 1 + (samples.size - frame_length) // hop_length
    starts = np.arange(frame_count) * hop_length
    frames = np.stack([samples[start : start + frame_length] for start in starts])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(np.max(rms))
    if peak <= 1e-7:
        return 0, 0

    edge_count = max(1, min(frame_count // 4, round(0.20 / (hop_length / sample_rate))))
    noise_floor = float(np.median(np.concatenate((rms[:edge_count], rms[-edge_count:]))))
    threshold = max(noise_floor * 2.5, peak * 0.06, 1e-6)
    active = np.flatnonzero(rms >= threshold)
    if not active.size:
        return 0, 0

    pad = round(0.07 * sample_rate)
    start = max(0, int(starts[active[0]]) - pad)
    end = min(samples.size, int(starts[active[-1]] + frame_length) + pad)
    return start, end


def prepare_utterance(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize, trim, and extract MFCC from one microphone utterance."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("录音必须是单声道音频")
    if sample_rate <= 0 or values.size == 0:
        raise ValueError("没有收到有效录音")
    if not np.isfinite(values).all():
        raise ValueError("录音中包含无效数值")

    if sample_rate != DEFAULT_CONFIG.sample_rate:
        divisor = int(np.gcd(sample_rate, DEFAULT_CONFIG.sample_rate))
        values = resample_poly(
            values,
            DEFAULT_CONFIG.sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
        sample_rate = DEFAULT_CONFIG.sample_rate

    values = remove_dc(values)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    if peak < 0.008 or rms < 0.0015:
        raise ValueError("录音声音太小或接近静音，请靠近麦克风再试一次")

    normalized = normalize_amplitude(values, peak=DEFAULT_CONFIG.normalize_peak)
    start, end = detect_endpoint(normalized, sample_rate)
    if end <= start or (end - start) / sample_rate < 0.18:
        start, end = _fallback_endpoint(normalized, sample_rate)
    if end <= start or (end - start) / sample_rate < 0.18:
        raise ValueError("没有检测到完整数字语音，请说一个数字后再停止录音")

    cropped = np.ascontiguousarray(normalized[start:end], dtype=np.float32)
    features = extract_mfcc(cropped, sample_rate)
    if not features.size or not np.isfinite(features).all():
        raise ValueError("语音特征提取失败，请重新录制")
    details = {
        "sample_rate": float(sample_rate),
        "start_seconds": start / sample_rate,
        "end_seconds": end / sample_rate,
        "speech_seconds": len(cropped) / sample_rate,
        "peak": peak,
        "rms": rms,
        "frame_count": float(len(features)),
    }
    return features, details


def load_digit_templates(
    project_root: str | Path, train_take: int | None = None
) -> dict[str, list[np.ndarray]]:
    """Load digit recordings listed in processed/metadata.csv as MFCC templates."""
    root = Path(project_root).resolve()
    metadata_path = root / "data" / "processed" / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到数据清单：{metadata_path}")

    templates: dict[str, list[np.ndarray]] = {str(digit): [] for digit in range(10)}
    skipped: list[str] = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("content", "")).strip()
            if label not in templates:
                continue
            if train_take is not None and str(row.get("take", "")).strip() != str(train_take):
                continue
            relative = Path(*str(row.get("relative_path", "")).split("/"))
            audio_path = (root / relative).resolve()
            if not audio_path.is_relative_to(root):
                raise ValueError(f"数据清单包含仓库目录外的路径：{relative}")
            if not audio_path.is_file():
                skipped.append(str(relative))
                continue
            audio = load_wav(audio_path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
            features, _ = prepare_utterance(audio.samples, audio.sample_rate)
            templates[label].append(features)

    missing = [label for label, examples in templates.items() if not examples]
    if missing:
        detail = f"；缺失文件示例：{', '.join(skipped[:3])}" if skipped else ""
        take_detail = f"take{train_take:02d}" if train_take is not None else "全部轮次"
        raise ValueError(f"{take_detail}模板不完整，缺少数字：{', '.join(missing)}{detail}")
    return templates


def score_features(
    features: np.ndarray, templates: dict[str, list[np.ndarray]]
) -> dict[str, float]:
    """Return each digit's closest-template DTW distance, sorted by distance."""
    scores = {
        label: min(dtw_distance(features, template) for template in examples)
        for label, examples in templates.items()
        if examples
    }
    if not scores:
        raise ValueError("没有可用的数字模板")
    return dict(sorted(scores.items(), key=lambda item: item[1]))


def recognize_utterance(
    samples: np.ndarray, sample_rate: int, templates: dict[str, list[np.ndarray]]
) -> dict[str, object]:
    """Run the complete microphone path and return prediction plus diagnostics."""
    features, details = prepare_utterance(samples, sample_rate)
    scores = score_features(features, templates)
    label = next(iter(scores))
    distance = scores[label]
    return {
        "digit": label,
        "distance": distance,
        "scores": scores,
        "audio": details,
    }
