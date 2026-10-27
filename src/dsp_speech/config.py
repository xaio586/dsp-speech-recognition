"""Shared parameters for the speech-processing experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared by the experiment 1 analysis pipeline."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    window: str = "hamming"
    pre_emphasis: float = 0.97
    normalize_peak: float = 0.95
    min_speech_ms: float = 50.0
    min_silence_ms: float = 120.0
    zcr_consecutive_frames: int = 3
    high_energy_ratio: float = 0.10
    low_energy_ratio: float = 0.025
    min_zcr_threshold: float = 0.08
    zcr_noise_multiplier: float = 2.0


DEFAULT_CONFIG = ExperimentConfig()
