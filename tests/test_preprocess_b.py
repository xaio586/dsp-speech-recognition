import numpy as np
import pytest

from scripts.preprocess_b import _check_quality, preprocess_waveform


def test_b_preprocessing_uses_shared_frame_and_spectrum_shape() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    power, quality = preprocess_waveform(signal, sample_rate)
    assert power.shape == (99, 257)
    assert np.isfinite(power).all()
    assert quality["duration_seconds"] == 1.0
    assert not _check_quality(quality)


def test_b_quality_check_flags_short_silent_and_clipped_audio() -> None:
    _, short_quality = preprocess_waveform(np.zeros(1_000, dtype=np.float32), 16_000)
    assert "too_short" in _check_quality(short_quality)
    assert "silent_or_near_silent" in _check_quality(short_quality)

    clipped = np.ones(8_000, dtype=np.float32)
    _, clipped_quality = preprocess_waveform(clipped, 16_000)
    assert "possible_clipping" in _check_quality(clipped_quality)


def test_b_preprocessing_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="non-empty mono"):
        preprocess_waveform(np.zeros((2, 2)), 16_000)
    with pytest.raises(ValueError, match="finite"):
        preprocess_waveform(np.array([np.nan], dtype=np.float32), 16_000)
