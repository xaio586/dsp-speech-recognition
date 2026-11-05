from pathlib import Path

import numpy as np
from scipy.io import wavfile

from dsp_speech.io_audio import load_wav


def test_load_wav_mixes_stereo_to_mono(tmp_path: Path) -> None:
    sample_rate = 8_000
    left = np.array([0, 10_000, -10_000, 0], dtype=np.int16)
    right = np.array([0, 5_000, -5_000, 0], dtype=np.int16)
    stereo = np.column_stack([left, right])
    path = tmp_path / "stereo.wav"
    wavfile.write(path, sample_rate, stereo)

    audio = load_wav(path)

    assert audio.sample_rate == sample_rate
    assert audio.original_channels == 2
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (4,)
    assert np.isfinite(audio.samples).all()


def test_load_wav_can_resample(tmp_path: Path) -> None:
    sample_rate = 8_000
    path = tmp_path / "mono.wav"
    wavfile.write(path, sample_rate, np.zeros(sample_rate, dtype=np.int16))

    audio = load_wav(path, target_sample_rate=16_000)

    assert audio.sample_rate == 16_000
    assert abs(audio.duration_seconds - 1.0) < 0.001


def test_rejects_non_wav_files(tmp_path: Path) -> None:
    path = tmp_path / "recording.m4a"
    path.write_bytes(b"not audio")

    try:
        load_wav(path)
    except ValueError as error:
        assert "real .wav" in str(error)
    else:
        raise AssertionError("Expected a ValueError for non-WAV input")

