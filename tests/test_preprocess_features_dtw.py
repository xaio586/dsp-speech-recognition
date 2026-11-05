import numpy as np

from dsp_speech.config import DEFAULT_CONFIG
from dsp_speech.dtw import dtw_distance, recognize_digit
from dsp_speech.features import extract_mfcc
from dsp_speech.preprocess import (
    apply_window,
    detect_endpoint,
    frame_signal,
    pre_emphasis,
    short_time_energy,
    short_time_magnitude,
    zero_crossing_rate,
)


def test_framing_and_features_have_expected_shapes() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    frames = frame_signal(pre_emphasis(signal), sample_rate)
    windowed = apply_window(frames)
    assert frames.shape == (99, 400)
    assert windowed.shape == frames.shape
    assert short_time_energy(frames).shape == (99,)
    assert zero_crossing_rate(frames).shape == (99,)
    assert extract_mfcc(signal, sample_rate).shape == (99, 13)


def test_member_c_framing_and_window_cases() -> None:
    sample_rate = DEFAULT_CONFIG.sample_rate
    signal = np.zeros(sample_rate, dtype=np.float32)
    frames = frame_signal(signal, sample_rate)
    assert frames.shape == (99, 400)
    assert np.array_equal(apply_window(np.ones((1, 4)), "rect"), np.ones((1, 4)))
    assert np.allclose(apply_window(np.ones((1, 4)), "hanning"), np.hanning(4)[None, :])
    assert np.allclose(apply_window(np.ones((1, 4)), "hamming"), np.hamming(4)[None, :])


def test_member_d_short_time_magnitude_and_zero_crossing_cases() -> None:
    frames = np.array([[1.0, -2.0, 3.0], [-1.0, -1.0, 2.0]], dtype=np.float32)
    assert np.allclose(short_time_energy(frames), [14.0, 6.0])
    assert np.allclose(short_time_magnitude(frames), [6.0, 4.0])
    assert np.allclose(zero_crossing_rate(np.array([[1.0, 0.0, -1.0, 1.0]])), [0.5])


def test_endpoint_contains_synthetic_tone() -> None:
    sample_rate = 16_000
    signal = np.zeros(sample_rate, dtype=np.float32)
    signal[4_000:12_000] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(8_000) / sample_rate)
    start, end = detect_endpoint(signal, sample_rate)
    assert start <= 4_000
    assert end >= 12_000
    assert start < end


def test_dtw_and_recognizer() -> None:
    zero = np.zeros((4, 2), dtype=np.float32)
    one = np.ones((5, 2), dtype=np.float32)
    assert dtw_distance(zero, zero) == 0.0
    label, score = recognize_digit(zero, {"0": [zero], "1": [one]})
    assert label == "0"
    assert score == 0.0
