from __future__ import annotations

import numpy as np
import pytest

from scripts.experiment2_mfcc import hz_to_mel, mel_filterbank, mel_to_hz, mfcc_from_power


def test_mel_hz_round_trip():
    frequencies = np.array([0.0, 100.0, 1000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(frequencies)), frequencies)


def test_filterbank_shape_and_order():
    bank, centers = mel_filterbank()
    assert bank.shape == (26, 257)
    assert centers.shape == (26,)
    assert np.all(np.diff(centers) > 0)
    assert np.all(bank >= 0)


def test_mfcc_shape_and_finite_for_silence():
    mel, mfcc, _ = mfcc_from_power(np.zeros((7, 257), dtype=np.float32))
    assert mel.shape == (7, 26)
    assert mfcc.shape == (7, 13)
    assert np.isfinite(mfcc).all()


def test_rejects_bad_power_input():
    with pytest.raises(ValueError):
        mfcc_from_power(np.full((2, 257), -1.0))
