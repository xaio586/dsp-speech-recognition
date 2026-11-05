import csv
from pathlib import Path

import numpy as np
import pytest

from dsp_speech.io_audio import load_wav
from dsp_speech.speaker_recognition import (
    FEATURE_DIM,
    SpeakerSVM,
    load_speaker_model,
    load_speaker_templates,
    recognize_speaker,
    utterance_feature_vector,
)


ROOT = Path(__file__).resolve().parents[1]


def test_five_speaker_templates_and_heldout_recording():
    templates, names = load_speaker_templates(ROOT)
    assert len(templates) == 5
    assert all(len(items) == 8 for items in templates.values())

    model, _ = load_speaker_model(ROOT)
    assert isinstance(model, SpeakerSVM)
    assert len(model.speaker_ids) == 5
    # 40 recordings x (original + 3 channel augmentations)
    assert sum(model.training_counts.values()) == 160

    with (ROOT / "data" / "speaker_recognition" / "metadata.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        sample = next(
            row
            for row in csv.DictReader(handle)
            if row["speaker_id"] == "spk01" and row["split"] == "test"
        )
    audio_path = ROOT / "data" / "speaker_recognition" / sample["file_name"]
    if not audio_path.is_file():
        pytest.skip("展开完整说话人数据包后可运行留出录音识别核验")
    audio = load_wav(
        audio_path,
        target_sample_rate=16_000,
    )
    result = recognize_speaker(audio.samples, audio.sample_rate, model, names)
    assert result["speaker_id"] == sample["speaker_id"]
    assert len(result["scores"]) == 5
    probabilities = [entry["probability"] for entry in result["scores"].values()]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= result["confidence"] <= 1.0


def test_utterance_feature_vector_is_fixed_length_and_content_agnostic():
    rng = np.random.default_rng(2026)
    mfcc_a = rng.normal(size=(120, 13))
    mfcc_b = rng.normal(size=(57, 13))
    lpc_a = rng.normal(size=(120, 12))
    lpc_b = rng.normal(size=(57, 12))
    vector_a = utterance_feature_vector(mfcc_a, lpc_a)
    vector_b = utterance_feature_vector(mfcc_b, lpc_b)
    assert vector_a.shape == (FEATURE_DIM,)
    assert vector_b.shape == (FEATURE_DIM,)
    assert np.isfinite(vector_a).all() and np.isfinite(vector_b).all()
    # MFCC-only fallback (templates.npz deployment path) stays 78-dim.
    assert utterance_feature_vector(mfcc_a).shape == (78,)


def test_low_confidence_flag_tracks_threshold():
    class _ConstantMargin:
        def decision_function(self, matrix):
            return np.zeros((matrix.shape[0], 5))

    model = SpeakerSVM(
        classifier=_ConstantMargin(),
        speaker_ids=[f"spk0{index}" for index in range(1, 6)],
        names={},
    )
    # Uniform margins -> probability 0.2 everywhere -> below the 0.55 threshold.
    prediction = model.predict_vector(np.zeros(FEATURE_DIM))
    assert prediction["low_confidence"] is True
    assert prediction["confidence"] == pytest.approx(0.2, abs=1e-6)
