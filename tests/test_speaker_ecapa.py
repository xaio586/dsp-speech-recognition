"""Tests for the deployed ECAPA-TDNN (ONNX) speaker recognition path."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dsp_speech.io_audio import load_wav
from dsp_speech.speaker_ecapa import load_ecapa_recognizer, recognize_speaker_ecapa

ROOT = Path(__file__).resolve().parents[1]
ONNX_FILES = [
    ROOT / "data" / "speaker_recognition" / "ecapa_encoder.onnx",
    ROOT / "data" / "speaker_recognition" / "ecapa_encoder.onnx.data",
    ROOT / "data" / "speaker_recognition" / "ecapa_centroids.npz",
]

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in ONNX_FILES),
    reason="ECAPA ONNX 模型文件未随仓库检出（大文件），跳过部署路径测试",
)


@pytest.fixture(scope="module")
def ecapa_bundle():
    bundle = load_ecapa_recognizer(ROOT)
    assert bundle is not None
    return bundle


def test_ecapa_recognizer_loads_five_speakers(ecapa_bundle):
    recognizer, names = ecapa_bundle
    assert len(recognizer.centroids) == 5
    assert set(names) >= set(recognizer.centroids)
    for vector in recognizer.centroids.values():
        assert vector.shape == (192,)
        assert pytest.approx(1.0, abs=1e-4) == float(vector @ vector)


def test_ecapa_identifies_heldout_recording(ecapa_bundle):
    recognizer, names = ecapa_bundle
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
    audio = load_wav(audio_path, target_sample_rate=16_000)
    result = recognize_speaker_ecapa(audio.samples, audio.sample_rate, recognizer, names)
    assert result["speaker_id"] == sample["speaker_id"]
    assert len(result["scores"]) == 5
    probabilities = [entry["probability"] for entry in result["scores"].values()]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
