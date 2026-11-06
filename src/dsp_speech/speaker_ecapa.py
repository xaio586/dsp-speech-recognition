"""Deployed speaker recognition: pretrained ECAPA-TDNN embeddings + cosine centroids.

The encoder (SpeechBrain spkrec-ecapa-voxceleb) is exported to ONNX so the
hosted service only needs onnxruntime + numpy — no PyTorch. Each enrolled
member is represented by the mean L2-normalized embedding of their training
recordings (data/speaker_recognition/ecapa_centroids.npz, built by
scripts/build_ecapa_centroids.py). The classical MFCC/LPC + RBF SVM system in
speaker_recognition.py is kept as the self-implemented course pipeline and as
a fallback when the ONNX model files are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .speaker_recognition import CONFIDENCE_THRESHOLD, trim_utterance

SOFTMAX_TEMPERATURE = 10.0  # cosine similarities of enrolled speakers are close; sharpen for display


@dataclass
class EcapaSpeakerRecognizer:
    session: object  # onnxruntime.InferenceSession
    centroids: dict[str, np.ndarray]
    names: dict[str, str]
    confidence_threshold: float = CONFIDENCE_THRESHOLD

    def embed(self, samples_16k: np.ndarray) -> np.ndarray:
        wav = np.ascontiguousarray(samples_16k, dtype=np.float32).reshape(1, -1)
        return np.asarray(self.session.run(["emb"], {"wav": wav})[0]).flatten()

    def predict_waveform(self, samples_16k: np.ndarray) -> dict[str, object]:
        embedding = self.embed(samples_16k)
        unit = embedding / (float(np.linalg.norm(embedding)) + 1e-9)
        similarities = {
            sid: float(np.dot(unit, centroid)) for sid, centroid in self.centroids.items()
        }
        ordered = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        best_id, best_similarity = ordered[0]
        weights = np.exp(np.array([similarities[s] for s in sorted(similarities)]) * SOFTMAX_TEMPERATURE)
        weights /= float(np.sum(weights))
        probabilities = dict(zip(sorted(similarities), (float(w) for w in weights)))
        confidence = float(probabilities[best_id])
        return {
            "speaker_id": best_id,
            "speaker_name": self.names.get(best_id, best_id),
            "confidence": confidence,
            "low_confidence": confidence < self.confidence_threshold,
            "probabilities": probabilities,
            "similarity": best_similarity,
        }


def load_ecapa_recognizer(root: str | Path) -> tuple[EcapaSpeakerRecognizer, dict[str, str]] | None:
    """Load the ONNX encoder + centroids; returns None when files are missing."""
    import csv

    import onnxruntime as ort

    root = Path(root).resolve()
    model_path = root / "data" / "speaker_recognition" / "ecapa_encoder.onnx"
    centroids_path = root / "data" / "speaker_recognition" / "ecapa_centroids.npz"
    manifest = root / "data" / "speaker_recognition" / "metadata.csv"
    if not (model_path.is_file() and centroids_path.is_file() and manifest.is_file()):
        return None
    names: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        import csv as csv_module

        for row in csv_module.DictReader(handle):
            names[row["speaker_id"]] = row["speaker_name"]
    session = ort.InferenceSession(str(model_path))
    centroids: dict[str, np.ndarray] = {}
    with np.load(centroids_path, allow_pickle=False) as archive:
        for sid in archive.files:
            vector = np.asarray(archive[sid], dtype=np.float32)
            centroids[sid] = vector / (float(np.linalg.norm(vector)) + 1e-9)
    if len(centroids) < 2:
        return None
    return EcapaSpeakerRecognizer(session, centroids, names), names


def recognize_speaker_ecapa(
    samples: np.ndarray,
    sample_rate: int,
    recognizer: EcapaSpeakerRecognizer,
    names: dict[str, str] | None = None,
) -> dict[str, object]:
    """Recognize one live utterance through the ECAPA embedding pipeline."""
    cropped, details = trim_utterance(samples, sample_rate)
    prediction = recognizer.predict_waveform(cropped)
    probabilities: dict[str, float] = prediction["probabilities"]  # type: ignore[assignment]
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "speaker_id": prediction["speaker_id"],
        "speaker_name": prediction["speaker_name"],
        "confidence": prediction["confidence"],
        "low_confidence": prediction["low_confidence"],
        "scores": {
            sid: {"name": recognizer.names.get(sid, sid), "probability": value}
            for sid, value in ordered
        },
        "audio": details,
    }
