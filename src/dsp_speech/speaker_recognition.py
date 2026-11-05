"""Experiment 4 speaker recognition: utterance-level MFCC/LPC statistics + RBF SVM.

Unlike experiment 3 (per-frame DTW template matching against fixed sentence
recordings), this module compresses each utterance into a fixed-length speaker
feature vector — per-coefficient mean/std of MFCC, its delta/delta-delta, and
per-frame LPC coefficients — and classifies it with a supervised
StandardScaler + RBF-SVM pipeline. Recognition therefore relies on voice
timbre statistics rather than on matching the spoken content frame by frame.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DEFAULT_CONFIG
from .features import extract_mfcc
from .io_audio import load_wav
from .preprocess import apply_window, detect_endpoint, frame_signal, normalize_amplitude, pre_emphasis, remove_dc
from .recognize import _fallback_endpoint, prepare_utterance

# Hyperparameters selected by leave-one-sentence-out grouped cross-validation
# on the 40 training utterances (see scripts/evaluate_speaker_svm.py).
SVM_C = 1.0
SVM_GAMMA = "scale"

# Below this top-class probability the closed-set answer is flagged as unsure.
CONFIDENCE_THRESHOLD = 0.55

LPC_ORDER = 12
N_MFCC = 13
# 13 MFCC * (mean,std) * (static,delta,delta2) + 12 LPC * (mean,std)
FEATURE_DIM = N_MFCC * 3 * 2 + LPC_ORDER * 2


def _levinson_durbin(autocorr: np.ndarray, order: int) -> np.ndarray:
    """Solve the Yule-Walker equations for LPC coefficients (a[1..order])."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    error = autocorr[0]
    for k in range(1, order + 1):
        acc = autocorr[k] + float(np.dot(a[1:k], autocorr[k - 1 : 0 : -1]))
        reflection = -acc / error
        previous = a.copy()
        a[k] = reflection
        for i in range(1, k):
            a[i] = previous[i] + reflection * previous[k - i]
        error *= 1.0 - reflection * reflection
        if error <= 1e-12:
            break
    return a[1:]


def extract_lpc_frames(samples: np.ndarray, sample_rate: int, order: int = LPC_ORDER) -> np.ndarray:
    """Per-frame LPC coefficients via the autocorrelation method."""
    emphasized = pre_emphasis(samples, DEFAULT_CONFIG.pre_emphasis)
    frames = apply_window(
        frame_signal(emphasized, sample_rate, DEFAULT_CONFIG.frame_ms, DEFAULT_CONFIG.hop_ms),
        DEFAULT_CONFIG.window,
    )
    result = np.zeros((len(frames), order), dtype=np.float64)
    for index, frame in enumerate(frames):
        autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1 : len(frame) - 1 + order + 1]
        if autocorr[0] <= 1e-12:
            continue
        result[index] = _levinson_durbin(autocorr, order)
    return result


def prepare_speaker_utterance(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Trim/resample one utterance and return (MFCC frames, LPC frames, details).

    Mirrors recognize.prepare_utterance (shared with experiment 3) but also
    returns the LPC frame sequence needed by the speaker feature vector.
    """
    mfcc, details = prepare_utterance(samples, sample_rate)

    values = np.asarray(samples, dtype=np.float32)
    if sample_rate != DEFAULT_CONFIG.sample_rate:
        divisor = int(np.gcd(sample_rate, DEFAULT_CONFIG.sample_rate))
        values = resample_poly(
            values, DEFAULT_CONFIG.sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
        sample_rate = DEFAULT_CONFIG.sample_rate
    values = remove_dc(values)
    normalized = normalize_amplitude(values, peak=DEFAULT_CONFIG.normalize_peak)
    start, end = detect_endpoint(normalized, sample_rate)
    if end <= start or (end - start) / sample_rate < 0.18:
        start, end = _fallback_endpoint(normalized, sample_rate)
    cropped = np.ascontiguousarray(normalized[start:end], dtype=np.float32)
    lpc = extract_lpc_frames(cropped, sample_rate)
    return mfcc, lpc, details


def utterance_feature_vector(mfcc: np.ndarray, lpc: np.ndarray | None = None) -> np.ndarray:
    """Compress frame-level MFCC (+LPC) sequences into a fixed-length vector.

    Layout: for MFCC and its delta / delta-delta, the per-coefficient mean and
    standard deviation (13 * 3 * 2 = 78 dims); for per-frame LPC coefficients,
    mean and std (12 * 2 = 24 dims). Total 102 dims describing timbre
    statistics rather than word timing.
    """
    frames = np.asarray(mfcc, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[0] < 3:
        raise ValueError("MFCC 帧数不足，无法统计说话人特征")
    delta = np.gradient(frames, axis=0)
    delta2 = np.gradient(delta, axis=0)
    parts = []
    for block in (frames, delta, delta2):
        parts.append(block.mean(axis=0))
        parts.append(block.std(axis=0))
    if lpc is not None:
        lpc = np.asarray(lpc, dtype=np.float64)
        parts.append(lpc.mean(axis=0))
        parts.append(lpc.std(axis=0))
    return np.concatenate(parts)


@dataclass
class SpeakerSVM:
    """Trained closed-set speaker classifier with probability-style scores."""

    classifier: object
    speaker_ids: list[str]
    names: dict[str, str]
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    feature_dim: int = field(default=FEATURE_DIM)
    training_counts: dict[str, int] = field(default_factory=dict)

    def _probabilities(self, vector: np.ndarray) -> dict[str, float]:
        decision = np.asarray(self.classifier.decision_function(vector.reshape(1, -1)))[0]
        if decision.size != len(self.speaker_ids):
            # Binary edge case; expand one-vs-rest margins defensively.
            decision = np.concatenate([decision, -decision])
        shifted = decision - float(np.max(decision))
        weights = np.exp(shifted)
        weights /= float(np.sum(weights))
        return {sid: float(w) for sid, w in zip(self.speaker_ids, weights)}

    def predict_vector(self, vector: np.ndarray) -> dict[str, object]:
        probabilities = self._probabilities(np.asarray(vector, dtype=np.float64))
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        best_id, confidence = ordered[0]
        return {
            "speaker_id": best_id,
            "speaker_name": self.names.get(best_id, best_id),
            "confidence": confidence,
            "low_confidence": confidence < self.confidence_threshold,
            "probabilities": probabilities,
        }

    def predict_mfcc(self, mfcc: np.ndarray, lpc: np.ndarray | None = None) -> dict[str, object]:
        return self.predict_vector(utterance_feature_vector(mfcc, lpc))


def load_speaker_templates(root: str | Path) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
    """Load training MFCC sequences (templates.npz preferred, WAV fallback)."""
    root = Path(root).resolve()
    manifest = root / "data" / "speaker_recognition" / "metadata.csv"
    templates: dict[str, list[np.ndarray]] = {}
    names: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        names[row["speaker_id"]] = row["speaker_name"]

    packed = root / "data" / "speaker_recognition" / "templates.npz"
    if packed.is_file():
        with np.load(packed, allow_pickle=False) as archive:
            for sid in names:
                keys = sorted(key for key in archive.files if key.startswith(f"{sid}_"))
                if keys:
                    templates[sid] = [np.asarray(archive[key], dtype=np.float32) for key in keys]
        if len(templates) < 2:
            raise ValueError("压缩说话人模板不足")
        return templates, names

    for row in rows:
        if row.get("split") != "train":
            continue
        sid = row["speaker_id"]
        audio = load_wav(root / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
        features, _ = prepare_utterance(audio.samples, audio.sample_rate)
        templates.setdefault(sid, []).append(features)
    if len(templates) < 2:
        raise ValueError("说话人模板不足：至少需要两位说话人的训练录音")
    return templates, names


def load_speaker_feature_archive(root: str | Path) -> dict[str, list[np.ndarray]] | None:
    """Load precomputed utterance-level feature vectors, if the archive exists."""
    archive_path = Path(root).resolve() / "data" / "speaker_recognition" / "svm_features.npz"
    if not archive_path.is_file():
        return None
    vectors: dict[str, list[np.ndarray]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for key in archive.files:
            sid = key.split("_")[0]
            vectors.setdefault(sid, []).append(np.asarray(archive[key], dtype=np.float64))
    return vectors or None


def train_speaker_svm(
    utterance_vectors: dict[str, list[np.ndarray]],
    names: dict[str, str],
    c: float = SVM_C,
    gamma: str | float = SVM_GAMMA,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> SpeakerSVM:
    """Fit StandardScaler + RBF SVM on utterance-level feature vectors."""
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for sid in sorted(utterance_vectors):
        for vector in utterance_vectors[sid]:
            vectors.append(np.asarray(vector, dtype=np.float64))
            labels.append(sid)
    if len(set(labels)) < 2:
        raise ValueError("说话人模板不足：至少需要两位说话人的训练录音")
    matrix = np.stack(vectors)
    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma))
    classifier.fit(matrix, labels)
    return SpeakerSVM(
        classifier=classifier,
        speaker_ids=sorted(set(labels)),
        names=dict(names),
        confidence_threshold=confidence_threshold,
        feature_dim=matrix.shape[1],
        training_counts={sid: len(utterance_vectors[sid]) for sid in utterance_vectors},
    )


def build_training_vectors(root: str | Path) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
    """Utterance-level vectors for every training recording (WAV-based)."""
    root = Path(root).resolve()
    manifest = root / "data" / "speaker_recognition" / "metadata.csv"
    vectors: dict[str, list[np.ndarray]] = {}
    names: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            names[row["speaker_id"]] = row["speaker_name"]
            if row.get("split") != "train":
                continue
            audio = load_wav(root / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
            mfcc, lpc, _ = prepare_speaker_utterance(audio.samples, audio.sample_rate)
            vectors.setdefault(row["speaker_id"], []).append(utterance_feature_vector(mfcc, lpc))
    return vectors, names


def load_speaker_model(root: str | Path) -> tuple[SpeakerSVM, dict[str, str]]:
    """Load training features and fit the SVM in one step (service startup path).

    Prefers the precomputed svm_features.npz archive (what gets deployed);
    falls back to templates.npz MFCC (78-dim model) and finally to raw WAVs.
    """
    root = Path(root).resolve()
    manifest = root / "data" / "speaker_recognition" / "metadata.csv"
    names: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            names[row["speaker_id"]] = row["speaker_name"]

    archived = load_speaker_feature_archive(root)
    if archived is not None and len(archived) >= 2:
        return train_speaker_svm(archived, names), names

    wav_dir = root / "data" / "speaker_recognition" / "wav"
    if wav_dir.is_dir():
        vectors, names = build_training_vectors(root)
        if len(vectors) >= 2:
            return train_speaker_svm(vectors, names), names

    templates, names = load_speaker_templates(root)
    mfcc_vectors = {sid: [utterance_feature_vector(m) for m in items] for sid, items in templates.items()}
    return train_speaker_svm(mfcc_vectors, names), names


def recognize_speaker(
    samples: np.ndarray,
    sample_rate: int,
    model: SpeakerSVM,
    names: dict[str, str] | None = None,
) -> dict[str, object]:
    """Recognize one live utterance; interface-compatible with the old DTW version."""
    if model.feature_dim == FEATURE_DIM:
        mfcc, lpc, details = prepare_speaker_utterance(samples, sample_rate)
        prediction = model.predict_mfcc(mfcc, lpc)
    else:
        mfcc, details = prepare_utterance(samples, sample_rate)
        prediction = model.predict_mfcc(mfcc)
    probabilities: dict[str, float] = prediction["probabilities"]  # type: ignore[assignment]
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "speaker_id": prediction["speaker_id"],
        "speaker_name": prediction["speaker_name"],
        "confidence": prediction["confidence"],
        "low_confidence": prediction["low_confidence"],
        "scores": {
            sid: {"name": model.names.get(sid, sid), "probability": value}
            for sid, value in ordered
        },
        "audio": details,
    }
