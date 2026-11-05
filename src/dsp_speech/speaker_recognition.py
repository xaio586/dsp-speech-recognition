"""MFCC/DTW speaker matching using the group's enrollment recordings."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .dtw import dtw_distance
from .features import extract_mfcc
from .io_audio import load_wav
from .recognize import prepare_utterance


def _compact(features: np.ndarray, limit: int = 240) -> np.ndarray:
    """Uniformly retain a compact MFCC sequence for responsive browser matching."""
    values = np.asarray(features, dtype=np.float32)[::2]
    if len(values) > limit:
        indexes = np.linspace(0, len(values) - 1, limit).astype(np.int64)
        values = values[indexes]
    return np.ascontiguousarray(values)


def load_speaker_templates(root: str | Path) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
    root = Path(root).resolve()
    manifest = root / "data" / "speaker_recognition" / "metadata.csv"
    templates: dict[str, list[np.ndarray]] = {}
    names: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        names[row["speaker_id"]] = row["speaker_name"]

    # Hosted builds can ship compact MFCC templates instead of every WAV file.
    packed = root / "data" / "speaker_recognition" / "templates.npz"
    if packed.is_file():
        with np.load(packed, allow_pickle=False) as archive:
            for sid in names:
                keys = sorted(key for key in archive.files if key.startswith(f"{sid}_"))
                if keys:
                    templates[sid] = [_compact(archive[key]) for key in keys]
        if len(templates) < 2:
            raise ValueError("压缩说话人模板不足")
        return templates, names

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train":
                continue
            sid = row["speaker_id"]
            names[sid] = row["speaker_name"]
            audio = load_wav(root / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
            features, _ = prepare_utterance(audio.samples, audio.sample_rate)
            features = _compact(features)
            templates.setdefault(sid, []).append(features)
    if len(templates) < 2:
        raise ValueError("说话人模板不足：至少需要两位说话人的训练录音")
    return templates, names


def recognize_speaker(samples: np.ndarray, sample_rate: int, templates: dict[str, list[np.ndarray]], names: dict[str, str]) -> dict[str, object]:
    features, details = prepare_utterance(samples, sample_rate)
    features = _compact(features)
    scores = {
        sid: min(dtw_distance(features, template) for template in examples)
        for sid, examples in templates.items() if examples
    }
    if not scores:
        raise ValueError("当前没有可用的说话人模板")
    ordered = sorted(scores.items(), key=lambda item: item[1])
    best_id, distance = ordered[0]
    return {
        "speaker_id": best_id,
        "speaker_name": names.get(best_id, best_id),
        "distance": distance,
        "scores": {sid: {"name": names.get(sid, sid), "distance": value} for sid, value in ordered},
        "audio": details,
    }
