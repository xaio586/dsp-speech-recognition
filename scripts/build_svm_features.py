"""Build data/speaker_recognition/svm_features.npz from the training WAVs.

The deployed service only ships metadata.csv + this small archive (one
102-dim utterance-level vector per training recording), so the Render build
does not need the raw audio to fit the SVM at startup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp_speech.speaker_recognition import build_training_vectors  # noqa: E402


def main() -> None:
    vectors, names = build_training_vectors(ROOT, augment=True)
    payload: dict[str, np.ndarray] = {}
    for sid, items in sorted(vectors.items()):
        for index, vector in enumerate(items):
            payload[f"{sid}_{index:02d}"] = vector.astype(np.float32)
    out = ROOT / "data" / "speaker_recognition" / "svm_features.npz"
    np.savez_compressed(out, **payload)
    counts = {sid: len(items) for sid, items in sorted(vectors.items())}
    dim = len(next(iter(next(iter(vectors.values())))))
    print(f"已写入 {out}：{len(payload)} 条训练向量，维度 {dim}，分布 {counts}")


if __name__ == "__main__":
    main()
