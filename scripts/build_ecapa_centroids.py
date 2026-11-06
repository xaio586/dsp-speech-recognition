"""Build ECAPA-TDNN speaker centroids from the training recordings (ONNX only).

For each enrolled speaker, embeddings of all their training WAVs (16 kHz,
as stored) are averaged and the mean vector is L2-normalized into the
enrollment centroid saved to data/speaker_recognition/ecapa_centroids.npz.

Only onnxruntime + numpy are needed — no PyTorch — so this script also runs
in the hosted environment. The encoder is the ONNX export of SpeechBrain
spkrec-ecapa-voxceleb (data/speaker_recognition/ecapa_encoder.onnx).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.speaker_ecapa import load_ecapa_recognizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "speaker_recognition" / "ecapa_centroids.npz"),
        help="输出 npz 路径（默认覆盖线上使用的质心文件）",
    )
    parser.add_argument(
        "--cache",
        default=str(ROOT / "tmp_ecapa_onnx_train"),
        help="逐文件嵌入缓存目录，中断后可续跑",
    )
    args = parser.parse_args()

    bundle = load_ecapa_recognizer(ROOT)
    if bundle is None:
        raise SystemExit("缺少 ecapa_encoder.onnx / metadata.csv，无法构建质心")
    recognizer, names = bundle

    manifest = ROOT / "data" / "speaker_recognition" / "metadata.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    print(f"训练录音 {len(train_rows)} 条，开始提取嵌入……")

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, list[np.ndarray]] = {}
    for index, row in enumerate(train_rows, 1):
        cache_file = cache_dir / (row["file_name"].replace("/", "__") + ".npy")
        if cache_file.exists():
            embedding = np.load(cache_file)
        else:
            audio = load_wav(ROOT / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
            embedding = recognizer.embed(audio.samples)
            np.save(cache_file, embedding)
        embeddings.setdefault(row["speaker_id"], []).append(embedding)
        if index % 40 == 0:
            print(f"  进度 {index}/{len(train_rows)}", flush=True)

    centroids = {}
    for sid, vectors in sorted(embeddings.items()):
        mean = np.mean(np.stack(vectors), axis=0)
        centroids[sid] = mean / (float(np.linalg.norm(mean)) + 1e-9)
        print(f"  {sid} {names.get(sid, sid)}: {len(vectors)} 条训练嵌入")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **centroids)
    print(f"质心已写入 {out_path}（{len(centroids)} 人，{len(next(iter(centroids.values())))} 维）")


if __name__ == "__main__":
    main()
