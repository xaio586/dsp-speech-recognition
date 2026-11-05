"""Evaluate DTW digit recognition with alternating take-based holdout folds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.recognize import load_digit_templates, prepare_utterance, score_features  # noqa: E402


def _test_files(take: int) -> list[tuple[str, Path]]:
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
    output: list[tuple[str, Path]] = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("content", "")).strip()
            if label not in {str(value) for value in range(10)}:
                continue
            if str(row.get("take", "")).strip() != str(take):
                continue
            relative = Path(*str(row["relative_path"]).split("/"))
            path = (PROJECT_ROOT / relative).resolve()
            if path.is_file():
                output.append((label, path))
    return sorted(output, key=lambda item: str(item[1]))


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    confusion = [[0 for _ in range(10)] for _ in range(10)]
    fold_summaries: list[dict[str, object]] = []

    for fold, (train_take, test_take) in enumerate(((1, 2), (2, 1)), start=1):
        templates = load_digit_templates(PROJECT_ROOT, train_take=train_take)
        test_files = _test_files(test_take)
        if not test_files:
            raise FileNotFoundError(f"metadata.csv 中没有 take{test_take:02d} 数字测试录音")
        fold_correct = 0
        fold_start = time.perf_counter()
        for label, audio_path in test_files:
            audio = load_wav(audio_path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
            features, detail = prepare_utterance(audio.samples, audio.sample_rate)
            started = time.perf_counter()
            scores = score_features(features, templates)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predicted = next(iter(scores))
            correct = predicted == label
            fold_correct += int(correct)
            confusion[int(label)][int(predicted)] += 1
            rows.append(
                {
                    "fold": fold,
                    "train_take": train_take,
                    "test_take": test_take,
                    "file_name": audio_path.name,
                    "true_digit": label,
                    "predicted_digit": predicted,
                    "correct": int(correct),
                    "dtw_distance": f"{scores[predicted]:.8f}",
                    "speech_seconds": f"{detail['speech_seconds']:.4f}",
                    "inference_ms": f"{elapsed_ms:.2f}",
                }
            )
        fold_total = len(test_files)
        fold_summaries.append(
            {
                "fold": fold,
                "train_take": train_take,
                "test_take": test_take,
                "total": fold_total,
                "correct": fold_correct,
                "accuracy": fold_correct / fold_total if fold_total else 0.0,
                "elapsed_seconds": time.perf_counter() - fold_start,
            }
        )

    total = len(rows)
    correct_total = sum(int(row["correct"]) for row in rows)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *range(10)])
        for label, values in enumerate(confusion):
            writer.writerow([label, *values])

    summary: dict[str, object] = {
        "experiment": "实验三：MFCC + DTW 孤立数字识别",
        "dataset": "data/processed/metadata.csv 中的0—9数字录音",
        "total_recordings": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "folds": fold_summaries,
        "split_method": "两折轮换：take01训练/take02测试，再反向；单条录音不会同时进入训练和测试",
        "sample_rate_hz": DEFAULT_CONFIG.sample_rate,
        "feature": "13维MFCC",
        "classifier": "逐类最小归一化DTW距离",
        "prediction_file": predictions_path.name,
        "confusion_matrix_file": confusion_path.name,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiment3",
        help="folder for predictions, confusion matrix, and summary",
    )
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
