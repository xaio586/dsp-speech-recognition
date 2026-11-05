"""Check split integrity and consistency of real evaluation outputs."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.evaluate import METADATA_FIELDS, MFCC_CONFIGS, load_recordings, run_evaluation


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_metadata(data_dir: Path, rows: list[dict]) -> None:
    with (data_dir / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Small synthetic fixtures are used only in tmp_path, never as experiment data."""
    data_dir = tmp_path / "processed"
    speaker_dir = data_dir / "digits" / "spk01"
    speaker_dir.mkdir(parents=True)
    rows = []
    for digit in range(10):
        for take, role in ((1, "train"), (2, "test")):
            file_name = f"spk01_digit_{digit}_take{take:02}.wav"
            signal = np.zeros(8_000, dtype=np.int16)
            time = np.arange(4_000) / 16_000
            signal[2_000:6_000] = (
                16_000 * np.sin(2 * np.pi * (200 + 80 * digit + take) * time)
            ).astype(np.int16)
            wavfile.write(speaker_dir / file_name, 16_000, signal)
            rows.append(
                dict(
                    zip(METADATA_FIELDS, (file_name, "spk01", str(digit), take, role), strict=True)
                )
            )
    write_metadata(data_dir, rows)
    return data_dir


@pytest.mark.parametrize("role,take", [("train", 2), ("test", 1), ("validation", 1)])
def test_rejects_conflicting_or_unknown_split(dataset: Path, role: str, take: int) -> None:
    rows = read_csv(dataset / "metadata.csv")
    rows[0].update(fold1_role=role, take=take)
    write_metadata(dataset, rows)
    with pytest.raises(ValueError, match="expected train/take=1 or test/take=2"):
        load_recordings(dataset)


def test_rejects_duplicate_recordings(dataset: Path) -> None:
    rows = read_csv(dataset / "metadata.csv")
    write_metadata(dataset, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate recording"):
        load_recordings(dataset)


def test_missing_recording_fails_without_metrics(dataset: Path, tmp_path: Path) -> None:
    missing = next(dataset.rglob("*.wav"))
    missing.unlink()
    output = tmp_path / "metrics"
    with pytest.raises(FileNotFoundError, match="missing recording"):
        run_evaluation(dataset, output, MFCC_CONFIGS[:1])
    assert not output.exists()


def test_endpoint_failure_does_not_silently_drop_sample(dataset: Path, tmp_path: Path) -> None:
    silent = next(dataset.rglob("*.wav"))
    wavfile.write(silent, 16_000, np.zeros(8_000, dtype=np.int16))
    output = tmp_path / "metrics"
    with pytest.raises(ValueError, match="Endpoint detection failed"):
        run_evaluation(dataset, output, MFCC_CONFIGS[:1])
    assert not output.exists()


def test_fixed_split_and_metrics_agree_in_serial_and_parallel(
    dataset: Path, tmp_path: Path
) -> None:
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    for workers, output in zip((1, 2), outputs, strict=True):
        run_evaluation(dataset, output, MFCC_CONFIGS[:2], workers=workers)
        predictions = read_csv(output / "predictions.csv")
        summaries = read_csv(output / "summary.csv")
        split = read_csv(output / "split_manifest.csv")
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        train_files = {row["file_name"] for row in split if row["fold1_role"] == "train"}
        test_files = [row["file_name"] for row in split if row["fold1_role"] == "test"]
        assert len(train_files) == len(test_files) == 10
        assert train_files.isdisjoint(test_files)
        assert len(predictions) == 20
        assert manifest["status"] == "complete"
        assert len({row["split_sha256"] for row in summaries}) == 1
        for summary in summaries:
            group = [row for row in predictions if row["config"] == summary["config"]]
            assert [row["file_name"] for row in group] == test_files
            assert all(row["take"] == "2" for row in group)
            correct = sum(row["true_label"] == row["predicted_label"] for row in group)
            assert int(summary["correct"]) == correct
            assert float(summary["accuracy"]) == correct / len(group)
            rows = read_csv(output / f"confusion_matrix_{summary['config']}.csv")
            matrix = np.array([[int(row[str(i)]) for i in range(10)] for row in rows])
            assert matrix.shape == (10, 10)
            assert matrix.sum() == 10
            assert np.trace(matrix) == correct
            for true in range(10):
                for predicted in range(10):
                    assert matrix[true, predicted] == sum(
                        row["true_label"] == str(true) and row["predicted_label"] == str(predicted)
                        for row in group
                    )
    for file_name in ("predictions.csv", "summary.csv", "split_manifest.csv"):
        assert (outputs[0] / file_name).read_bytes() == (outputs[1] / file_name).read_bytes()
