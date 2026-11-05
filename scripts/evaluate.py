"""Compare MFCC presets on the fixed take-1/train, take-2/test metadata split.

Run from the repository: python scripts/evaluate.py --workers 4
Only NumPy and SciPy (already used by the core modules) are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.dtw import recognize_digit  # noqa: E402
from dsp_speech.features import extract_mfcc  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.preprocess import analyze_time_domain  # noqa: E402

LABELS = tuple(str(digit) for digit in range(10))
METADATA_FIELDS = ("file_name", "speaker_id", "digit", "take", "fold1_role")
BASE_METADATA_FIELDS = ("file_name", "speaker_id", "take")


@dataclass(frozen=True)
class MFCCConfig:
    """The three parameters exposed by the existing extract_mfcc function."""

    name: str
    n_mfcc: int
    n_filters: int
    n_fft: int


# Each variant changes one parameter from the existing function's defaults.
MFCC_CONFIGS = (
    MFCCConfig("baseline", n_mfcc=13, n_filters=26, n_fft=512),
    MFCCConfig("mfcc_20", n_mfcc=20, n_filters=26, n_fft=512),
    MFCCConfig("mel_40", n_mfcc=13, n_filters=40, n_fft=512),
    MFCCConfig("fft_1024", n_mfcc=13, n_filters=26, n_fft=1024),
)


@dataclass(frozen=True)
class Recording:
    """One metadata row; labels come from metadata, never from filenames."""

    file_name: str
    speaker_id: str
    digit: str
    take: int
    fold1_role: str
    path: Path


def load_recordings(data_dir: Path) -> tuple[Recording, ...]:
    """Validate the fixed split in either project or compact test metadata format."""
    data_dir = data_dir.resolve()
    metadata_path = data_dir / "metadata.csv"
    recordings = []
    seen_paths = set()
    seen_takes = set()
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(BASE_METADATA_FIELDS) - fields
        if not ({"digit", "content"} & fields):
            missing.add("digit (or content)")
        if missing:
            raise ValueError(f"metadata.csv missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            values = {key: (value or "").strip() for key, value in row.items() if key}
            # The project metadata also contains 15 name recordings; this evaluator
            # intentionally uses only the ten spoken-digit classes.
            if values.get("content", "").lower() == "name":
                continue
            required_values = ("file_name", "speaker_id", "take")
            if any(not values.get(field) for field in required_values):
                raise ValueError(f"metadata.csv row {line}: required value is empty")
            digit = values.get("digit") or values.get("content", "")
            if digit not in LABELS:
                raise ValueError(f"metadata.csv row {line}: invalid digit {digit!r}")
            take = int(values["take"])
            role = values.get("fold1_role") or {1: "train", 2: "test"}.get(take, "")
            if (role, take) not in {("train", 1), ("test", 2)}:
                raise ValueError(
                    f"metadata.csv row {line}: fold1_role={role!r}, take={take}; "
                    "expected train/take=1 or test/take=2"
                )
            file_name, speaker = values["file_name"], values["speaker_id"]
            if any(
                name in {".", ".."} or "/" in name or "\\" in name for name in (file_name, speaker)
            ):
                raise ValueError(f"metadata.csv row {line}: expected plain file/speaker names")
            relative_path = values.get("relative_path", "")
            candidates = []
            if relative_path:
                supplied_path = Path(relative_path)
                if supplied_path.is_absolute():
                    candidates.append(supplied_path)
                else:
                    if relative_path.replace("\\", "/").startswith("data/processed/"):
                        candidates.append(PROJECT_ROOT / supplied_path)
                    candidates.extend((data_dir / supplied_path, PROJECT_ROOT / supplied_path))
            else:
                candidates.extend(
                    (
                        data_dir / "digits" / speaker / file_name,
                        data_dir / speaker / "digits" / file_name,
                    )
                )
            path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if path is None:
                path = candidates[0].resolve()
            if not any(path.is_relative_to(root) for root in {data_dir, PROJECT_ROOT.resolve()}):
                raise ValueError(f"metadata.csv row {line}: recording path escapes project/data directory")
            if path.suffix.lower() != ".wav":
                raise ValueError(f"metadata.csv row {line}: invalid WAV path {path}")
            if not path.is_file():
                raise FileNotFoundError(f"metadata.csv row {line}: missing recording {path}")
            take_key = (speaker, digit, take)
            if path in seen_paths or take_key in seen_takes:
                raise ValueError(f"metadata.csv row {line}: duplicate recording {file_name}")
            seen_paths.add(path)
            seen_takes.add(take_key)
            recordings.append(Recording(file_name, speaker, digit, take, role, path))
    train_labels = {row.digit for row in recordings if row.fold1_role == "train"}
    if train_labels != set(LABELS):
        raise ValueError(f"Training templates missing digits: {sorted(set(LABELS) - train_labels)}")
    if not any(row.fold1_role == "test" for row in recordings):
        raise ValueError("metadata.csv contains no test recordings")
    return tuple(recordings)


def prepare_audio(
    recordings: tuple[Recording, ...], data_dir: Path
) -> tuple[list[np.ndarray], list[dict]]:
    """Run the existing experiment-1 preprocessing once for every recording."""
    waveforms, manifest = [], []
    for row in recordings:
        audio = load_wav(row.path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
        if not audio.samples.size or not np.isfinite(audio.samples).all():
            raise ValueError(f"Empty or non-finite audio: {row.file_name}")
        result = analyze_time_domain(audio.samples, audio.sample_rate)
        start, end = result["endpoints"]
        if not 0 <= start < end <= len(audio.samples):
            # Never drop a difficult recording or silently replace its detected endpoints.
            raise ValueError(
                f"Endpoint detection failed: {row.file_name}, endpoints=({start}, {end})"
            )
        waveforms.append(result["samples"][start:end])
        manifest.append(
            {
                "file_name": row.file_name,
                "speaker_id": row.speaker_id,
                "digit": row.digit,
                "take": row.take,
                "fold1_role": row.fold1_role,
                "relative_path": row.path.relative_to(data_dir.resolve()).as_posix(),
                "wav_sha256": hashlib.sha256(row.path.read_bytes()).hexdigest(),
                "sample_rate": audio.sample_rate,
                "original_channels": audio.original_channels,
                "original_dtype": audio.original_dtype,
                "original_sample_count": len(audio.samples),
                "start_sample": start,
                "end_sample": end,
                "speech_sample_count": end - start,
            }
        )
    return waveforms, manifest


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write portable, explicit CSV headers using only the standard library."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
    configs: tuple[MFCCConfig, ...] = MFCC_CONFIGS,
    workers: int = 1,
) -> list[dict]:
    """Rebuild templates for each preset and score exactly the same test rows."""
    started = perf_counter()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not configs or len({config.name for config in configs}) != len(configs):
        raise ValueError("Select at least one configuration, with unique names")
    frame_length = round(DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.frame_ms / 1000)
    for config in configs:
        if (
            not config.name.isidentifier()
            or not 1 <= config.n_mfcc <= config.n_filters
            or config.n_fft < frame_length
        ):
            raise ValueError(f"Invalid MFCC configuration: {config}")

    recordings = load_recordings(data_dir)
    waveforms, manifest = prepare_audio(recordings, data_dir)
    # These index tuples are created once, outside the parameter-comparison loop.
    train_indices = tuple(i for i, row in enumerate(recordings) if row.fold1_role == "train")
    test_indices = tuple(i for i, row in enumerate(recordings) if row.fold1_role == "test")
    split_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(f"Fixed split: {len(train_indices)} train, {len(test_indices)} test", flush=True)

    predictions, summaries, matrices = [], [], {}
    pool_context = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else nullcontext()
    )
    with pool_context as pool:
        for config in configs:
            parameters = asdict(config)
            parameters.pop("name")
            print(f"{config.name}: {parameters}", flush=True)
            features = [
                extract_mfcc(samples, DEFAULT_CONFIG.sample_rate, **parameters)
                for samples in waveforms
            ]
            if any(
                matrix.ndim != 2
                or matrix.shape[1] != config.n_mfcc
                or not len(matrix)
                or not np.isfinite(matrix).all()
                for matrix in features
            ):
                raise ValueError(f"Invalid MFCC output for {config.name}")
            templates = {label: [] for label in LABELS}
            for index in train_indices:
                templates[recordings[index].digit].append(features[index])
            queries = [features[index] for index in test_indices]
            # Multiprocessing calls the unchanged recognizer; map preserves test order.
            results = (
                pool.map(recognize_digit, queries, repeat(templates))
                if pool is not None
                else map(recognize_digit, queries, repeat(templates))
            )
            confusion = np.zeros((10, 10), dtype=np.int64)
            for count, (index, (predicted, distance)) in enumerate(
                zip(test_indices, results, strict=True), start=1
            ):
                row = recordings[index]
                if predicted not in LABELS or not math.isfinite(distance):
                    raise ValueError(f"Invalid recognition result for {row.file_name}")
                confusion[int(row.digit), int(predicted)] += 1
                predictions.append(
                    {
                        "config": config.name,
                        **parameters,
                        "file_name": row.file_name,
                        "speaker_id": row.speaker_id,
                        "take": row.take,
                        "true_label": row.digit,
                        "predicted_label": predicted,
                        "correct": int(predicted == row.digit),
                        "dtw_distance": float(distance),
                    }
                )
                if count % 10 == 0 or count == len(test_indices):
                    print(f"  recognized {count}/{len(test_indices)}", flush=True)
            correct = int(np.trace(confusion))
            summaries.append(
                {
                    "config": config.name,
                    **parameters,
                    "train_count": len(train_indices),
                    "test_count": len(test_indices),
                    "correct": correct,
                    "accuracy": correct / len(test_indices),
                    "split_sha256": split_sha256,
                }
            )
            matrices[config.name] = confusion
            print(f"  accuracy: {correct}/{len(test_indices)} = {summaries[-1]['accuracy']:.2%}")

    # Publish metrics only after every selected configuration completes successfully.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "split_manifest.csv", list(manifest[0]), manifest)
    write_csv(output_dir / "predictions.csv", list(predictions[0]), predictions)
    write_csv(output_dir / "summary.csv", list(summaries[0]), summaries)
    for name, matrix in matrices.items():
        rows = [
            {"true_label": label, **dict(zip(LABELS, map(int, matrix[index]), strict=True))}
            for index, label in enumerate(LABELS)
        ]
        write_csv(output_dir / f"confusion_matrix_{name}.csv", ["true_label", *LABELS], rows)
    run_manifest = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "workers": workers,
        "metadata_sha256": hashlib.sha256((data_dir / "metadata.csv").read_bytes()).hexdigest(),
        "split_sha256": split_sha256,
        "split_rule": {
            "train": {"fold1_role": "train", "take": 1},
            "test": {"fold1_role": "test", "take": 2},
        },
        "counts_by_digit": {
            role: dict(sorted(Counter(r.digit for r in recordings if r.fold1_role == role).items()))
            for role in ("train", "test")
        },
        "shared_config": asdict(DEFAULT_CONFIG),
        "preprocessing": "load_wav -> analyze_time_domain -> samples[start:end] -> extract_mfcc",
        "endpoint_failures": 0,
        "templates": "All train MFCC sequences, grouped by digit; no averaging",
        "core_source_sha256": {
            name: hashlib.sha256(
                (PROJECT_ROOT / "src" / "dsp_speech" / name).read_bytes()
            ).hexdigest()
            for name in ("config.py", "io_audio.py", "preprocess.py", "features.py", "dtw.py")
        },
        "confusion_matrix_axes": {"rows": "true_label", "columns": "predicted_label"},
        "accuracy_unit": "fraction in [0, 1]",
        "results": summaries,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 固定划分 MFCC 参数评估",
        "",
        "运行命令（在仓库目录）：`python scripts/evaluate.py --workers 4`。",
        "",
        "训练：fold1_role=train 且 take=1；测试：fold1_role=test 且 take=2。",
        f"训练 {len(train_indices)} 条，测试 {len(test_indices)} 条；所有配置使用同一划分。",
        "预处理复用 analyze_time_domain，截取检测端点后提取 MFCC；没有随机划分或跳过录音。",
        "每个数字保留全部训练模板，识别复用 recognize_digit。",
        "",
        "| 配置 | n_mfcc | n_filters | n_fft | 正确/测试 | 准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['config']} | {row['n_mfcc']} | {row['n_filters']} | {row['n_fft']} | "
            f"{row['correct']}/{row['test_count']} | {row['accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "predictions.csv 保存逐条真实标签、预测标签、正确标记和 DTW 距离；",
            "summary.csv 保存参数及总体准确率（0–1）。",
            "confusion_matrix_*.csv 的行为真实标签、列为预测标签。",
            "split_manifest.csv 保存录音划分、音频校验值及端点。",
            "run_manifest.json 保存环境、共享参数和源码校验值。",
            "",
            "这是同一批说话人第 1 遍训练、第 2 遍测试的已见说话人实验。",
            "参数优劣仅描述这份固定测试集；未评估未见说话人，也没有独立验证集或第二折实验。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Metrics saved to {output_dir.resolve()}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "metrics")
    parser.add_argument("--workers", type=int, default=1, help="recognition processes (default: 1)")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[config.name for config in MFCC_CONFIGS],
        help="presets to compare (default: all four)",
    )
    args = parser.parse_args()
    selected = tuple(
        config for config in MFCC_CONFIGS if not args.configs or config.name in args.configs
    )
    try:
        run_evaluation(args.data_dir, args.output, selected, args.workers)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
