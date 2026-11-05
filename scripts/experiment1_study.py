"""Run reproducible parameter, ablation, and window studies for experiment 1.

Metrics describe endpoint availability and sensitivity to the repository's
baseline detector. They are not endpoint accuracy because recordings lack
sample-level human ground-truth annotations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.preprocess import analyze_time_domain  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "outputs" / "experiment1_study"

for font_path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        break
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "savefig.facecolor": "white",
    }
)


def _digit_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*_digit_[0-9]_take*.wav"))
    if not files:
        raise FileNotFoundError(f"No digit WAV recordings under {data_dir}")
    return files


def _summary_row(kind: str, name: str, records: list[dict]) -> dict[str, object]:
    good = [row for row in records if row["detected"]]
    shifts = [row["boundary_shift_ms"] for row in good if row["boundary_shift_ms"] is not None]
    durations = [row["duration_ms"] for row in good]
    return {
        "study": kind,
        "configuration": name,
        "recordings": len(records),
        "detected": len(good),
        "nonempty_endpoint_rate_pct": 100 * len(good) / len(records) if records else 0.0,
        "mean_duration_ms": float(np.mean(durations)) if durations else 0.0,
        "duration_std_ms": float(np.std(durations)) if durations else 0.0,
        "mean_boundary_shift_vs_baseline_ms": float(np.mean(shifts)) if shifts else 0.0,
        "p95_boundary_shift_vs_baseline_ms": float(np.percentile(shifts, 95)) if shifts else 0.0,
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in records]))
        if records
        else 0.0,
    }


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> dict[str, object]:
    data_dir, out_dir = data_dir.resolve(), out_dir.resolve()
    paths = _digit_files(data_dir)
    samples: list[tuple[Path, str, np.ndarray, int]] = []
    for path in paths:
        audio = load_wav(path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
        samples.append((path, path.stem, audio.samples, audio.sample_rate))

    baseline_results = {}
    baseline_records = []
    for path, key, x, fs in samples:
        t0 = perf_counter()
        result = analyze_time_domain(x, fs, DEFAULT_CONFIG)
        elapsed = (perf_counter() - t0) * 1000
        start, end = result["endpoints"]
        detected = end > start
        baseline_results[key] = (start, end)
        baseline_records.append(
            {
                "study": "baseline",
                "configuration": "default",
                "file": path.relative_to(data_dir).as_posix(),
                "detected": detected,
                "start_ms": start * 1000 / fs,
                "end_ms": end * 1000 / fs,
                "duration_ms": (end - start) * 1000 / fs if detected else 0.0,
                "boundary_shift_ms": 0.0 if detected else None,
                "runtime_ms": elapsed,
            }
        )

    scenarios: list[tuple[str, str, object]] = []
    for high, low in itertools.product((0.05, 0.10, 0.20), (0.015, 0.025, 0.05)):
        scenarios.append(
            (
                "tuning",
                f"high{high:.3f}_low{low:.3f}",
                replace(DEFAULT_CONFIG, high_energy_ratio=high, low_energy_ratio=low),
            )
        )
    scenarios.extend(
        [
            (
                "ablation",
                "no_zcr_extension",
                replace(DEFAULT_CONFIG, min_zcr_threshold=1.01, zcr_noise_multiplier=1.0),
            ),
            (
                "ablation",
                "no_hysteresis",
                replace(DEFAULT_CONFIG, low_energy_ratio=DEFAULT_CONFIG.high_energy_ratio),
            ),
            ("ablation", "no_min_speech_filter", replace(DEFAULT_CONFIG, min_speech_ms=0.0)),
            ("comparison", "rectangular_window", replace(DEFAULT_CONFIG, window="rectangular")),
            ("comparison", "hamming_window", replace(DEFAULT_CONFIG, window="hamming")),
            ("comparison", "hann_window", replace(DEFAULT_CONFIG, window="hann")),
        ]
    )

    all_records = list(baseline_records)
    summaries = [_summary_row("baseline", "default", baseline_records)]
    for kind, name, config in scenarios:
        group = []
        for path, key, x, fs in samples:
            t0 = perf_counter()
            result = analyze_time_domain(x, fs, config)
            elapsed = (perf_counter() - t0) * 1000
            start, end = result["endpoints"]
            detected = end > start
            base_start, base_end = baseline_results[key]
            shift = (
                (abs(start - base_start) + abs(end - base_end)) * 500 / fs
                if detected and base_end > base_start
                else None
            )
            row = {
                "study": kind,
                "configuration": name,
                "file": path.relative_to(data_dir).as_posix(),
                "detected": detected,
                "start_ms": start * 1000 / fs,
                "end_ms": end * 1000 / fs,
                "duration_ms": (end - start) * 1000 / fs if detected else 0.0,
                "boundary_shift_ms": shift,
                "runtime_ms": elapsed,
            }
            group.append(row)
            all_records.append(row)
        summaries.append(_summary_row(kind, name, group))

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "study",
        "configuration",
        "file",
        "detected",
        "start_ms",
        "end_ms",
        "duration_ms",
        "boundary_shift_ms",
        "runtime_ms",
    ]
    with (out_dir / "per_recording_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    summary_fields = list(summaries[0])
    with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    # Parameter sensitivity: show duration and endpoint movement, not a misleading accuracy score.
    grid = [row for row in summaries if row["study"] == "tuning"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.suptitle(
        "实验一阈值参数敏感性分析（100条数字录音）", fontsize=17, fontweight="bold", color="#18324B"
    )
    for ax, field, title, cmap, unit in [
        (axes[0], "mean_duration_ms", "平均检测区间时长", "YlGnBu", "毫秒"),
        (axes[1], "mean_boundary_shift_vs_baseline_ms", "相对基准的平均端点变化", "OrRd", "毫秒"),
    ]:
        matrix = np.zeros((3, 3))
        for row in grid:
            high = float(row["configuration"].split("_low")[0].replace("high", ""))
            low = float(row["configuration"].split("_low")[1])
            matrix[(0.05, 0.10, 0.20).index(high), (0.015, 0.025, 0.05).index(low)] = float(
                row[field]
            )
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(3), ["0.015", "0.025", "0.050"])
        ax.set_yticks(range(3), ["0.05", "0.10", "0.20"])
        ax.set_xlabel("低能量阈值比例")
        ax.set_ylabel("高能量阈值比例")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="#172B3A",
                    fontsize=11,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, shrink=0.82, label=unit)
    fig.text(
        0.5,
        0.025,
        "9组设置的非空输出比例均为100%；该比例不代表端点检测准确率。",
        ha="center",
        fontsize=9,
        color="#596B78",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(out_dir / "threshold_tuning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline_duration = float(summaries[0]["mean_duration_ms"])
    displays = {
        "no_zcr_extension": "去掉过零率扩展",
        "no_hysteresis": "取消高低门限差异",
        "no_min_speech_filter": "去掉最短语音筛选",
        "rectangular_window": "矩形窗",
        "hamming_window": "海明窗（基准）",
        "hann_window": "汉宁窗",
    }
    for kind, filename, title in [
        ("ablation", "ablation_comparison.png", "实验一消融实验"),
        ("comparison", "window_comparison.png", "实验一窗函数比较"),
    ]:
        rows = [row for row in summaries if row["study"] == kind]
        labels = [displays[str(row["configuration"])] for row in rows]
        duration_delta = [float(row["mean_duration_ms"]) - baseline_duration for row in rows]
        boundary_shift = [float(row["mean_boundary_shift_vs_baseline_ms"]) for row in rows]
        y = np.arange(len(rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, constrained_layout=True)
        fig.suptitle(title + "（100条数字录音）", fontsize=17, fontweight="bold", color="#18324B")
        for ax, values, heading, xlabel, color in [
            (axes[0], duration_delta, "区间时长变化", "相对基准变化（毫秒）", "#3B7A78"),
            (axes[1], boundary_shift, "端点位置变化", "相对基准平均变化（毫秒）", "#D17A42"),
        ]:
            bars = ax.barh(y, values, color=color, height=0.58, zorder=3)
            ax.axvline(0, color="#607080", linewidth=1, zorder=2)
            ax.set_title(heading)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", color="#DCE3E8", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            pad = max(max(abs(v) for v in values) * 0.04, 2)
            ax.set_xlim(min(min(values) - pad * 3, 0), max(max(values) + pad * 5, pad * 5))
            for bar, value in zip(bars, values, strict=True):
                x_text = value + pad
                ax.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.1f}",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="#243746",
                )
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    manifest = {
        "study": "Experiment 1 endpoint detection parameter/ablation/window comparisons",
        "recording_count": len(samples),
        "digit_recording_count": len(samples),
        "sample_rate_hz": DEFAULT_CONFIG.sample_rate,
        "baseline": {
            "frame_ms": DEFAULT_CONFIG.frame_ms,
            "hop_ms": DEFAULT_CONFIG.hop_ms,
            "window": DEFAULT_CONFIG.window,
            "high_energy_ratio": DEFAULT_CONFIG.high_energy_ratio,
            "low_energy_ratio": DEFAULT_CONFIG.low_energy_ratio,
            "zcr_consecutive_frames": DEFAULT_CONFIG.zcr_consecutive_frames,
        },
        "metrics_note": (
            "No sample-level human ground truth is available. Non-empty endpoint output rate "
            "only means the detector returned start<end; boundary shift is absolute change "
            "vs baseline, not accuracy."
        ),
        "summary_rows": summaries,
        "outputs": [
            "summary_metrics.csv",
            "per_recording_metrics.csv",
            "threshold_tuning.png",
            "ablation_comparison.png",
            "window_comparison.png",
        ],
    }
    (out_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"recordings": len(samples), "summary": summaries, "output": str(out_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
