"""Pre-emphasize and analyze the group's raw M4A recordings for experiment 2.

Default input and output paths are resolved relative to the repository, so the
script works from any working directory after the repository is cloned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.preprocess import apply_window, frame_signal, pre_emphasis  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "week03_b_preprocess"
N_FFT = 512
MIN_DURATION_SECONDS = 0.3
SILENCE_PEAK_THRESHOLD = 1e-5
CLIPPING_THRESHOLD = 0.999


def decode_m4a(path: Path, ffmpeg: str) -> np.ndarray:
    """Decode M4A to 16 kHz mono float32 samples using the project's ffmpeg."""
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_CONFIG.sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError("decoded audio is empty")
    return samples


def preprocess_waveform(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the shared-config magnitude spectrum and basic audio QC values."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty mono waveform")
    if not np.isfinite(values).all():
        raise ValueError("samples must contain only finite values")
    if sample_rate != DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"expected {DEFAULT_CONFIG.sample_rate} Hz, received {sample_rate} Hz")

    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    duration = values.size / sample_rate
    clipped_ratio = float(np.mean(np.abs(values) >= CLIPPING_THRESHOLD))

    emphasized = pre_emphasis(values, DEFAULT_CONFIG.pre_emphasis)
    frames = frame_signal(
        emphasized,
        sample_rate,
        DEFAULT_CONFIG.frame_ms,
        DEFAULT_CONFIG.hop_ms,
    )
    windowed = apply_window(frames, DEFAULT_CONFIG.window)
    spectrum = (np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)) ** 2 / N_FFT).astype(np.float32)
    quality = {
        "duration_seconds": float(duration),
        "peak": peak,
        "rms": rms,
        "clipped_ratio": clipped_ratio,
    }
    return spectrum, quality


def _check_quality(quality: dict[str, float]) -> list[str]:
    warnings = []
    if quality["duration_seconds"] < MIN_DURATION_SECONDS:
        warnings.append("too_short")
    if quality["peak"] < SILENCE_PEAK_THRESHOLD:
        warnings.append("silent_or_near_silent")
    if quality["clipped_ratio"] > 0.001:
        warnings.append("possible_clipping")
    return warnings


def run(input_root: Path, output_root: Path, expected_count: int | None = 115) -> dict[str, object]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_root}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be installed and available on PATH")

    sources = sorted(p for p in input_root.rglob("*.m4a") if p.is_file())
    if not sources:
        raise FileNotFoundError(f"no .m4a recordings found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(input_root)
        output = (output_root / relative).with_suffix(".npy")
        record: dict[str, object] = {
            "input_file": relative.as_posix(),
            "status": "ok",
        }
        try:
            waveform = decode_m4a(source, ffmpeg)
            spectrum, quality = preprocess_waveform(waveform, DEFAULT_CONFIG.sample_rate)
            record.update(quality)
            warnings = _check_quality(quality)
            blocking_warnings = [
                warning for warning in warnings if warning in {"too_short", "silent_or_near_silent"}
            ]
            if blocking_warnings:
                record["status"] = "abnormal"
                record["warnings"] = blocking_warnings
                records.append(record)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, spectrum, allow_pickle=False)
            record["output_file"] = (
                output.relative_to(PROJECT_ROOT).as_posix()
                if output.is_relative_to(PROJECT_ROOT)
                else str(output)
            )
            record["spectrum_shape"] = list(spectrum.shape)
            record["dtype"] = str(spectrum.dtype)
            if warnings:
                record["status"] = "warning"
                record["warnings"] = warnings
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            record["status"] = "read_fail"
            if isinstance(error, subprocess.CalledProcessError):
                detail = error.stderr.decode("utf-8", errors="replace").strip()
                record["message"] = detail or "ffmpeg could not decode this file"
            else:
                record["message"] = str(error)
        records.append(record)

    counts = Counter(str(row["status"]) for row in records)
    mismatch = expected_count is not None and len(sources) != expected_count
    manifest: dict[str, object] = {
        "input_directory": input_root.relative_to(PROJECT_ROOT).as_posix()
        if input_root.is_relative_to(PROJECT_ROOT)
        else str(input_root),
        "output_directory": output_root.relative_to(PROJECT_ROOT).as_posix()
        if output_root.is_relative_to(PROJECT_ROOT)
        else str(output_root),
        "input_extension": ".m4a",
        "input_count": len(sources),
        "expected_count": expected_count,
        "input_count_matches_expected": not mismatch,
        "status_counts": dict(counts),
        "spectra_saved": sum("output_file" in row for row in records),
        "sample_rate_hz": DEFAULT_CONFIG.sample_rate,
        "pre_emphasis_coefficient": DEFAULT_CONFIG.pre_emphasis,
        "frame_length_ms": DEFAULT_CONFIG.frame_ms,
        "frame_hop_ms": DEFAULT_CONFIG.hop_ms,
        "window": DEFAULT_CONFIG.window,
        "n_fft": N_FFT,
        "spectrum_type": "one-sided linear power spectrum; not playable audio",
        "qc_note": (
            "Out-of-range decoded peaks are warnings and retained; "
            "short or silent files are skipped."
        ),
    }
    (output_root / "log.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw M4A directory")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="generated NPY/log directory"
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=115,
        help="warn in the manifest if the number of M4A files differs; use 0 to disable",
    )
    args = parser.parse_args()
    expected = None if args.expected_count == 0 else args.expected_count
    manifest = run(args.input, args.output, expected)
    if manifest["status_counts"].get("read_fail", 0):
        raise SystemExit(1)
    if not manifest["input_count_matches_expected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
