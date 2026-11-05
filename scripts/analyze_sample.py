"""Draw waveform, short-time features, and detected endpoints for one WAV file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.preprocess import analyze_time_domain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    args = parser.parse_args()

    audio = load_wav(args.wav_path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
    result = analyze_time_domain(audio.samples, audio.sample_rate)
    cleaned = result["samples"]
    energy = result["energy"]
    magnitude = result["magnitude"]
    zcr = result["zcr"]
    start, end = result["endpoints"]
    assert isinstance(cleaned, np.ndarray)
    assert isinstance(energy, np.ndarray)
    assert isinstance(magnitude, np.ndarray)
    assert isinstance(zcr, np.ndarray)

    time = np.arange(len(cleaned)) / audio.sample_rate
    frame_time = np.arange(len(energy)) * DEFAULT_CONFIG.hop_ms / 1000
    figure, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, cleaned, color="#1F4E78", linewidth=0.75)
    if end > start:
        axes[0].axvspan(
            start / audio.sample_rate,
            end / audio.sample_rate,
            color="#F4B183",
            alpha=0.3,
        )
    axes[0].set_ylabel("Amplitude")
    endpoint_label = (
        f"{start / audio.sample_rate:.2f}–{end / audio.sample_rate:.2f} s"
        if end > start
        else "no speech detected"
    )
    axes[0].set_title(f"{args.wav_path.name} | endpoint {endpoint_label}")
    axes[1].plot(frame_time, energy, color="#2F7D32", linewidth=1.0)
    axes[1].set_ylabel("Energy")
    axes[2].plot(frame_time, magnitude, color="#4472C4", linewidth=1.0)
    axes[2].set_ylabel("Magnitude")
    axes[3].plot(frame_time, zcr, color="#C55A11", linewidth=1.0)
    axes[3].set_ylabel("ZCR")
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.wav_path.stem}_analysis.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(f"duration: {audio.duration_seconds:.3f} s")
    print(
        f"frames: {len(energy)}, frame_length: "
        f"{round(audio.sample_rate * DEFAULT_CONFIG.frame_ms / 1000)}"
    )
    print(f"endpoint_samples: ({start}, {end})")
    print(f"figure: {destination}")


if __name__ == "__main__":
    main()
