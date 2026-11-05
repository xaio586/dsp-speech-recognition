"""Print WAV information and save a waveform plot."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a WAV file and draw its waveform")
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_CONFIG.sample_rate)
    args = parser.parse_args()

    audio = load_wav(args.wav_path, target_sample_rate=args.sample_rate)
    print(f"file: {args.wav_path}")
    print(f"sample_rate: {audio.sample_rate} Hz")
    print(f"original_channels: {audio.original_channels}")
    print(f"original_dtype: {audio.original_dtype}")
    print(f"samples: {len(audio.samples)}")
    print(f"duration: {audio.duration_seconds:.3f} s")
    print(f"peak: {float(np.max(np.abs(audio.samples))):.4f}")

    time_axis = np.arange(len(audio.samples)) / audio.sample_rate
    figure, axis = plt.subplots(figsize=(10, 3.8))
    axis.plot(time_axis, audio.samples, linewidth=0.8, color="#1F4E78")
    axis.set_title(args.wav_path.name)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.wav_path.stem}_waveform.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"waveform: {output_path}")


if __name__ == "__main__":
    main()
