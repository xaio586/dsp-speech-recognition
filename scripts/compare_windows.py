"""Save the window-shape and single-frame comparisons from member C's delivery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.preprocess import (  # noqa: E402
    apply_window,
    frame_signal,
    normalize_amplitude,
    remove_dc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    parser.add_argument(
        "--frame", type=int, default=None, help="frame index; default selects loudest"
    )
    args = parser.parse_args()

    audio = load_wav(args.wav_path, target_sample_rate=DEFAULT_CONFIG.sample_rate)
    samples = normalize_amplitude(remove_dc(audio.samples), DEFAULT_CONFIG.normalize_peak)
    frames = frame_signal(
        samples,
        audio.sample_rate,
        DEFAULT_CONFIG.frame_ms,
        DEFAULT_CONFIG.hop_ms,
    )
    window_length = frames.shape[1]
    names = ("rectangular", "hamming", "hann")
    windows = {name: apply_window(np.ones((1, window_length)), name)[0] for name in names}

    args.output.mkdir(parents=True, exist_ok=True)
    shape_path = args.output / f"{args.wav_path.stem}_window_shapes.png"
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name, weights in windows.items():
        axis.plot(weights, label=name)
    axis.set_title(f"Window comparison ({DEFAULT_CONFIG.frame_ms:g} ms)")
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(shape_path, dpi=200)
    plt.close(figure)

    energy = np.sum(frames * frames, axis=1)
    frame_index = args.frame if args.frame is not None else int(np.argmax(energy))
    if not 0 <= frame_index < len(frames):
        raise ValueError(f"frame index must be in [0, {len(frames) - 1}]")
    frame = frames[frame_index]
    time_ms = np.arange(window_length) * 1000 / audio.sample_rate
    effect_path = args.output / f"{args.wav_path.stem}_window_effects.png"
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(time_ms, frame, label="raw frame", alpha=0.55, linestyle="--")
    for name in names:
        axis.plot(time_ms, frame * windows[name], label=name)
    axis.set_title(f"Window effect on frame {frame_index}")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(effect_path, dpi=200)
    plt.close(figure)
    print(f"window_shapes: {shape_path}")
    print(f"window_effects: {effect_path}")


if __name__ == "__main__":
    main()
