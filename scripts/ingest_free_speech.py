"""Ingest the 5-minute free-speech recordings into the speaker training set.

Each long M4A is converted to 16 kHz mono WAV, then segmented into speech
chunks of roughly 4-8 s using an energy-based activity detector. Chunks are
stored under data/speaker_recognition/wav/<speaker>/ as additional TRAIN rows
(prompt_id=free01) appended to metadata.csv. The 27 held-out test recordings
are never touched.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path("C:/Users/Lenovo/Downloads/dsp/dsp-speech-recognition")
sys.path.insert(0, str(ROOT / "src"))

import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SOURCES = {
    "spk01": ("肖绚聍", Path("D:/KimiData/kimi/tasks/2026-11-05/21-37-58-bdaee609/new_audio/肖绚聍.m4a")),
    "spk02": ("王晗宇", Path("D:/KimiData/kimi/tasks/2026-11-05/21-37-58-bdaee609/new_audio/王晗宇.m4a")),
    "spk03": ("靳博文", Path("D:/KimiData/kimi/tasks/2026-11-05/21-37-58-bdaee609/new_audio/靳博文.m4a")),
    "spk04": ("李欣彤", Path("D:/KimiData/kimi/tasks/2026-11-05/21-37-58-bdaee609/new_audio/李欣彤.m4a")),
    "spk05": ("赵丹阳", Path("D:/KimiData/kimi/tasks/2026-11-05/21-37-58-bdaee609/new_audio/赵丹阳.m4a")),
}
SR = 16000
CHUNK_MIN_S, CHUNK_MAX_S = 3.0, 8.0
GAP_MERGE_S = 1.0  # 朗读中的自然停顿小于 1 秒时仍算同一段语流


def decode_m4a(path: Path) -> np.ndarray:
    """Decode to 16 kHz mono float32 PCM via ffmpeg pipe."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def speech_chunks(samples: np.ndarray) -> list[tuple[int, int]]:
    """Group active frames into 4-8 s chunks, splitting on >=0.35 s silence."""
    frame, hop = int(0.025 * SR), int(0.010 * SR)
    n_frames = 1 + max(0, len(samples) - frame) // hop
    starts = np.arange(n_frames) * hop
    frames = np.stack([samples[s : s + frame] for s in starts])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 1e-6:
        return []
    # Absolute threshold relative to the loud-speech level (p90), not the peak:
    # a single click must not silence a quiet-but-clean recording.
    speech_level = float(np.percentile(rms, 90))
    noise_floor = float(np.percentile(rms, 20))
    active = rms >= max(noise_floor * 2.0, speech_level * 0.15)

    # active frame index runs -> merge gaps < 0.35 s -> chunk by duration
    idx = np.flatnonzero(active)
    if not idx.size:
        return []
    segments: list[tuple[int, int]] = []  # in frame units
    seg_start = prev = int(idx[0])
    gap_frames = int(GAP_MERGE_S / 0.010)
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap_frames:
            segments.append((seg_start, prev))
            seg_start = i
        prev = i
    segments.append((seg_start, prev))

    chunks: list[tuple[int, int]] = []
    min_f, max_f = int(CHUNK_MIN_S / 0.010), int(CHUNK_MAX_S / 0.010)
    for a, b in segments:
        length = b - a + 1
        if length < min_f:
            continue
        pos = a
        while b - pos + 1 >= min_f:
            end = min(pos + max_f - 1, b)
            chunks.append((starts[pos], int(starts[end]) + frame))
            pos = end + 1
    return chunks


def main() -> None:
    manifest = ROOT / "data" / "speaker_recognition" / "metadata.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    existing = {r["file_name"] for r in rows}

    new_rows = []
    for sid, (name, src) in SOURCES.items():
        print(f"== {name} ({sid}) 解码 {src.name} ……")
        samples = decode_m4a(src)
        duration = len(samples) / SR
        chunks = speech_chunks(samples)
        frame_rms_ref = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) or 1.0
        kept = 0
        out_dir = ROOT / "data" / "speaker_recognition" / "wav" / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (a, b) in enumerate(chunks, 1):
            seg = samples[a:b]
            if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) < frame_rms_ref * 0.3:
                continue
            kept += 1
            fname = f"wav/{sid}/{sid}_train_free01_take{kept:02d}.wav"
            if fname in existing:
                continue
            pcm = np.clip(seg, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2")
            import soundfile as sf
            sf.write(str(ROOT / "data" / "speaker_recognition" / fname), pcm16, SR, subtype="PCM_16")
            new_rows.append({
                "file_name": fname, "speaker_id": sid, "speaker_name": name,
                "split": "train", "prompt_id": "free01", "take": str(kept),
                "original_name": src.name, "original_format": "M4A",
                "source_codec": "aac/alac", "source_sample_rate_hz": "48000",
                "source_channels": "1", "duration_s": f"{(b - a) / SR:.3f}",
                "wav_sample_rate_hz": "16000", "wav_channels": "1", "wav_subtype": "PCM_16",
            })
        print(f"   全长 {duration:.1f}s -> {len(chunks)} 个语音块，保留 {kept} 条")
    rows.extend(new_rows)
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"metadata.csv 新增 {len(new_rows)} 条训练记录，总计 {len(rows)} 行")


if __name__ == "__main__":
    main()
