"""Convert five members' phone recordings into a checked WAV dataset.

The input manifest is intentionally kept out of Git because it contains names and
local paths.  See ``data/import_manifest.example.csv`` for the required columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg"}
TAKE_WORDS = {"第一遍": 1, "第一次": 1, "第二遍": 2, "第二次": 2, "第三遍": 3, "第三次": 3}


@dataclass(frozen=True)
class Recording:
    speaker_id: str
    member_code: str
    display_name: str
    source: Path
    original_name: str
    content: str
    take: int


def _safe_zip_entries(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        entries = []
        for entry in archive.infolist():
            pure = Path(entry.filename.replace("\\", "/"))
            if entry.is_dir() or pure.is_absolute() or ".." in pure.parts:
                continue
            if "__MACOSX" in pure.parts or pure.name.startswith("._") or pure.name == ".DS_Store":
                continue
            if pure.suffix.lower() in AUDIO_SUFFIXES:
                entries.append(entry)
        return entries


def _repair_zip_name(name: str) -> str:
    """Repair common Windows/Chinese ZIP names stored without the UTF-8 flag."""
    for encoding in ("utf-8", "gbk"):
        try:
            return name.encode("cp437").decode(encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


def _take_from_name(name: str) -> int | None:
    for word, take in TAKE_WORDS.items():
        if word in name:
            return take
    match = re.search(r"(?:take|第)?0?([123])(?:遍|次)?", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _classify(name: str) -> tuple[str, int] | None:
    stem = Path(name).stem
    take = _take_from_name(stem)
    if take is None:
        return None
    digit = re.match(r"^([0-9])", stem)
    if digit:
        return digit.group(1), take
    if "姓名" in stem or not re.search(r"(^|[_\-\s])[0-9]([_\-\s]|$)", stem):
        return "name", take
    return None


def _copy_sources(row: dict[str, str], temp_dir: Path) -> list[Recording]:
    source = Path(row["source_path"])
    if not source.exists():
        raise FileNotFoundError(source)

    files: list[Path]
    if source.is_dir():
        files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    elif source.suffix.lower() == ".zip":
        extract_root = temp_dir / row["speaker_id"]
        extract_root.mkdir(parents=True, exist_ok=True)
        files = []
        with zipfile.ZipFile(source) as archive:
            for entry in _safe_zip_entries(source):
                destination = extract_root / Path(_repair_zip_name(entry.filename)).name
                with archive.open(entry) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                files.append(destination)
    else:
        raise ValueError(f"Unsupported source: {source}")

    recordings = []
    for file_path in sorted(files, key=lambda item: item.name):
        classified = _classify(file_path.name)
        if classified is None:
            print(f"warning: skipped unrecognized file name: {file_path.name}")
            continue
        content, take = classified
        recordings.append(
            Recording(
                speaker_id=row["speaker_id"],
                member_code=row["member_code"],
                display_name=row["display_name"],
                source=file_path,
                original_name=file_path.name,
                content=content,
                take=take,
            )
        )
    return recordings


def _probe(path: Path, ffprobe: str) -> dict[str, object]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration") or 0
    return {
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration_seconds": float(duration),
    }


def _convert(recording: Recording, output_root: Path, ffmpeg: str) -> tuple[Path, dict[str, float]]:
    category = "digits" if recording.content != "name" else "names"
    token = f"digit_{recording.content}" if category == "digits" else "name"
    output = output_root / recording.speaker_id / category / (
        f"{recording.speaker_id}_{token}_take{recording.take:02d}.wav"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(recording.source),
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_CONFIG.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )
    sample_rate, samples = wavfile.read(output)
    values = samples.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
    clipped = float(np.mean(np.abs(values) >= 0.999)) if values.size else 0.0
    return output, {
        "sample_rate": float(sample_rate),
        "duration_seconds": len(values) / sample_rate,
        "peak": peak,
        "rms": rms,
        "clipped_ratio": clipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.csv"))
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for member in manifest:
        speaker_root = args.output / member["speaker_id"]
        if speaker_root.exists():
            for generated_wav in speaker_root.rglob("*.wav"):
                generated_wav.unlink()
    with tempfile.TemporaryDirectory(prefix="dsp-intake-") as temporary:
        temp_dir = Path(temporary)
        for member in manifest:
            for recording in _copy_sources(member, temp_dir):
                original = _probe(recording.source, ffprobe)
                output, quality = _convert(recording, args.output, ffmpeg)
                warnings = []
                if quality["duration_seconds"] < 0.5:
                    warnings.append("too_short")
                if quality["peak"] < 0.02:
                    warnings.append("very_quiet")
                if quality["clipped_ratio"] > 0.001:
                    warnings.append("clipping")
                rows.append(
                    {
                        "file_name": output.name,
                        "relative_path": output.as_posix(),
                        "speaker_id": recording.speaker_id,
                        "member_code": recording.member_code,
                        "display_name": recording.display_name,
                        "content": recording.content,
                        "take": recording.take,
                        "original_file": recording.original_name,
                        "original_format": recording.source.suffix.lower().lstrip("."),
                        "original_codec": original["codec"],
                        "original_sample_rate": original["sample_rate"],
                        "original_channels": original["channels"],
                        "sample_rate": int(quality["sample_rate"]),
                        "channels": 1,
                        "duration_seconds": f"{quality['duration_seconds']:.3f}",
                        "peak": f"{quality['peak']:.5f}",
                        "rms": f"{quality['rms']:.5f}",
                        "clipped_ratio": f"{quality['clipped_ratio']:.6f}",
                        "status": "warning" if warnings else "ok",
                        "notes": ";".join(warnings),
                    }
                )

    rows.sort(key=lambda row: (str(row["speaker_id"]), str(row["content"]), int(row["take"])))
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"converted: {len(rows)} recordings")
    print(f"metadata: {args.metadata}")


if __name__ == "__main__":
    main()
