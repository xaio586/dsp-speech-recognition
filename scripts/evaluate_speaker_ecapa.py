"""Evaluate the deployed ECAPA-TDNN + cosine-centroid speaker recognizer (ONNX).

Three test groups, all with pure onnxruntime inference (no PyTorch):
1. The 27 held-out test recordings, as recorded (clean).
2. The same 27 after the simulated Bluetooth/telephone channel.
3. Extra out-of-lab recordings of spk01 (IELTS practice, English, random
   content) under spk01_extra_test/ — segmented with the same energy VAD as
   the training ingestion. These files are evaluation-only and are never
   added to the training set.

Writes outputs/experiment4_speaker_ecapa/{summary.json,predictions*.csv}.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.speaker_ecapa import load_ecapa_recognizer  # noqa: E402
from evaluate_speaker_svm import bluetooth_channel, normalize_prompt  # noqa: E402

OUT_DIR = ROOT / "outputs" / "experiment4_speaker_ecapa"
EXTRA_TEST_DIR = ROOT / "spk01_extra_test"


def predict_row(recognizer, samples: np.ndarray) -> dict[str, object]:
    return recognizer.predict_waveform(np.ascontiguousarray(samples, dtype=np.float32))


def run_group(recognizer, items, label: str) -> list[dict[str, object]]:
    """items: list of (samples_16k, true_speaker_id, true_name, tag, content_group)."""
    predictions = []
    for samples, true_id, true_name, tag, group in items:
        result = predict_row(recognizer, samples)
        predictions.append({
            "group": label,
            "file_name": tag,
            "content_group": group,
            "true_speaker_id": true_id,
            "true_name": true_name,
            "predicted_speaker_id": result["speaker_id"],
            "predicted_name": result["speaker_name"],
            "correct": result["speaker_id"] == true_id,
            "confidence": round(float(result["confidence"]), 4),
            "top_similarity": round(float(result["similarity"]), 4),
            "low_confidence": result["low_confidence"],
        })
    ok = sum(p["correct"] for p in predictions)
    print(f"{label}: {ok}/{len(predictions)} = {ok / len(predictions):.1%}")
    for p in predictions:
        if not p["correct"]:
            print(f"  误判: {p['file_name']} -> {p['predicted_name']} (相似度 {p['top_similarity']})")
    return predictions


def summarize(predictions, ids):
    cm = {a: {b: 0 for b in ids} for a in ids}
    for p in predictions:
        cm[p["true_speaker_id"]][p["predicted_speaker_id"]] += 1
    return {
        "n": len(predictions),
        "accuracy": float(np.mean([p["correct"] for p in predictions])),
        "per_speaker_recall": {
            sid: (cm[sid][sid] / sum(cm[sid].values()) if sum(cm[sid].values()) else 0.0)
            for sid in ids
        },
        "confusion_matrix": cm,
        "misclassified": [p["file_name"] for p in predictions if not p["correct"]],
    }


def main() -> None:
    bundle = load_ecapa_recognizer(ROOT)
    if bundle is None:
        raise SystemExit("缺少 ecapa_encoder.onnx / ecapa_centroids.npz / metadata.csv")
    recognizer, names = bundle

    manifest = ROOT / "data" / "speaker_recognition" / "metadata.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]

    def load_test_items(channel: str):
        items = []
        for row in test_rows:
            audio = load_wav(ROOT / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
            signal = audio.samples if channel == "clean" else bluetooth_channel(audio.samples, audio.sample_rate)
            group = "personal" if normalize_prompt(row["prompt_id"]).startswith("personal") else "shared"
            items.append((signal, row["speaker_id"], row["speaker_name"], row["file_name"], group))
        return items

    clean = run_group(recognizer, load_test_items("clean"), "原测试集·干净")
    bluetooth = run_group(recognizer, load_test_items("bluetooth"), "原测试集·蓝牙声道模拟")

    extra = []
    if EXTRA_TEST_DIR.is_dir():
        from ingest_free_speech import decode_m4a, speech_chunks

        for path in sorted(EXTRA_TEST_DIR.glob("*.m4a")):
            samples = decode_m4a(path)
            for index, (start, end) in enumerate(speech_chunks(samples), 1):
                extra.append((samples[start:end], "spk01", names.get("spk01", "spk01"), f"{path.stem}#{index}", "extra"))
        extra = run_group(recognizer, extra, "spk01 雅思录音（英语随机内容）")
    else:
        print("未找到 spk01_extra_test/，跳过额外测试组")

    ids = sorted(names)
    summary = {
        "experiment": "ECAPA-TDNN 嵌入 + 余弦质心说话人识别（实验四部署版，ONNX 推理）",
        "encoder": "SpeechBrain spkrec-ecapa-voxceleb（ONNX 导出，fp32）",
        "enrollment": "每位说话人全部训练录音的嵌入均值，L2 归一化为质心",
        "decision": "余弦相似度 argmax；softmax(相似度×10) 作为展示置信度，阈值 0.55 标记低置信度",
        "speakers": names,
        "training_recordings": len(train_rows),
        "test_recordings": len(test_rows),
        "clean_channel": summarize(clean, ids),
        "simulated_bluetooth_channel": summarize(bluetooth, ids),
        "spk01_extra_ielts": summarize(extra, ids) if extra else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    for name, preds in (("predictions", clean), ("predictions_bluetooth_sim", bluetooth), ("predictions_spk01_extra", extra)):
        if not preds:
            continue
        with (OUT_DIR / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(preds[0].keys()))
            writer.writeheader()
            writer.writerows(preds)
    print(f"\n结果已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
