"""Evaluate the experiment-4 MFCC/LPC-statistics + RBF SVM speaker recognizer.

Protocol:
1. Training uses channel augmentation (telephone bandpass + 8 kHz round-trip,
   6 kHz low-pass, 20 dB SNR noise): 40 recordings -> 160 training vectors.
   This was added after live Bluetooth-headset tests collapsed onto one
   speaker; augmentation targets robustness to microphone/channel shift.
2. Hyperparameters (C, gamma) are chosen by grouped cross-validation on the
   training set only; augmented copies of one recording stay in the same
   group so no variant straddles train/validation.
3. The final model is refit on all training vectors and evaluated once on
   the 27 held-out test recordings, both as-is and after applying the
   simulated Bluetooth/telephone channel. Personal free-speech prompts are
   the primary content-independent metric; shared prompts are a secondary
   check.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.speaker_recognition import (  # noqa: E402
    build_training_vectors,
    prepare_speaker_utterance,
    train_speaker_svm,
    utterance_feature_vector,
)

GRID_C = [1.0, 10.0, 100.0]
GRID_GAMMA = ["scale", 0.03, 0.1]
AUGMENT_FACTOR = 4  # original + 3 channel variants
OUT_DIR = ROOT / "outputs" / "experiment4_speaker_svm"


def normalize_prompt(prompt_id: str) -> str:
    """Unify prompt ids like 'sentence01'/'sentence1' and 'shared01'/'shared1'."""
    text = prompt_id.strip().lower()
    for prefix in ("sentence", "shared", "personal"):
        if text.startswith(prefix):
            suffix = text[len(prefix):].lstrip("0") or "0"
            return f"{prefix}{suffix}"
    return text


def load_rows() -> list[dict[str, str]]:
    manifest = ROOT / "data" / "speaker_recognition" / "metadata.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bluetooth_channel(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulated Bluetooth/telephone channel: 300-3400 Hz bandpass + 8 kHz round-trip."""
    sos = butter(4, [300, 3400], btype="band", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, samples)
    filtered = resample_poly(filtered, 1, 2)
    filtered = resample_poly(filtered, 2, 1)
    return filtered.astype(np.float32)


def grouped_cv(train_rows, vectors_by_speaker) -> list[dict[str, object]]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    for sid in sorted(vectors_by_speaker):
        for index, vector in enumerate(vectors_by_speaker[sid]):
            vectors.append(vector)
            labels.append(sid)
            groups.append(index // AUGMENT_FACTOR)  # augmented copies stay together
    matrix = np.stack(vectors)
    labels_arr = np.array(labels)
    groups_arr = np.array([f"{sid}_{g}" for sid, g in zip(labels, groups)])
    logo = LeaveOneGroupOut()
    results = []
    for c in GRID_C:
        for gamma in GRID_GAMMA:
            scores = []
            for fit_idx, hold_idx in logo.split(matrix, labels_arr, groups_arr):
                clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma))
                clf.fit(matrix[fit_idx], labels_arr[fit_idx])
                scores.append(float(np.mean(clf.predict(matrix[hold_idx]) == labels_arr[hold_idx])))
            results.append({"C": c, "gamma": gamma, "mean_accuracy": float(np.mean(scores))})
    return results


def evaluate_test_set(model, test_rows, channel: str) -> list[dict[str, object]]:
    predictions = []
    for row in test_rows:
        audio = load_wav(ROOT / "data" / "speaker_recognition" / row["file_name"], target_sample_rate=16000)
        signal = audio.samples if channel == "clean" else bluetooth_channel(audio.samples, audio.sample_rate)
        mfcc, lpc, _ = prepare_speaker_utterance(signal, audio.sample_rate)
        result = model.predict_mfcc(mfcc, lpc)
        predictions.append({
            "file_name": row["file_name"],
            "prompt_id": row["prompt_id"],
            "content_group": "personal" if normalize_prompt(row["prompt_id"]).startswith("personal") else "shared",
            "true_speaker_id": row["speaker_id"],
            "true_name": row["speaker_name"],
            "predicted_speaker_id": result["speaker_id"],
            "predicted_name": result["speaker_name"],
            "correct": result["speaker_id"] == row["speaker_id"],
            "confidence": round(float(result["confidence"]), 4),
            "low_confidence": result["low_confidence"],
        })
    return predictions


def summarize(predictions, ids):
    cm = {a: {b: 0 for b in ids} for a in ids}
    for p in predictions:
        cm[p["true_speaker_id"]][p["predicted_speaker_id"]] += 1
    recall = {sid: (cm[sid][sid] / sum(cm[sid].values()) if sum(cm[sid].values()) else 0.0) for sid in ids}

    def subset(group: str) -> float | None:
        items = [p for p in predictions if p["content_group"] == group]
        return float(np.mean([p["correct"] for p in items])) if items else None

    return {
        "accuracy_overall": float(np.mean([p["correct"] for p in predictions])),
        "accuracy_personal_free_speech": subset("personal"),
        "accuracy_shared_prompts": subset("shared"),
        "per_speaker_recall": recall,
        "confusion_matrix": cm,
        "misclassified": [p for p in predictions if not p["correct"]],
    }


def main() -> None:
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    names = {r["speaker_id"]: r["speaker_name"] for r in rows}
    print(f"训练 {len(train_rows)} 条（增强后 {len(train_rows) * AUGMENT_FACTOR} 条向量），测试 {len(test_rows)} 条")
    print("正在提取增强训练特征（含声道模拟）……")
    vectors_by_speaker, _ = build_training_vectors(ROOT, augment=True)

    print("正在做按录音分组的交叉验证（仅训练集，增强副本同组）……")
    cv_results = grouped_cv(train_rows, vectors_by_speaker)
    best = max(cv_results, key=lambda item: (item["mean_accuracy"], -float(item["C"])))
    print(f"交叉验证最优：C={best['C']} gamma={best['gamma']} 平均准确率={best['mean_accuracy']:.3f}")

    model = train_speaker_svm(vectors_by_speaker, names, c=float(best["C"]), gamma=best["gamma"])

    clean = evaluate_test_set(model, test_rows, "clean")
    bluetooth = evaluate_test_set(model, test_rows, "bluetooth")
    ids = sorted(names)

    summary = {
        "experiment": "MFCC/LPC 统计特征 + RBF SVM 说话人识别（实验四，声道增强版）",
        "feature": "MFCC+delta+delta2 逐系数均值/标准差（78 维）+ 逐帧 LPC(12) 均值/标准差（24 维），共 102 维定长向量",
        "augmentation": "每条训练录音增加 3 个声道变体（电话带通+8kHz 往返、6kHz 低通、20dB 白噪声），训练向量 40→160",
        "speakers": names,
        "training_recordings": len(train_rows),
        "training_vectors": sum(len(v) for v in vectors_by_speaker.values()),
        "test_recordings": len(test_rows),
        "cv_protocol": "训练集按原始录音 LeaveOneGroupOut（增强副本同组）",
        "cv_results": cv_results,
        "selected_hyperparameters": {"C": best["C"], "gamma": best["gamma"]},
        "clean_channel": summarize(clean, ids),
        "simulated_bluetooth_channel": summarize(bluetooth, ids),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0].keys()))
        writer.writeheader()
        writer.writerows(clean)
    with (OUT_DIR / "predictions_bluetooth_sim.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bluetooth[0].keys()))
        writer.writeheader()
        writer.writerows(bluetooth)
    with (OUT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + ids)
        cm = summary["clean_channel"]["confusion_matrix"]
        for sid in ids:
            writer.writerow([sid] + [cm[sid][p] for p in ids])

    print(json.dumps({k: v for k, v in summary.items() if k not in ("cv_results", "speakers")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
