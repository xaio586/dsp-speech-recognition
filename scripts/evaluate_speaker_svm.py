"""Evaluate the experiment-4 MFCC/LPC-statistics + RBF SVM speaker recognizer.

Protocol:
1. Feature set (MFCC+delta+delta2 stats, optionally plus LPC stats) and
   hyperparameters (C, gamma) are chosen by leave-one-sentence-out grouped
   cross-validation on the 40 training utterances only (groups = prompt id,
   so the two takes of the same sentence never straddle train/validation).
2. The final model is refit on all training utterances and evaluated once on
   the 27 held-out test recordings. Personal free-speech prompts are the
   primary content-independent metric; shared prompts are a secondary check.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.speaker_recognition import (  # noqa: E402
    prepare_speaker_utterance,
    train_speaker_svm,
    utterance_feature_vector,
)

GRID_C = [1.0, 10.0, 100.0]
GRID_GAMMA = ["scale", 0.03, 0.1]
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


def extract_features(rows: list[dict[str, str]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """file_name -> (MFCC frames, LPC frames), endpoint-trimmed at 16 kHz."""
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        path = ROOT / "data" / "speaker_recognition" / row["file_name"]
        audio = load_wav(path, target_sample_rate=16000)
        mfcc, lpc, _ = prepare_speaker_utterance(audio.samples, audio.sample_rate)
        cache[row["file_name"]] = (mfcc, lpc)
    return cache


def grouped_cv(train_rows, features) -> list[dict[str, object]]:
    vectors = np.stack(
        [utterance_feature_vector(*features[r["file_name"]]) for r in train_rows]
    )
    labels = np.array([r["speaker_id"] for r in train_rows])
    groups = np.array([normalize_prompt(r["prompt_id"]) for r in train_rows])
    logo = LeaveOneGroupOut()
    results = []
    for c in GRID_C:
        for gamma in GRID_GAMMA:
            scores = []
            for fit_idx, hold_idx in logo.split(vectors, labels, groups):
                clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma))
                clf.fit(vectors[fit_idx], labels[fit_idx])
                scores.append(float(np.mean(clf.predict(vectors[hold_idx]) == labels[hold_idx])))
            results.append({"C": c, "gamma": gamma, "fold_accuracies": scores, "mean_accuracy": float(np.mean(scores))})
    return results


def main() -> None:
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    names = {r["speaker_id"]: r["speaker_name"] for r in rows}
    print(f"训练 {len(train_rows)} 条，测试 {len(test_rows)} 条，说话人 {len(names)} 人")
    print("正在提取 MFCC/LPC 特征……")
    features = extract_features(train_rows + test_rows)

    print("正在做按句子分组的交叉验证（仅训练集）……")
    cv_results = grouped_cv(train_rows, features)
    best = max(cv_results, key=lambda item: (item["mean_accuracy"], -float(item["C"])))
    print(f"交叉验证最优：C={best['C']} gamma={best['gamma']} 平均准确率={best['mean_accuracy']:.3f}")

    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        vectors[row["speaker_id"]].append(utterance_feature_vector(*features[row["file_name"]]))
    model = train_speaker_svm(dict(vectors), names, c=float(best["C"]), gamma=best["gamma"])

    predictions = []
    for row in test_rows:
        mfcc, lpc = features[row["file_name"]]
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

    ids = sorted(names)
    cm = {a: {b: 0 for b in ids} for a in ids}
    for p in predictions:
        cm[p["true_speaker_id"]][p["predicted_speaker_id"]] += 1
    recall = {sid: (cm[sid][sid] / sum(cm[sid].values()) if sum(cm[sid].values()) else 0.0) for sid in ids}
    overall = float(np.mean([p["correct"] for p in predictions]))

    def subset_accuracy(group: str) -> float | None:
        subset = [p for p in predictions if p["content_group"] == group]
        return float(np.mean([p["correct"] for p in subset])) if subset else None

    summary = {
        "experiment": "MFCC/LPC 统计特征 + RBF SVM 说话人识别（实验四）",
        "feature": "MFCC+delta+delta2 逐系数均值/标准差（78 维）+ 逐帧 LPC(12) 均值/标准差（24 维），共 102 维定长向量",
        "speakers": names,
        "training_recordings": len(train_rows),
        "test_recordings": len(predictions),
        "cv_protocol": "训练集按句子 LeaveOneGroupOut（4 折，同句两遍不跨折）",
        "cv_results": cv_results,
        "selected_hyperparameters": {"C": best["C"], "gamma": best["gamma"]},
        "accuracy_overall": overall,
        "accuracy_personal_free_speech": subset_accuracy("personal"),
        "accuracy_shared_prompts": subset_accuracy("shared"),
        "per_speaker_recall": recall,
        "confusion_matrix": cm,
        "misclassified": [p for p in predictions if not p["correct"]],
        "low_confidence_count": sum(1 for p in predictions if p["low_confidence"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    with (OUT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + ids)
        for sid in ids:
            writer.writerow([sid] + [cm[sid][p] for p in ids])

    print(json.dumps({k: v for k, v in summary.items() if k != "cv_results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
