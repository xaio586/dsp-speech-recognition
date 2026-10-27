"""Dynamic time warping for experiment 3."""

from __future__ import annotations

import numpy as np


def dtw_distance(query: np.ndarray, template: np.ndarray) -> float:
    first = np.asarray(query, dtype=np.float32)
    second = np.asarray(template, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("inputs must have shapes (frames, features) with equal feature counts")
    costs = np.full((len(first) + 1, len(second) + 1), np.inf, dtype=np.float64)
    steps = np.zeros_like(costs, dtype=np.int32)
    costs[0, 0] = 0.0
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            candidates = (costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])
            choice = int(np.argmin(candidates))
            predecessors = ((i - 1, j), (i, j - 1), (i - 1, j - 1))
            previous_i, previous_j = predecessors[choice]
            costs[i, j] = candidates[choice] + np.linalg.norm(first[i - 1] - second[j - 1])
            steps[i, j] = steps[previous_i, previous_j] + 1
    return float(costs[-1, -1] / max(steps[-1, -1], 1))


def recognize_digit(
    features: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> tuple[str, float]:
    scores = {
        label: min(dtw_distance(features, template) for template in examples)
        for label, examples in templates.items()
        if examples
    }
    if not scores:
        raise ValueError("templates must contain at least one example")
    label = min(scores, key=scores.get)
    return label, scores[label]
