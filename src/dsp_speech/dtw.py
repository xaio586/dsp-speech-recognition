"""Dynamic time warping for experiment 3."""

from __future__ import annotations

import numpy as np


def dtw_distance(query: np.ndarray, template: np.ndarray) -> float:
    first = np.asarray(query, dtype=np.float32)
    second = np.asarray(template, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("inputs must have shapes (frames, features) with equal feature counts")
    if not len(first) or not len(second):
        raise ValueError("DTW inputs must contain at least one frame")

    # Calculate all frame-to-frame distances in NumPy's optimized array code.
    # The former implementation called np.linalg.norm and np.argmin for every
    # DTW cell, which is costly when a live query is compared with 160 templates.
    differences = first[:, None, :] - second[None, :, :]
    local_costs = np.sqrt(np.sum(differences * differences, axis=2, dtype=np.float64))

    columns = len(second)
    previous_costs = np.full(columns + 1, np.inf, dtype=np.float64)
    previous_steps = np.zeros(columns + 1, dtype=np.int32)
    previous_costs[0] = 0.0

    for i in range(1, len(first) + 1):
        current_costs = np.full(columns + 1, np.inf, dtype=np.float64)
        current_steps = np.zeros(columns + 1, dtype=np.int32)
        for j in range(1, columns + 1):
            up = previous_costs[j]
            left = current_costs[j - 1]
            diagonal = previous_costs[j - 1]
            # Keep np.argmin's original tie order: up, then left, then diagonal.
            if up <= left and up <= diagonal:
                best_cost, best_steps = up, previous_steps[j]
            elif left <= diagonal:
                best_cost, best_steps = left, current_steps[j - 1]
            else:
                best_cost, best_steps = diagonal, previous_steps[j - 1]
            current_costs[j] = best_cost + local_costs[i - 1, j - 1]
            current_steps[j] = best_steps + 1
        previous_costs = current_costs
        previous_steps = current_steps

    return float(previous_costs[-1] / max(previous_steps[-1], 1))


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

