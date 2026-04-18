from __future__ import annotations

import numpy as np

from momsa.types import Array


def _min_distances(points: Array, reference: Array) -> Array:
    diff = points[:, None, :] - reference[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.min(dist, axis=1)


def generational_distance(front: Array, true_front: Array) -> float:
    if len(front) == 0:
        return float("inf")
    distances = _min_distances(front, true_front)
    return float(np.sqrt(np.mean(distances**2)))


def spacing(front: Array) -> float:
    n = len(front)
    if n <= 1:
        return 0.0

    d = np.full(n, np.inf, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            manhattan = float(np.sum(np.abs(front[i] - front[j])))
            d[i] = min(d[i], manhattan)
    mean_d = np.mean(d)
    return float(np.sqrt(np.sum((d - mean_d) ** 2) / (n - 1)))


def spread(front: Array, true_front: Array) -> float:
    n = len(front)
    if n <= 1:
        return float("inf")

    front_sorted = front[np.argsort(front[:, 0])]
    true_sorted = true_front[np.argsort(true_front[:, 0])]
    consecutive = np.linalg.norm(front_sorted[1:] - front_sorted[:-1], axis=1)
    mean_c = float(np.mean(consecutive)) if len(consecutive) else 0.0
    df = float(np.linalg.norm(front_sorted[0] - true_sorted[0]))
    dl = float(np.linalg.norm(front_sorted[-1] - true_sorted[-1]))
    numerator = df + dl + np.sum(np.abs(consecutive - mean_c))
    denominator = df + dl + (n - 1) * mean_c
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(numerator / denominator)


def maximum_spread(front: Array, true_front: Array) -> float:
    if len(front) == 0:
        return 0.0

    f_min = np.min(front, axis=0)
    f_max = np.max(front, axis=0)
    pf_min = np.min(true_front, axis=0)
    pf_max = np.max(true_front, axis=0)

    overlap = np.minimum(f_max, pf_max) - np.maximum(f_min, pf_min)
    denom = pf_max - pf_min
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    normalized = np.zeros_like(overlap)
    normalized[valid] = np.maximum(overlap[valid], 0.0) / denom[valid]
    return float(np.sqrt(np.mean(normalized[valid] ** 2)))
