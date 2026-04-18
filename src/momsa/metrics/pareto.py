from __future__ import annotations

import numpy as np

from momsa.types import Array


def dominates(a: Array, b: Array) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(points: Array) -> Array:
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if dominates(points[j], points[i]):
                mask[i] = False
                break
    return mask


def filter_nondominated(xs: Array, fs: Array) -> tuple[Array, Array]:
    mask = non_dominated_mask(fs)
    return xs[mask], fs[mask]


def crowding_distance(points: Array) -> Array:
    n, m = points.shape
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    distances = np.zeros(n, dtype=float)
    for obj in range(m):
        order = np.argsort(points[:, obj])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        span = points[order[-1], obj] - points[order[0], obj]
        if np.isclose(span, 0.0):
            continue

        for rank in range(1, n - 1):
            left = points[order[rank - 1], obj]
            right = points[order[rank + 1], obj]
            distances[order[rank]] += (right - left) / span

    return distances


def truncate_by_crowding(xs: Array, fs: Array, max_size: int) -> tuple[Array, Array]:
    if len(fs) <= max_size:
        return xs, fs

    distances = crowding_distance(fs)
    order = np.argsort(-distances)
    keep = order[:max_size]
    return xs[keep], fs[keep]
