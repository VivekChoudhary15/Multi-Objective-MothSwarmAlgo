from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Array = np.ndarray


@dataclass(slots=True)
class Bounds:
    lower: Array
    upper: Array

    def clip(self, x: Array) -> Array:
        return np.clip(x, self.lower, self.upper)

    @property
    def dim(self) -> int:
        return int(self.lower.shape[0])


@dataclass(slots=True)
class SingleObjectiveResult:
    x_best: Array
    f_best: float
    history: list[float] = field(default_factory=list)


@dataclass(slots=True)
class MultiObjectiveResult:
    archive_x: Array
    archive_f: Array
    history: list[int] = field(default_factory=list)
