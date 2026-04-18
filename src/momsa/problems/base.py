from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from momsa.types import Array, Bounds


class SingleObjectiveProblem(Protocol):
    name: str
    bounds: Bounds

    def evaluate(self, x: Array) -> float: ...


class MultiObjectiveProblem(Protocol):
    name: str
    bounds: Bounds
    n_obj: int

    def evaluate(self, x: Array) -> Array: ...

    def pareto_front(self, n_points: int = 200) -> Array: ...


@dataclass(slots=True)
class SphereProblem:
    dim: int = 30
    name: str = "sphere"
    bounds: Bounds = field(init=False)

    def __post_init__(self) -> None:
        self.bounds = Bounds(
            lower=np.full(self.dim, -5.12, dtype=float),
            upper=np.full(self.dim, 5.12, dtype=float),
        )

    def evaluate(self, x: Array) -> float:
        return float(np.sum(np.square(x)))
