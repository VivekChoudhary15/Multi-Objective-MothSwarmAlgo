from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from momsa.types import Array, Bounds


@dataclass(slots=True)
class DTLZProblem:
    name: str
    n_var: int
    n_obj: int
    kind: str
    bounds: Bounds = field(init=False)

    def __post_init__(self) -> None:
        self.bounds = Bounds(
            lower=np.zeros(self.n_var, dtype=float),
            upper=np.ones(self.n_var, dtype=float),
        )

    def evaluate(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        k = self.n_var - self.n_obj + 1
        xm = x[self.n_obj - 1 :]

        if self.kind == "dtlz1":
            g = 100.0 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5))))
            f = np.full(self.n_obj, 0.5 * (1.0 + g), dtype=float)
            for i in range(self.n_obj):
                for j in range(self.n_obj - i - 1):
                    f[i] *= x[j]
                if i > 0:
                    f[i] *= 1.0 - x[self.n_obj - i - 1]
            return f

        if self.kind == "dtlz2":
            g = np.sum((xm - 0.5) ** 2)
            f = np.full(self.n_obj, 1.0 + g, dtype=float)
            for i in range(self.n_obj):
                for j in range(self.n_obj - i - 1):
                    f[i] *= np.cos(0.5 * np.pi * x[j])
                if i > 0:
                    f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1])
            return f

        raise ValueError(f"Unsupported DTLZ problem kind: {self.kind}")

    def pareto_front(self, n_points: int = 200) -> Array:
        if self.n_obj != 3:
            raise ValueError("This helper currently supports only tri-objective Pareto fronts.")

        side = max(8, int(np.sqrt(n_points)))
        u = np.linspace(0.0, 1.0, side)
        grid = np.array(np.meshgrid(u, u)).reshape(2, -1).T
        pf = []
        for a, b in grid:
            if self.kind == "dtlz1" and a + b <= 1.0:
                pf.append([0.5 * a, 0.5 * b, 0.5 * (1.0 - a - b)])
            elif self.kind == "dtlz2":
                theta1 = 0.5 * np.pi * a
                theta2 = 0.5 * np.pi * b
                pf.append(
                    [
                        np.cos(theta1) * np.cos(theta2),
                        np.cos(theta1) * np.sin(theta2),
                        np.sin(theta1),
                    ]
                )
        return np.asarray(pf, dtype=float)


def make_dtlz_suite() -> list[DTLZProblem]:
    return [
        DTLZProblem(name="DTLZ1", n_var=7, n_obj=3, kind="dtlz1"),
        DTLZProblem(name="DTLZ2", n_var=12, n_obj=3, kind="dtlz2"),
    ]
