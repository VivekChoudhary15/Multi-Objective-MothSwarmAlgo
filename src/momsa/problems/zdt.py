from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from momsa.types import Array, Bounds


@dataclass(slots=True)
class ZDTProblem:
    name: str
    n_var: int
    kind: str
    n_obj: int = field(init=False)
    bounds: Bounds = field(init=False)

    def __post_init__(self) -> None:
        self.n_obj = 2
        self.bounds = Bounds(
            lower=np.zeros(self.n_var, dtype=float),
            upper=np.ones(self.n_var, dtype=float),
        )

    def evaluate(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        f1 = x[0]

        if self.kind in {"zdt1", "zdt2", "zdt3"}:
            g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        elif self.kind == "zdt4":
            g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        elif self.kind == "zdt6":
            g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        else:
            raise ValueError(f"Unsupported ZDT problem kind: {self.kind}")

        if self.kind == "zdt1":
            h = 1.0 - np.sqrt(f1 / g)
        elif self.kind == "zdt2":
            h = 1.0 - (f1 / g) ** 2
        elif self.kind == "zdt3":
            h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        elif self.kind == "zdt4":
            h = 1.0 - np.sqrt(f1 / g)
        elif self.kind == "zdt6":
            f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
            h = 1.0 - (f1 / g) ** 2
        else:
            raise ValueError(f"Unsupported ZDT problem kind: {self.kind}")

        f2 = g * h
        return np.array([f1, f2], dtype=float)

    def pareto_front(self, n_points: int = 200) -> Array:
        x = np.linspace(0.0, 1.0, n_points)

        if self.kind == "zdt1":
            return np.column_stack([x, 1.0 - np.sqrt(x)])
        if self.kind == "zdt2":
            return np.column_stack([x, 1.0 - x**2])
        if self.kind == "zdt3":
            intervals = [
                (0.0, 0.0830015349),
                (0.1822287280, 0.2577623634),
                (0.4093136748, 0.4538821041),
                (0.6183967944, 0.6525117038),
                (0.8233317983, 0.8518328654),
            ]
            points_per_interval = max(2, int(np.ceil(n_points / len(intervals))))
            xs = np.concatenate(
                [
                    np.linspace(start, end, points_per_interval, endpoint=True)
                    for start, end in intervals
                ]
            )
            f2 = 1.0 - np.sqrt(xs) - xs * np.sin(10.0 * np.pi * xs)
            return np.column_stack([xs, f2])
        if self.kind == "zdt4":
            return np.column_stack([x, 1.0 - np.sqrt(x)])
        if self.kind == "zdt6":
            f1 = np.linspace(0.280775, 1.0, n_points)
            # f1 = 1.0 - np.exp(-4.0 * x) * np.sin(6.0 * np.pi * x) ** 6
            f2 = 1.0 - f1**2
            return np.column_stack([f1, f2])

        raise ValueError(f"Unsupported ZDT problem kind: {self.kind}")


def make_zdt_suite() -> list[ZDTProblem]:
    return [
        ZDTProblem(name="ZDT1", n_var=30, kind="zdt1"),
        ZDTProblem(name="ZDT2", n_var=30, kind="zdt2"),
        ZDTProblem(name="ZDT3", n_var=30, kind="zdt3"),
        ZDTProblem(name="ZDT4", n_var=10, kind="zdt4"),
        ZDTProblem(name="ZDT6", n_var=10, kind="zdt6"),
    ]
