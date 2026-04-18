from __future__ import annotations

import importlib.util
import time
from typing import Callable

import numpy as np

from momsa.benchmarks.runner import evaluate_front


LogFn = Callable[[str], None] | None


def has_pymoo() -> bool:
    return importlib.util.find_spec("pymoo") is not None


def baseline_note() -> str:
    if has_pymoo():
        return "pymoo is available, so baseline comparisons can run for NSGA-II, SPEA2, and MOEA/D."
    return "pymoo is not installed; benchmark problems and metrics still work, but standard algorithm baselines are disabled."


def run_pymoo_baselines(problem, n_pf_points: int = 300, iterations: int = 250, runs: int = 10, log: LogFn = None) -> list[dict]:
    if not has_pymoo():
        return []

    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    class WrappedProblem(Problem):
        def __init__(self, source_problem):
            self.source_problem = source_problem
            super().__init__(
                n_var=source_problem.bounds.dim,
                n_obj=source_problem.n_obj,
                n_ieq_constr=0,
                xl=source_problem.bounds.lower,
                xu=source_problem.bounds.upper,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.asarray([self.source_problem.evaluate(row) for row in x], dtype=float)

    wrapped = WrappedProblem(problem)
    true_front = problem.pareto_front(n_pf_points)
    algorithms = []
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    pop_size = max(100, len(ref_dirs))

    algorithms.extend(
        [
            ("nsga2", lambda: NSGA2(pop_size=pop_size)),
            ("spea2", lambda: SPEA2(pop_size=pop_size)),
            ("moead", lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=min(15, len(ref_dirs)), prob_neighbor_mating=0.7)),
        ]
    )

    summaries = []
    for name, builder in algorithms:
        run_records = []
        best_front = np.empty((0, problem.n_obj))
        best_gd = float("inf")

        for run_idx in range(runs):
            seed = run_idx
            if log:
                log(f"[{problem.name}] {name.upper()} run {run_idx + 1}/{runs} started with seed={seed}")
            started = time.perf_counter()
            result = minimize(wrapped, builder(), ("n_gen", iterations), seed=seed, verbose=False)
            elapsed = time.perf_counter() - started
            front = np.asarray(result.F, dtype=float)
            metrics = evaluate_front(front, true_front)
            record = {
                "algorithm": name,
                "problem": problem.name,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "front": front,
                **metrics,
            }
            run_records.append(record)

            if record["gd"] < best_gd:
                best_gd = record["gd"]
                best_front = front

            if log:
                log(
                    f"[{problem.name}] {name.upper()} run {run_idx + 1}/{runs} finished in {elapsed:.2f}s "
                    f"(GD={record['gd']:.10f}, S={record['spacing']:.10f}, "
                    f"Delta={record['spread']:.10f}, MS={record['max_spread']:.10f})"
                )

        summaries.append(
            {
                "algorithm": name,
                "problem": problem.name,
                "runs": len(run_records),
                "gd_mean": float(np.mean([r["gd"] for r in run_records])),
                "gd_std": float(np.std([r["gd"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "spacing_mean": float(np.mean([r["spacing"] for r in run_records])),
                "spacing_std": float(np.std([r["spacing"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "spread_mean": float(np.mean([r["spread"] for r in run_records])),
                "spread_std": float(np.std([r["spread"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "max_spread_mean": float(np.mean([r["max_spread"] for r in run_records])),
                "max_spread_std": float(np.std([r["max_spread"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "archive_mean": float(np.mean([r["archive_size"] for r in run_records])),
                "archive_std": float(np.std([r["archive_size"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "elapsed_mean": float(np.mean([r["elapsed_seconds"] for r in run_records])),
                "elapsed_std": float(np.std([r["elapsed_seconds"] for r in run_records], ddof=1)) if len(run_records) > 1 else 0.0,
                "records": [
                    {key: value for key, value in record.items() if key != "front"}
                    for record in run_records
                ],
                "true_front": true_front,
                "plot_front": best_front,
            }
        )

    return summaries
