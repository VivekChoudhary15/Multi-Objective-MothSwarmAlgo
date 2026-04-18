from __future__ import annotations

from dataclasses import replace
from statistics import mean
import time
from typing import Callable

import numpy as np

from momsa.algorithms import MOMSA, MOMSAConfig
from momsa.metrics import generational_distance, maximum_spread, spacing, spread


LogFn = Callable[[str], None] | None


def evaluate_front(front: np.ndarray, true_front: np.ndarray) -> dict[str, float]:
    return {
        "gd": generational_distance(front, true_front),
        "spacing": spacing(front),
        "spread": spread(front, true_front),
        "max_spread": maximum_spread(front, true_front),
        "archive_size": float(len(front)),
    }


def _mean_metric(records: list[dict], key: str) -> float:
    return float(mean(record[key] for record in records))


def _std_metric(records: list[dict], key: str) -> float:
    if len(records) <= 1:
        return 0.0
    values = np.asarray([record[key] for record in records], dtype=float)
    return float(np.std(values, ddof=1))


def run_momsa_benchmark(problem, config, seeds: list[int] | None = None, log: LogFn = None) -> dict:
    seeds = seeds or list(range(config.runs))
    records = []
    true_front = problem.pareto_front(config.n_pf_points)
    best_run = None

    for run_index, seed in enumerate(seeds, start=1):
        if log:
            log(f"[{problem.name}] MOMSA run {run_index}/{len(seeds)} started with seed={seed}")

        started = time.perf_counter()
        algo_config: MOMSAConfig = replace(config.algorithm, seed=seed)
        result = MOMSA(algo_config).optimize(problem)
        elapsed = time.perf_counter() - started
        front = result.archive_f
        metrics = evaluate_front(front, true_front)
        record = {
            "algorithm": "momsa",
            "problem": problem.name,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "front": front,
            "archive_x": result.archive_x,
            **metrics,
        }
        records.append(record)

        if best_run is None or record["gd"] < best_run["gd"]:
            best_run = record

        if log:
            log(
                f"[{problem.name}] MOMSA run {run_index}/{len(seeds)} finished in {elapsed:.2f}s "
                f"(GD={record['gd']:.10f}, S={record['spacing']:.10f}, "
                f"Delta={record['spread']:.10f}, MS={record['max_spread']:.10f})"
            )

    summary = {
        "algorithm": "momsa",
        "problem": problem.name,
        "runs": len(records),
        "gd_mean": _mean_metric(records, "gd"),
        "gd_std": _std_metric(records, "gd"),
        "spacing_mean": _mean_metric(records, "spacing"),
        "spacing_std": _std_metric(records, "spacing"),
        "spread_mean": _mean_metric(records, "spread"),
        "spread_std": _std_metric(records, "spread"),
        "max_spread_mean": _mean_metric(records, "max_spread"),
        "max_spread_std": _std_metric(records, "max_spread"),
        "archive_mean": _mean_metric(records, "archive_size"),
        "archive_std": _std_metric(records, "archive_size"),
        "elapsed_mean": _mean_metric(records, "elapsed_seconds"),
        "elapsed_std": _std_metric(records, "elapsed_seconds"),
        "records": records,
        "true_front": true_front,
        "plot_front": best_run["front"] if best_run else np.empty((0, problem.n_obj)),
    }
    return summary


def format_summary(summary: dict) -> str:
    return (
        f"{summary['problem']} [{summary['algorithm']}]: "
        f"GD={summary['gd_mean']:.10f}, "
        f"S={summary['spacing_mean']:.10f}, "
        f"Delta={summary['spread_mean']:.10f}, "
        f"MS={summary['max_spread_mean']:.10f}, "
        f"Archive={summary['archive_mean']:.10f}, "
        f"Time={summary['elapsed_mean']:.10f}s"
    )


def build_metric_table(rows: list[dict]) -> str:
    headers = ["Problem", "Algorithm", "GD", "S", "Delta", "MS", "Archive", "Time(s)"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["problem"],
                row["algorithm"],
                f"{row['gd_mean']:.10f}",
                f"{row['spacing_mean']:.10f}",
                f"{row['spread_mean']:.10f}",
                f"{row['max_spread_mean']:.10f}",
                f"{row['archive_mean']:.10f}",
                f"{row['elapsed_mean']:.10f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in table_rows)
    return "\n".join(lines)


def serializable_summary(summary: dict) -> dict:
    compact_records = []
    for record in summary["records"]:
        compact_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"front", "archive_x"}
            }
        )

    return {
        key: value
        for key, value in summary.items()
        if key not in {"records", "true_front", "plot_front"}
    } | {"records": compact_records}
