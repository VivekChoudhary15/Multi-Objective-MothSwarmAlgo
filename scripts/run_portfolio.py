"""MOMSA on a long-only mean-variance portfolio problem.

Validates the algorithm against the analytical Markowitz efficient frontier
(closed-form long-only QP per target return) and contrasts with NSGA-II if
pymoo is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import strftime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/momsa-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momsa.algorithms import MOMSA, MOMSAConfig
from momsa.benchmarks.baselines import has_pymoo
from momsa.metrics import filter_nondominated, generational_distance
from momsa.problems import make_portfolio_problem


def log(msg: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {msg}", flush=True)


def run_momsa(problem, iterations: int, seed: int) -> np.ndarray:
    config = MOMSAConfig(
        population_size=100,
        pathfinder_count=5,
        prospector_start_ratio=0.4,
        prospector_end_ratio=0.1,
        iterations=iterations,
        archive_size=100,
        seed=seed,
    )
    result = MOMSA(config).optimize(problem)
    return result.archive_f


def run_nsga2(problem, iterations: int, seed: int) -> np.ndarray | None:
    if not has_pymoo():
        return None
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class Wrapped(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=problem.bounds.dim,
                n_obj=problem.n_obj,
                n_ieq_constr=0,
                xl=problem.bounds.lower,
                xu=problem.bounds.upper,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.asarray([problem.evaluate(row) for row in x], dtype=float)

    res = minimize(Wrapped(), NSGA2(pop_size=100), ("n_gen", iterations), seed=seed, verbose=False)
    return np.asarray(res.F, dtype=float)


def front_summary(label: str, front: np.ndarray, true_front: np.ndarray) -> dict:
    front_nd, _ = filter_nondominated(front, front)
    gd = generational_distance(front_nd, true_front) if len(front_nd) else float("inf")
    return {
        "label": label,
        "points": int(len(front_nd)),
        "min_var": float(front_nd[:, 0].min()) if len(front_nd) else float("nan"),
        "max_ret": float(-front_nd[:, 1].min()) if len(front_nd) else float("nan"),
        "gd_vs_analytical": gd,
    }


def plot_frontier(
    output_path: Path,
    analytical: np.ndarray,
    momsa_front: np.ndarray,
    nsga2_front: np.ndarray | None,
    n_assets: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(analytical):
        # Convert to (vol, return) for the standard finance view
        ax.plot(
            np.sqrt(analytical[:, 0]),
            -analytical[:, 1],
            color="black",
            linewidth=2.0,
            label="Analytical Markowitz frontier (SLSQP)",
        )

    if len(momsa_front):
        order = np.argsort(momsa_front[:, 0])
        ax.scatter(
            np.sqrt(momsa_front[order, 0]),
            -momsa_front[order, 1],
            s=22,
            color="tab:red",
            alpha=0.8,
            label="MOMSA archive",
            zorder=3,
        )

    if nsga2_front is not None and len(nsga2_front):
        order = np.argsort(nsga2_front[:, 0])
        ax.scatter(
            np.sqrt(nsga2_front[order, 0]),
            -nsga2_front[order, 1],
            s=22,
            color="tab:blue",
            alpha=0.6,
            marker="^",
            label="NSGA-II archive",
            zorder=2,
        )

    ax.set_xlabel("Portfolio volatility (annualized std. dev.)")
    ax.set_ylabel("Expected annual return")
    ax.set_title(f"Long-only mean-variance efficient frontier ({n_assets} assets)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOMSA on a portfolio problem.")
    parser.add_argument("--n-assets", type=int, default=15)
    parser.add_argument("--n-days", type=int, default=504)
    parser.add_argument("--n-sectors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--pf-points", type=int, default=80)
    parser.add_argument("--no-nsga2", action="store_true", help="Skip NSGA-II baseline")
    parser.add_argument("--output-plot", type=str, default="outputs/plots/portfolio_frontier.png")
    parser.add_argument("--output-table", type=str, default="outputs/portfolio_summary.txt")
    args = parser.parse_args()

    log(f"Building synthetic market: {args.n_assets} assets, {args.n_days} days, {args.n_sectors} sectors")
    problem, _daily, _sector = make_portfolio_problem(
        n_assets=args.n_assets,
        n_days=args.n_days,
        n_sectors=args.n_sectors,
        seed=args.seed,
    )

    log(f"Computing analytical Markowitz frontier ({args.pf_points} target returns)")
    analytical = problem.pareto_front(n_points=args.pf_points)
    analytical_nd, _ = filter_nondominated(analytical, analytical)
    log(f"  -> {len(analytical_nd)} non-dominated reference points")

    log(f"Running MOMSA (iterations={args.iterations}, seed={args.seed})")
    momsa_front = run_momsa(problem, iterations=args.iterations, seed=args.seed)
    log(f"  -> MOMSA archive size: {len(momsa_front)}")

    nsga2_front = None
    if not args.no_nsga2:
        if has_pymoo():
            log(f"Running NSGA-II baseline (iterations={args.iterations}, seed={args.seed})")
            nsga2_front = run_nsga2(problem, iterations=args.iterations, seed=args.seed)
            log(f"  -> NSGA-II front size: {len(nsga2_front) if nsga2_front is not None else 0}")
        else:
            log("pymoo not installed; skipping NSGA-II baseline")

    rows = [
        front_summary("Analytical (reference)", analytical_nd, analytical_nd),
        front_summary("MOMSA", momsa_front, analytical_nd),
    ]
    if nsga2_front is not None:
        rows.append(front_summary("NSGA-II", nsga2_front, analytical_nd))

    headers = ["Algorithm", "Points", "MinVar", "MaxRet", "GD-vs-analytical"]
    widths = [max(len(h), 22) for h in headers]
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [
            r["label"],
            str(r["points"]),
            f"{r['min_var']:.6f}",
            f"{r['max_ret']:.6f}",
            f"{r['gd_vs_analytical']:.6f}",
        ]
        lines.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)))
    table = "\n".join(lines)

    table_path = Path(args.output_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n")
    log(f"Saved summary table to {table_path}")
    print("\n" + table + "\n", flush=True)

    plot_path = Path(args.output_plot)
    plot_frontier(plot_path, analytical_nd, momsa_front, nsga2_front, args.n_assets)
    log(f"Saved frontier plot to {plot_path}")


if __name__ == "__main__":
    main()
