from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import strftime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from momsa.benchmarks import (
    BenchmarkConfig,
    baseline_note,
    build_metric_table,
    format_summary,
    paper_suite,
    run_momsa_benchmark,
    run_pymoo_baselines,
    save_pareto_plot,
    serializable_summary,
)


def log(message: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["problem", "algorithm", "gd_mean", "spacing_mean", "spread_mean", "max_spread_mean", "archive_mean", "elapsed_mean"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for name in fieldnames:
                value = row.get(name, "")
                if isinstance(value, float):
                    formatted[name] = f"{value:.10f}"
                else:
                    formatted[name] = value
            writer.writerow(formatted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSA/MOMSA benchmark suites.")
    parser.add_argument("--suite", default="paper", choices=["paper"])
    parser.add_argument("--algorithm", default="momsa", choices=["momsa"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--problems", type=str, default="", help="Optional comma-separated problem names, e.g. ZDT3 or ZDT1,DTLZ2.")
    parser.add_argument("--output", type=str, default="outputs/results.json")
    parser.add_argument("--plot-dir", type=str, default="outputs/plots")
    parser.add_argument("--table-output", type=str, default="outputs/summary_table.txt")
    parser.add_argument("--csv-output", type=str, default="outputs/summary_metrics.csv")
    args = parser.parse_args()

    config = BenchmarkConfig()
    config.runs = args.runs
    config.algorithm.iterations = args.iterations

    output_records: list[dict] = []
    table_rows: list[dict] = []
    plot_dir = Path(args.plot_dir)
    selected = {name.strip().upper() for name in args.problems.split(",") if name.strip()}
    problems = [problem for problem in paper_suite() if not selected or problem.name.upper() in selected]

    if not problems:
        raise SystemExit(f"No matching problems found for --problems={args.problems!r}")

    log(baseline_note())
    for problem_index, problem in enumerate(problems, start=1):
        log(f"Starting problem {problem_index}/{len(problems)}: {problem.name}")
        summary = run_momsa_benchmark(problem, config, log=log)
        log(format_summary(summary))
        output_records.append(serializable_summary(summary))
        table_rows.append(summary)

        plot_fronts = [("MOMSA", summary["plot_front"])]

        if args.include_baselines:
            baseline_summaries = run_pymoo_baselines(
                problem,
                n_pf_points=config.n_pf_points,
                iterations=args.iterations,
                runs=args.runs,
                log=log,
            )
            for baseline_summary in baseline_summaries:
                log(format_summary(baseline_summary))
                output_records.append(serializable_summary(baseline_summary))
                table_rows.append(baseline_summary)
                plot_fronts.append((baseline_summary["algorithm"].upper(), baseline_summary["plot_front"]))

        save_pareto_plot(problem.name, summary["true_front"], plot_fronts, plot_dir / f"{problem.name.lower()}_pareto.png")
        log(f"Saved Pareto plot to {plot_dir / f'{problem.name.lower()}_pareto.png'}")

    table_text = build_metric_table(table_rows)
    table_output = Path(args.table_output)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text(table_text + "\n")
    log(f"Saved summary table to {table_output}")
    print("\n" + table_text + "\n", flush=True)

    csv_output = Path(args.csv_output)
    write_csv(table_rows, csv_output)
    log(f"Saved summary CSV to {csv_output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_records, indent=2))
    log(f"Saved detailed JSON to {output_path}")


if __name__ == "__main__":
    main()
