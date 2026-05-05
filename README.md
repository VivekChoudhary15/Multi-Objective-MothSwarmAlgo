# MOMSA Benchmark and Portfolio Project

Python implementation and benchmark harness for the single-objective `MSA`
and the multi-objective `MOMSA` of:

> Sharifi, M. R., Akbarifard, S., Qaderi, K., Madadi, M. R.
> *A new optimization algorithm to solve multi-objective problems.*
> Scientific Reports, 11:20326, 2021. DOI: `10.1038/s41598-021-99617-x`

The project covers three layers:

1. **MSA** --- single-objective Moth Swarm Algorithm.
2. **MOMSA** --- multi-objective extension with archive and crowding-distance
   selection of pathfinders and the moonlight; benchmarked on the standard
   `ZDT` and `DTLZ` suites and compared against `NSGA-II`, `SPEA2`, and
   `MOEA/D`.
3. **Quant-finance application** --- MOMSA applied to long-only mean--variance
   portfolio selection on a synthetic equity market, validated against the
   closed-form Markowitz efficient frontier solved by SciPy SLSQP.

## Features

- Modular implementations of `MSA` and `MOMSA`.
- Built-in `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6` and `DTLZ1`, `DTLZ2`
  benchmark problems with their analytical Pareto fronts.
- Long-only mean--variance portfolio problem class with synthetic
  multi-sector equity-market generator and an SLSQP-based analytical
  Markowitz frontier for ground-truth comparison.
- Pareto-dominance, non-dominated filtering, crowding distance and archive
  truncation utilities.
- Multi-objective indicators: `GD`, `Spacing`, `Spread`, `Maximum Spread`.
- Benchmark runner with live progress logs, summary tables (text + CSV) and
  detailed JSON.
- Pareto-front plots (2D and 3D) plus a portfolio-frontier plot.
- Optional `pymoo` baselines (`NSGA-II`, `SPEA2`, `MOEA/D`).
- Two LaTeX project reports (article and IEEEtran formats) describing the
  implementation, benchmarks, and portfolio extension.

## Environment Setup

A dedicated virtual environment keeps the project isolated and reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .                    # core install (NumPy, SciPy, Matplotlib)
python -m pip install -e ".[baselines]"       # adds pymoo for NSGA-II / SPEA2 / MOEA/D
```

To leave the environment:

```bash
deactivate
```

**Python:** 3.11 or newer (declared in `pyproject.toml`).

**Required packages** (installed automatically by `pip install -e .`):
- `numpy >= 2.0` --- vector and matrix computation
- `scipy >= 1.11` --- `scipy.optimize.minimize` (SLSQP) for the analytical
  Markowitz frontier in the portfolio module
- `matplotlib >= 3.8` --- Pareto-front and efficient-frontier plotting

**Optional packages** (only needed for baselines and reports):
- `pymoo >= 0.6.1` --- `NSGA-II`, `SPEA2`, `MOEA/D` baseline implementations
- A LaTeX distribution (`texlive-latex-extra` or MiKTeX) --- only if you
  want to compile the project reports locally; otherwise upload the `.tex`
  files to Overleaf.

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── Project_Report.tex          # long-form report (article class)
├── report.pdf                  
├── s41598-021-99617-x-1.pdf    # MOMSA reference paper (PDF)
├── extracted.txt               # OCR-extracted plain text of the paper
├── scripts/
│   ├── run_msa_example.py      # single-objective MSA demo (sphere problem)
│   ├── run_benchmarks.py       # ZDT/DTLZ benchmark suite + baselines
│   └── run_portfolio.py        # quant-finance portfolio demo + plot
├── src/
│   └── momsa/
│       ├── __init__.py
│       ├── types.py            # Bounds, result dataclasses
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── msa.py          # single-objective MSA
│       │   └── momsa.py        # multi-objective MOMSA with archive
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── baselines.py    # pymoo NSGA-II/SPEA2/MOEA/D wrappers
│       │   ├── config.py       # default benchmark configuration
│       │   ├── runner.py       # repeated runs, aggregation, table output
│       │   └── visualization.py# Pareto-front PNG generation
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── indicators.py   # GD, Spacing, Spread, Maximum Spread
│       │   └── pareto.py       # dominance, crowding, truncation
│       └── problems/
│           ├── __init__.py
│           ├── base.py         # protocols + Sphere example
│           ├── dtlz.py         # DTLZ1, DTLZ2 (3-objective)
│           ├── zdt.py          # ZDT1..ZDT6 (bi-objective)
│           └── portfolio.py    # MeanVariancePortfolio + synthetic market
└── outputs/
    ├── results.json                   # detailed benchmark records
    ├── summary_metrics.csv            # spreadsheet-friendly summary
    ├── summary_table.txt              # text-rendered benchmark table
    ├── portfolio_summary.txt          # portfolio recovery comparison
    └── plots/
        ├── zdt1_pareto.png
        ├── zdt2_pareto.png
        ├── zdt3_pareto.png
        ├── zdt4_pareto.png
        ├── zdt6_pareto.png
        ├── dtlz1_pareto.png
        ├── dtlz2_pareto.png
        └── portfolio_frontier.png     # MOMSA + NSGA-II + analytical Markowitz
```

## File Guide

**Algorithms**
- `src/momsa/algorithms/msa.py` --- single-objective MSA: pathfinder
  (Levy + DE-style update), prospector (logarithmic spiral), onlooker
  (Gaussian walk and associative learning).
- `src/momsa/algorithms/momsa.py` --- multi-objective MOMSA: external
  non-dominated archive, crowding-distance-based selection of pathfinders
  and moonlight, NSGA-II-style elitist parent--offspring selection.

**Problems**
- `src/momsa/problems/base.py` --- problem protocols and a simple `Sphere`
  example for single-objective testing.
- `src/momsa/problems/zdt.py` --- the five bi-objective `ZDT` problems with
  their analytical Pareto fronts. ZDT4 uses the standard `[-5, 5]` bounds
  on `x_2,...,x_d`.
- `src/momsa/problems/dtlz.py` --- `DTLZ1` and `DTLZ2` (tri-objective).
- `src/momsa/problems/portfolio.py` --- `MeanVariancePortfolio` problem
  class + `make_synthetic_market` deterministic equity-market generator
  + analytical long-only Markowitz frontier solver via SLSQP.

**Metrics**
- `src/momsa/metrics/pareto.py` --- Pareto dominance, non-dominated mask,
  crowding distance, archive truncation.
- `src/momsa/metrics/indicators.py` --- generational distance, spacing,
  spread, maximum spread.

**Benchmarks**
- `src/momsa/benchmarks/runner.py` --- multi-seed runs, aggregation, table
  formatting, JSON serialization.
- `src/momsa/benchmarks/baselines.py` --- pymoo wrappers for NSGA-II,
  SPEA2, MOEA/D (gracefully skipped if pymoo is not installed).
- `src/momsa/benchmarks/visualization.py` --- 2D and 3D Pareto-front PNG
  plotting via Matplotlib.

**Scripts**
- `scripts/run_msa_example.py` --- single-objective MSA on the sphere
  problem (sanity check).
- `scripts/run_benchmarks.py` --- the main ZDT/DTLZ benchmark entry point.
- `scripts/run_portfolio.py` --- portfolio efficient-frontier recovery
  demo (MOMSA + analytical Markowitz + optional NSGA-II overlay).

**Reports**
- `Project_Report.tex` --- long-form project report (`article` class)
  covering problem statement, literature, methodology, novelty
  (portfolio extension), data, findings (with all 7 ZDT/DTLZ Pareto-front
  plots and the portfolio frontier), conclusion, acknowledgments,
  references.
- `Project_Report_IEEE.tex` --- the same content in IEEEtran conference
  format (2-column).
- `ieee_template.tex` --- the original IEEE template used as the basis
  for `Project_Report_IEEE.tex`.

## Quick Start

### 1. Single-objective MSA (sanity check)
```bash
python scripts/run_msa_example.py
```

### 2. ZDT/DTLZ benchmark suite (MOMSA only)
```bash
python scripts/run_benchmarks.py --runs 10 --iterations 250
```

### 3. ZDT/DTLZ with baseline comparison
```bash
python scripts/run_benchmarks.py --include-baselines --runs 10 --iterations 250
```

Both benchmark commands write `outputs/results.json`, `outputs/summary_table.txt`,
`outputs/summary_metrics.csv`, and 7 Pareto-front PNGs into `outputs/plots/`.

### 4. Portfolio efficient-frontier recovery
```bash
python scripts/run_portfolio.py --iterations 250 --n-assets 15
```

This runs MOMSA on a 15-asset synthetic equity market, computes the
analytical Markowitz efficient frontier with SLSQP, optionally overlays
NSGA-II if pymoo is installed, and writes:

- `outputs/portfolio_summary.txt` --- comparison table (points, min variance,
  max return, GD-vs-analytical) for analytical, MOMSA and NSGA-II.
- `outputs/plots/portfolio_frontier.png` --- the efficient-frontier figure.

Useful flags: `--n-assets`, `--n-days`, `--n-sectors`, `--seed`,
`--iterations`, `--pf-points`, `--no-nsga2`, `--output-plot`,
`--output-table`.

## Typical Benchmark Output

The benchmark runner prints progress logs while running, e.g.:

```text
[14:33:01] Starting problem 1/7: ZDT1
[14:33:01] [ZDT1] MOMSA run 1/10 started with seed=0
[14:33:05] [ZDT1] MOMSA run 1/10 finished in 4.23s (GD=0.00362, S=0.00725, Delta=0.397, MS=1.000)
...
[15:31:22] Saved detailed JSON to outputs/results.json
```

Final summary table (after all 7 problems × 10 seeds × 250 iters):

```
Problem | Algorithm | GD            | S            | Delta        | MS           | Archive        | Time(s)
--------+-----------+---------------+--------------+--------------+--------------+----------------+--------
ZDT1    | momsa     | 0.0036        | 0.0072       | 0.397        | 1.000        | 100.0          | 43.21
ZDT1    | nsga2     | 0.0035        | 0.0069       | 0.359        | 0.999        | 100.0          | 0.94
...
```

## Saved Artifacts

By default, benchmark and portfolio runs save:

- `outputs/results.json` --- detailed per-run records and summary statistics.
- `outputs/summary_table.txt` --- text-rendered benchmark metric table.
- `outputs/summary_metrics.csv` --- spreadsheet-friendly summary file.
- `outputs/plots/*.png` --- Pareto-front comparison figures (one per
  problem) plus the portfolio efficient-frontier plot.
- `outputs/portfolio_summary.txt` --- portfolio recovery comparison
  (analytical vs MOMSA vs NSGA-II).

## Metrics

The benchmark pipeline reports:

| Metric | Symbol | Direction |
|--------|--------|-----------|
| Generational Distance | `GD`    | lower is better |
| Spacing               | `S`     | lower is better |
| Spread                | `Delta` | lower is better |
| Maximum Spread        | `MS`    | higher is better |

For the portfolio script, the comparison metric is generational distance
of the recovered front against the SLSQP-resolved analytical Markowitz
efficient frontier (lower is better; zero would indicate exact recovery).

## Reproducibility Notes

- All experiments use fixed random seeds; pass `--seed N` to vary them.
- Metaheuristic algorithms are stochastic --- repeated runs across
  multiple seeds are essential. The benchmark runner defaults to 10 seeds.
- The synthetic equity market in `portfolio.py` is fully deterministic
  (seed 42 by default) with no external network call, so the portfolio
  results in `outputs/portfolio_summary.txt` and the frontier plot are
  reproducible bit-for-bit.
- For serious experiments, use fixed seeds, consistent iteration counts,
  and the same archive and population settings across algorithms.

## Acknowledgments

- Reference paper: Sharifi et al. (2021), Scientific Reports, DOI
  `10.1038/s41598-021-99617-x`.
- Benchmark suites: ZDT (Zitzler, Deb, Thiele 2000) and DTLZ
  (Deb, Thiele, Laumanns, Zitzler 2005).
- Baseline algorithms via [pymoo](https://pymoo.org/) (Blank & Deb, 2020).
- AI assistance: Claude (Anthropic) via Claude Code was used for code
  review, debugging, and report drafting; all final code and
  algorithmic decisions were verified by the author.
