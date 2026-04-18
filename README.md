# MOMSA Benchmark Project

Python reference implementation and benchmark harness for the single-objective `MSA` and the multi-objective `MOMSA` described in the paper:

Sharifi, M. R., Akbarifard, S., Qaderi, K., and Madadi, M. R.  
`A new optimization algorithm to solve multi-objective problems`  
Scientific Reports, 2021. DOI: `10.1038/s41598-021-99617-x`

This repository is organized as a small research-style codebase:
- implement `MSA` first
- extend it into `MOMSA`
- evaluate `MOMSA` on standard `ZDT` and `DTLZ` benchmark suites
- compare against baseline algorithms such as `NSGA-II`, `SPEA2`, and `MOEA/D`

**Features**
- modular implementations of `MSA` and `MOMSA`
- built-in `ZDT` and `DTLZ` benchmark problems
- Pareto and multi-objective evaluation utilities
- benchmark runner with live progress logs
- summary tables in text and CSV formats
- Pareto-front plots for 2-objective and 3-objective problems
- optional `pymoo` baselines for standard comparisons

**Environment Setup**
Use a dedicated virtual environment so the project remains isolated and reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pymoo
```

To leave the environment:

```bash
deactivate
```

**Repository Structure**
```text
.
├── README.md
├── pyproject.toml
├── scripts/
│   ├── run_msa_example.py
│   └── run_benchmarks.py
├── src/
│   └── momsa/
│       ├── __init__.py
│       ├── types.py
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── msa.py
│       │   └── momsa.py
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── baselines.py
│       │   ├── config.py
│       │   ├── runner.py
│       │   └── visualization.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── indicators.py
│       │   └── pareto.py
│       └── problems/
│           ├── __init__.py
│           ├── base.py
│           ├── dtlz.py
│           └── zdt.py
└── outputs/
    ├── results.json
    ├── summary_metrics.csv
    ├── summary_table.txt
    └── plots/
```

**File Guide**
- `README.md`: project overview, setup, usage, and outputs.
- `pyproject.toml`: package metadata and Python dependencies.
- `scripts/run_msa_example.py`: small single-objective demo run for `MSA`.
- `scripts/run_benchmarks.py`: main benchmark entry point for `MOMSA` and baseline comparisons.
- `src/momsa/__init__.py`: top-level package exports.
- `src/momsa/types.py`: shared dataclasses and array-related helper types.
- `src/momsa/algorithms/msa.py`: single-objective `MSA` implementation.
- `src/momsa/algorithms/momsa.py`: multi-objective `MOMSA` implementation with archive and crowding-distance logic.
- `src/momsa/problems/base.py`: basic problem protocols and a simple `Sphere` example.
- `src/momsa/problems/zdt.py`: `ZDT` benchmark problem definitions used in multi-objective testing.
- `src/momsa/problems/dtlz.py`: `DTLZ` benchmark problem definitions for tri-objective evaluation.
- `src/momsa/metrics/pareto.py`: Pareto dominance, non-dominated filtering, crowding distance, and archive truncation utilities.
- `src/momsa/metrics/indicators.py`: benchmark metrics such as `GD`, `Spacing`, `Spread`, and `Maximum Spread`.
- `src/momsa/benchmarks/config.py`: default benchmark configuration values.
- `src/momsa/benchmarks/runner.py`: repeated-run execution, summary aggregation, and formatted result tables.
- `src/momsa/benchmarks/baselines.py`: optional `pymoo` baseline integration for `NSGA-II`, `SPEA2`, and `MOEA/D`.
- `src/momsa/benchmarks/visualization.py`: Pareto-front plot generation for saved figures.
- `outputs/`: generated artifacts from benchmark runs.

**Quick Start**
Run a simple single-objective example:

```bash
python scripts/run_msa_example.py
```

Run the paper-style benchmark suite for `MOMSA` only:

```bash
python scripts/run_benchmarks.py --runs 10 --iterations 250
```

Run the same suite with standard baseline comparisons:

```bash
python scripts/run_benchmarks.py --include-baselines --runs 10 --iterations 250
```

**Typical Benchmark Output**
The benchmark runner prints progress logs while running, for example:

```text
[17:44:20] Starting problem 1/7: ZDT1
[17:44:20] [ZDT1] MOMSA run 1/10 started with seed=0
[17:44:21] [ZDT1] MOMSA run 1/10 finished in 0.03s (GD=..., S=..., Delta=..., MS=...)
```

**Saved Artifacts**
By default, benchmark runs save:

- `outputs/results.json`: detailed per-run records and summary statistics
- `outputs/summary_table.txt`: terminal-style metric table
- `outputs/summary_metrics.csv`: spreadsheet-friendly summary file
- `outputs/plots/*.png`: Pareto-front comparison figures

**Metrics**
The benchmark pipeline reports these metrics :

- `GD`: Generational Distance
- `S`: Spacing
- `Delta`: Spread
- `MS`: Maximum Spread

Lower values are generally better for `GD`, `S`, and `Delta`, while higher values are typically better for `MS`.

**Reproducibility Notes**
- The current implementation is paper-inspired and modular, intended for transparent experimentation and further refinement.
- Metaheuristic algorithms are stochastic, so repeated runs with multiple seeds are essential.
- Results from Python will not necessarily match MATLAB line-by-line, but they should be compared through the same benchmark problems and evaluation metrics.
- For serious experiments, use fixed seeds, consistent iteration counts, and the same archive/population settings across algorithms.
