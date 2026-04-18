from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from momsa.algorithms import MSA, MSAConfig
from momsa.problems import SphereProblem


def main() -> None:
    problem = SphereProblem(dim=30)
    solver = MSA(MSAConfig(population_size=40, iterations=150, seed=7))
    result = solver.optimize(problem)
    print(f"Problem: {problem.name}")
    print(f"Best objective: {result.f_best:.8f}")
    print(f"Best vector first 5 dims: {result.x_best[:5]}")


if __name__ == "__main__":
    main()
