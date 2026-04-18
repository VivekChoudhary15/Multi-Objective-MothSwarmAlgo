from __future__ import annotations

from dataclasses import dataclass, field

from momsa.algorithms import MOMSAConfig
from momsa.problems import make_dtlz_suite, make_zdt_suite


@dataclass(slots=True)
class BenchmarkConfig:
    runs: int = 10
    n_pf_points: int = 300
    algorithm: MOMSAConfig = field(
        default_factory=lambda: MOMSAConfig(
            population_size=100,
            pathfinder_count=5,
            prospector_start_ratio=0.4,
            prospector_end_ratio=0.1,
            iterations=250,
            archive_size=100,
            seed=None,
        )
    )


def paper_suite():
    return make_zdt_suite() + make_dtlz_suite()
