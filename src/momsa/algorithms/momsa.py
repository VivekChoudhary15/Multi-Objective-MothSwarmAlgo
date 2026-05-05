from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from momsa.metrics import crowding_distance, filter_nondominated, non_dominated_mask, truncate_by_crowding
from momsa.types import Array, Bounds, MultiObjectiveResult

from .msa import MSA, MSAConfig


@dataclass(slots=True)
class MOMSAConfig(MSAConfig):
    archive_size: int = 100


class MOMSA(MSA):
    """Multi-objective MSA with archive and crowding distance."""

    def __init__(self, config: MOMSAConfig | None = None) -> None:
        super().__init__(config or MOMSAConfig())
        self.config: MOMSAConfig

    def optimize(self, problem) -> MultiObjectiveResult:
        bounds: Bounds = problem.bounds
        pop = self._initialize(bounds)
        scores = np.array([problem.evaluate(x) for x in pop], dtype=float)
        archive_x, archive_f = filter_nondominated(pop, scores)
        archive_x, archive_f = truncate_by_crowding(archive_x, archive_f, self.config.archive_size)
        history: list[int] = [len(archive_f)]

        for iteration in range(self.config.iterations):
            pathfinders, moonlight = self._select_guides(archive_x, archive_f, pop)
            prospector_count = self._prospector_count(iteration)
            pathfinder_count = len(pathfinders)
            onlooker_start = pathfinder_count + prospector_count

            ranked_idx = self._rank_population(scores)
            pop = pop[ranked_idx]
            scores = scores[ranked_idx]
            new_pop = pop.copy()

            for idx in range(pathfinder_count):
                new_pop[idx] = self._update_pathfinder(pop, idx, bounds, iteration)

            for idx in range(pathfinder_count, min(onlooker_start, len(pop))):
                target = pathfinders[(idx - pathfinder_count) % pathfinder_count]
                new_pop[idx] = self._update_prospector(pop[idx], target, bounds, iteration)

            for idx in range(min(onlooker_start, len(pop)), len(pop)):
                prospector = pop[pathfinder_count + (idx - onlooker_start) % max(prospector_count, 1)]
                new_pop[idx] = self._update_onlooker(pop[idx], moonlight, prospector, bounds, iteration)

            new_scores = np.array([problem.evaluate(x) for x in new_pop], dtype=float)

            # Elitist selection: combine parent + offspring, keep best N by
            # non-dominated rank with crowding-distance tie-breaking.
            combined_x = np.vstack([pop, new_pop])
            combined_f = np.vstack([scores, new_scores])
            pop, scores = self._select_population(
                combined_x, combined_f, self.config.population_size
            )

            merged_x = np.vstack([archive_x, pop])
            merged_f = np.vstack([archive_f, scores])
            archive_x, archive_f = filter_nondominated(merged_x, merged_f)
            archive_x, archive_f = truncate_by_crowding(archive_x, archive_f, self.config.archive_size)
            history.append(len(archive_f))

        return MultiObjectiveResult(archive_x=archive_x, archive_f=archive_f, history=history)

    def _rank_population(self, scores: Array) -> Array:
        mask = non_dominated_mask(scores)
        crowd = np.full(len(scores), -np.inf, dtype=float)
        if np.any(mask):
            crowd[mask] = crowding_distance(scores[mask])
        dominance_rank = np.where(mask, 0, 1)
        return np.lexsort((-crowd, dominance_rank))

    def _select_population(self, xs: Array, fs: Array, n: int) -> tuple[Array, Array]:
        """NSGA-II-style truncation: fill ranks until full, last rank by crowding."""
        if len(fs) <= n:
            return xs, fs

        selected_x: list[Array] = []
        selected_f: list[Array] = []
        remaining = np.arange(len(fs))
        while len(remaining) > 0 and len(selected_x) < n:
            sub_f = fs[remaining]
            mask = non_dominated_mask(sub_f)
            layer = remaining[mask]
            slots = n - len(selected_x)
            if len(layer) <= slots:
                for i in layer:
                    selected_x.append(xs[i])
                    selected_f.append(fs[i])
            else:
                crowd = crowding_distance(fs[layer])
                order = np.argsort(-crowd)
                for i in layer[order[:slots]]:
                    selected_x.append(xs[i])
                    selected_f.append(fs[i])
            remaining = remaining[~mask]

        return np.asarray(selected_x, dtype=float), np.asarray(selected_f, dtype=float)

    def _select_guides(self, archive_x: Array, archive_f: Array, fallback_pop: Array) -> tuple[Array, Array]:
        if len(archive_x) == 0:
            return fallback_pop[: self.config.pathfinder_count], fallback_pop[0]

        crowd = crowding_distance(archive_f)
        order = np.argsort(-crowd)
        pathfinders = archive_x[order[: max(1, min(self.config.pathfinder_count, len(archive_x)))]]
        moonlight = pathfinders[0]
        return pathfinders, moonlight
