from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from momsa.types import Array, Bounds, SingleObjectiveResult


@dataclass(slots=True)
class MSAConfig:
    population_size: int = 40
    pathfinder_count: int = 3
    prospector_start_ratio: float = 0.4
    prospector_end_ratio: float = 0.1
    iterations: int = 200
    levy_scale: float = 0.01
    spiral_b: float = 1.0
    gaussian_scale: float = 0.1
    learning_rate: float = 0.6
    seed: int | None = None


class MSA:
    """Paper-inspired single-objective Moth Swarm Algorithm."""

    def __init__(self, config: MSAConfig | None = None) -> None:
        self.config = config or MSAConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def optimize(self, problem) -> SingleObjectiveResult:
        bounds: Bounds = problem.bounds
        pop = self._initialize(bounds)
        fitness = np.array([problem.evaluate(x) for x in pop], dtype=float)
        history: list[float] = []

        for iteration in range(self.config.iterations):
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]
            history.append(float(fitness[0]))

            pathfinder_count = min(self.config.pathfinder_count, len(pop))
            prospector_count = self._prospector_count(iteration)
            onlooker_start = pathfinder_count + prospector_count

            moonlight = pop[0].copy()
            pathfinders = pop[:pathfinder_count].copy()

            new_pop = pop.copy()

            for idx in range(pathfinder_count):
                new_pop[idx] = self._update_pathfinder(pop, idx, bounds, iteration)

            for idx in range(pathfinder_count, onlooker_start):
                target = pathfinders[(idx - pathfinder_count) % pathfinder_count]
                new_pop[idx] = self._update_prospector(pop[idx], target, bounds, iteration)

            for idx in range(onlooker_start, len(pop)):
                prospector = pop[pathfinder_count + (idx - onlooker_start) % max(prospector_count, 1)]
                new_pop[idx] = self._update_onlooker(pop[idx], moonlight, prospector, bounds, iteration)

            new_fitness = np.array([problem.evaluate(x) for x in new_pop], dtype=float)
            improved = new_fitness < fitness
            pop[improved] = new_pop[improved]
            fitness[improved] = new_fitness[improved]

        best = int(np.argmin(fitness))
        return SingleObjectiveResult(x_best=pop[best], f_best=float(fitness[best]), history=history)

    def _initialize(self, bounds: Bounds) -> Array:
        u = self.rng.random((self.config.population_size, bounds.dim))
        return bounds.lower + u * (bounds.upper - bounds.lower)

    def _prospector_count(self, iteration: int) -> int:
        progress = iteration / max(1, self.config.iterations - 1)
        ratio = (1.0 - progress) * self.config.prospector_start_ratio + progress * self.config.prospector_end_ratio
        return max(1, int(self.config.population_size * ratio))

    def _levy_step(self, dim: int) -> Array:
        beta = 1.5
        sigma_u = (
            math.gamma(1.0 + beta)
            * np.sin(np.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)
        u = self.rng.normal(0.0, sigma_u, dim)
        v = self.rng.normal(0.0, 1.0, dim)
        return u / (np.abs(v) ** (1.0 / beta) + 1e-12)

    def _update_pathfinder(self, pop: Array, idx: int, bounds: Bounds, iteration: int) -> Array:
        available = np.delete(np.arange(len(pop)), idx)
        ids = self.rng.choice(available, size=5, replace=False)
        base = pop[ids[0]]
        diff = self.rng.random() * (pop[ids[1]] - pop[ids[2]]) + self.rng.random() * (pop[ids[3]] - pop[ids[4]])
        step = self.config.levy_scale * self._levy_step(bounds.dim)
        progress = 1.0 - iteration / max(1, self.config.iterations - 1)
        candidate = base + progress * diff + step
        return bounds.clip(candidate)

    def _update_prospector(self, moth: Array, target: Array, bounds: Bounds, iteration: int) -> Array:
        # Eq. (8): x_new = |x_target - x_moth| * e^(b*t) * cos(2*pi*t) + x_target
        distance = np.abs(target - moth)
        t = self.rng.uniform(-1.0, 1.0)
        spiral = np.exp(self.config.spiral_b * t) * np.cos(2.0 * np.pi * t)
        candidate = target + spiral * distance
        return bounds.clip(candidate)

    def _update_onlooker(self, moth: Array, moonlight: Array, prospector: Array, bounds: Bounds, iteration: int) -> Array:
        progress = iteration / max(1, self.config.iterations - 1)
        if self.rng.random() < 0.5:
            sigma = self.config.gaussian_scale * (1.0 - 0.7 * progress)
            candidate = prospector + self.rng.normal(0.0, sigma, bounds.dim) * (moonlight - moth)
        else:
            # Eq. (11): associative learning with a small immigration term
            low = bounds.lower - moth
            high = bounds.upper - moth
            immigration = 0.001 * (low + self.rng.random(bounds.dim) * (high - low))
            r1 = self.rng.random(bounds.dim)
            r2 = self.rng.random(bounds.dim)
            candidate = (
                moth
                + immigration
                + self.config.learning_rate * r1 * (moonlight - moth)
                + (1.0 - self.config.learning_rate) * r2 * (prospector - moth)
            )
        return bounds.clip(candidate)
