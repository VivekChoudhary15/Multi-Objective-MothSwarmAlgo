from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from momsa.types import Array, Bounds


def make_synthetic_market(
    n_assets: int = 15,
    n_days: int = 504,
    n_sectors: int = 3,
    seed: int = 42,
) -> tuple[Array, Array, Array, Array]:
    """Generate a deterministic synthetic equity market.

    Returns (daily_returns, mu_annual, cov_annual, sector_labels).

    The covariance has a block-correlation structure: assets in the same sector
    are correlated at ~0.55, across sectors at ~0.10. Daily returns are drawn
    from a multivariate normal with these annualized parameters scaled to 252
    trading days.
    """
    rng = np.random.default_rng(seed)

    sector = np.repeat(np.arange(n_sectors), int(np.ceil(n_assets / n_sectors)))[:n_assets]

    mu_annual = rng.uniform(0.05, 0.22, n_assets)
    vol_annual = rng.uniform(0.15, 0.35, n_assets)

    corr = np.full((n_assets, n_assets), 0.10)
    for s in range(n_sectors):
        idx = np.where(sector == s)[0]
        for i in idx:
            for j in idx:
                corr[i, j] = 0.55
    np.fill_diagonal(corr, 1.0)

    cov_annual = corr * np.outer(vol_annual, vol_annual)

    mu_daily = mu_annual / 252.0
    cov_daily = cov_annual / 252.0

    L = np.linalg.cholesky(cov_daily)
    z = rng.standard_normal((n_days, n_assets))
    daily_returns = mu_daily + z @ L.T

    return daily_returns, mu_annual, cov_annual, sector


def _normalize_weights(x: Array) -> Array:
    """Project a non-negative vector onto the unit simplex via L1 normalization."""
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 1e-12:
        return np.ones_like(x) / len(x)
    return x / s


@dataclass(slots=True)
class MeanVariancePortfolio:
    """Long-only mean-variance portfolio (two objectives).

    Decision variable x in [0, 1]^n is L1-normalized to portfolio weights w.
    Objective 1: portfolio variance  (minimize)
    Objective 2: -expected return    (minimize -> maximize return)
    """

    expected_returns: Array
    covariance: Array
    name: str = "MeanVariancePortfolio"
    n_obj: int = field(init=False)
    bounds: Bounds = field(init=False)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.expected_returns).shape[0])
        self.n_obj = 2
        self.bounds = Bounds(
            lower=np.zeros(n, dtype=float),
            upper=np.ones(n, dtype=float),
        )

    @property
    def n_assets(self) -> int:
        return self.bounds.dim

    def weights(self, x: Array) -> Array:
        return _normalize_weights(np.asarray(x, dtype=float))

    def evaluate(self, x: Array) -> Array:
        w = self.weights(x)
        ret = float(self.expected_returns @ w)
        var = float(w @ self.covariance @ w)
        return np.array([var, -ret], dtype=float)

    def pareto_front(self, n_points: int = 80) -> Array:
        """Analytical long-only efficient frontier via SLSQP (one QP per target return)."""
        from scipy.optimize import minimize

        n = self.n_assets
        lo = float(self.expected_returns.min())
        hi = float(self.expected_returns.max())
        # Pull endpoints in slightly to keep the SLSQP feasibility margin healthy.
        target_returns = np.linspace(lo + 1e-4, hi - 1e-4, n_points)

        bnds = [(0.0, 1.0)] * n
        front: list[list[float]] = []

        for tr in target_returns:
            cons = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w, t=tr: float(self.expected_returns @ w) - t},
            )
            x0 = np.ones(n) / n
            res = minimize(
                lambda w: float(w @ self.covariance @ w),
                x0,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if not res.success:
                continue
            w = res.x
            front.append([float(w @ self.covariance @ w), -float(self.expected_returns @ w)])

        if not front:
            return np.empty((0, 2), dtype=float)
        return np.asarray(front, dtype=float)


def make_portfolio_problem(
    n_assets: int = 15,
    n_days: int = 504,
    n_sectors: int = 3,
    seed: int = 42,
) -> tuple[MeanVariancePortfolio, Array, Array]:
    """Convenience builder: returns (problem, daily_returns, sector_labels)."""
    daily_returns, mu_annual, cov_annual, sector = make_synthetic_market(
        n_assets=n_assets, n_days=n_days, n_sectors=n_sectors, seed=seed
    )
    problem = MeanVariancePortfolio(expected_returns=mu_annual, covariance=cov_annual)
    return problem, daily_returns, sector
