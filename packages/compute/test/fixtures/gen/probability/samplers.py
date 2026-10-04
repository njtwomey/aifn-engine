"""Reference distributions for the aifn-compute/random sampler tests (scipy.stats).

Continuous samplers: quantiles at a probability grid (so the test compares the empirical cdf with F at 199 points, a
Kolmogorov–Smirnov statistic on a grid) and the exact mean and variance. Discrete samplers: the pmf over a range holding
all but about 1e-12 of the mass, for a chi-square goodness-of-fit test, and the exact mean and variance.
"""

from typing import Any, Protocol, cast

import numpy as np
from scipy import stats

PROBS = np.linspace(0.005, 0.995, 199)


class Frozen(Protocol):
    """The frozen-distribution methods the continuous cases use (scipy's rv_frozen, which is private)."""

    def ppf(self, q: Any) -> Any: ...
    def mean(self) -> Any: ...
    def var(self) -> Any: ...


class FrozenDiscrete(Frozen, Protocol):
    """Calling an rv_discrete is typed as returning either frozen kind, so the discrete cases cast to this."""

    def isf(self, q: Any) -> Any: ...
    def pmf(self, k: Any) -> Any: ...


def continuous(name: str, dist: Frozen, args: dict[str, float]) -> dict[str, object]:
    return {
        "name": name,
        "args": args,
        "p": PROBS,
        "x": dist.ppf(PROBS),
        "mean": dist.mean(),
        "var": dist.var(),
    }


def discrete(name: str, dist: FrozenDiscrete, args: dict[str, float]) -> dict[str, object]:
    lo = int(max(dist.ppf(1e-13) - 1, 0))
    hi = int(dist.isf(1e-13) + 1)
    k = np.arange(lo, hi + 1)
    return {"name": name, "args": args, "k": k, "pmf": dist.pmf(k), "mean": dist.mean(), "var": dist.var()}


def cases() -> dict[str, object]:
    cont = [
        continuous("uniform", stats.uniform(-1, 4), {"a": -1, "b": 3}),
        continuous("normal", stats.norm(1.5, 2), {"mean": 1.5, "sd": 2}),
        continuous("exponential", stats.expon(scale=1 / 0.7), {"rate": 0.7}),
        *[
            continuous(f"gamma {a}", stats.gamma(a, scale=2), {"shape": a, "scale": 2})
            for a in [0.05, 0.3, 1.0, 2.5, 30.0]
        ],
        *[
            continuous(f"beta {a} {b}", stats.beta(a, b), {"a": a, "b": b})
            for a, b in [(0.3, 0.5), (2.0, 5.0), (0.05, 0.08), (40.0, 10.0)]
        ],
        continuous("chiSquare 4", stats.chi2(4), {"df": 4}),
        continuous("chiSquare 0.5", stats.chi2(0.5), {"df": 0.5}),
        continuous("studentT 3", stats.t(3), {"df": 3, "loc": 0, "scale": 1}),
        continuous("studentT 0.8", stats.t(0.8, loc=2, scale=0.5), {"df": 0.8, "loc": 2, "scale": 0.5}),
        continuous("studentT 30", stats.t(30), {"df": 30, "loc": 0, "scale": 1}),
    ]
    disc = [
        *[
            discrete(f"poisson {lam}", cast(FrozenDiscrete, stats.poisson(lam)), {"lambda": lam})
            for lam in [0.5, 4.0, 9.9, 10.0, 25.0, 300.0]
        ],
        *[
            discrete(f"binomial {n} {p}", cast(FrozenDiscrete, stats.binom(n, p)), {"n": n, "p": p})
            for n, p in [(10, 0.3), (40, 0.9), (1000, 0.2), (1_000_000, 0.4), (100_000, 1e-4), (61, 0.5)]
        ],
    ]
    return {"continuous": cont, "discrete": disc}
