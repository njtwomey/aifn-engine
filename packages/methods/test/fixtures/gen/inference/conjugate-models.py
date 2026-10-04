"""Golden values for aifn-methods/inference/conjugate-models: the normal-gamma log evidence by numerical
integration (scipy), and data sets."""

from typing import cast

import numpy as np
from scipy import integrate, stats


def cases() -> dict[str, object]:
    rng = np.random.default_rng(7)
    x = rng.normal(1.5, 0.8, size=12)
    mu0, l0, a0, b0 = 0.5, 2.0, 1.5, 0.7

    def joint(mu: float, tau: float) -> float:
        prior = stats.gamma.pdf(tau, a0, scale=1 / b0) * stats.norm.pdf(mu, mu0, 1 / np.sqrt(l0 * tau))
        return float(prior * np.prod(stats.norm.pdf(x, mu, 1 / np.sqrt(tau))))

    # The integrand is scaled by exp(c) to stay in floating-point range.
    c = 20.0
    # dblquad's return type also covers full_output, which is not requested: the result is (value, error).
    z, _ = cast(
        tuple[float, float],
        integrate.dblquad(lambda mu, tau: joint(mu, tau) * np.exp(c), 1e-6, 12.0, -3.0, 6.0, epsabs=0, epsrel=1e-10),
    )
    return {
        "normal_gamma": {
            "x": x,
            "prior": {"mean": mu0, "precisionScale": l0, "shape": a0, "rate": b0},
            "log_evidence": float(np.log(z) - c),
        }
    }
