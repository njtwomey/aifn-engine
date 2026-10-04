"""Reference values for aifn-compute/probability/privacy: Opacus's RDP accountant (`compute_rdp` and `get_privacy_spent`
over
its default orders) for the subsampled Gaussian mechanism, and the analytic Gaussian mechanism's sigma (Balle and Wang,
2018) by scipy root finding on the exact privacy profile."""

import math
from typing import Any

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from scipy.optimize import brentq
from scipy.stats import norm


def gaussian_delta(sensitivity: float, sigma: float, epsilon: float) -> float:
    a = sensitivity / (2 * sigma)
    b = epsilon * sigma / sensitivity
    return float(norm.cdf(a - b) - math.exp(epsilon) * norm.cdf(-a - b))


def cases() -> dict[str, Any]:
    orders = RDPAccountant.DEFAULT_ALPHAS
    accounting: list[dict[str, Any]] = []
    for q, sigma, steps, delta in [
        (0.01, 1.1, 1000, 1e-5),
        (0.004, 0.8, 5000, 1e-5),
        (0.1, 2.0, 200, 1e-6),
        (1.0, 4.0, 10, 1e-5),
        (0.05, 0.6, 50, 1e-3),
    ]:
        rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
        eps, best = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        accounting.append(
            {
                "q": q,
                "sigma": sigma,
                "steps": steps,
                "delta": delta,
                "rdp": np.asarray(rdp),
                "epsilon": float(eps),
                "order": float(best),
            }
        )
    analytic: list[dict[str, Any]] = []
    for sensitivity, epsilon, delta in [(1.0, 0.5, 1e-5), (1.0, 1.0, 1e-6), (2.0, 3.0, 1e-5), (1.0, 8.0, 1e-9)]:

        def gap(s: float, sensitivity: float = sensitivity, epsilon: float = epsilon, delta: float = delta) -> float:
            return gaussian_delta(sensitivity, s, epsilon) - delta

        sigma: float = brentq(gap, 1e-6, 1e6, xtol=1e-14, rtol=1e-14)  # pyright: ignore[reportAssignmentType, reportArgumentType]
        analytic.append({"sensitivity": sensitivity, "epsilon": epsilon, "delta": delta, "sigma": float(sigma)})
    return {"orders": orders, "accounting": accounting, "analytic": analytic}
