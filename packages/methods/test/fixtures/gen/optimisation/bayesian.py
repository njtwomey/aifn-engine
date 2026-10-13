"""Golden values for aifn-methods optimisation/bayesian: the acquisition functions in closed form from scipy.stats.norm.

- Probability of improvement, expected improvement and the confidence bound, for minimisation and maximisation, with
  and without a margin xi, including points with sd = 0 (the point-mass limits).
- The GP-UCB weight beta_t of Srinivas et al. (2010), Theorem 1.
- Max-value entropy search (Wang and Jegelka, 2017, eq. 6) for given samples of the optimum.
- The Gumbel fit of the max-value samples: the quartiles and median of prod_i Phi((y - mu_i) / sd_i), found by brentq,
  give the scale b and location a.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MU = np.array([-1.5, -0.4, 0.0, 0.2, 0.7, 1.3, 2.5, -0.1, 0.3])
SD = np.array([0.3, 1.2, 0.5, 0.05, 2.0, 0.8, 1.0, 0.0, 0.0])
BEST = 0.1


def improvement(mu: np.ndarray, best: float, xi: float, minimise: bool) -> np.ndarray:
    """The mean improvement delta over the incumbent, less the margin."""
    return best - mu - xi if minimise else mu - best - xi


def pi(mu: np.ndarray, sd: np.ndarray, best: float, xi: float, minimise: bool) -> np.ndarray:
    """Probability of improvement, with the point-mass limit where sd = 0."""
    delta = improvement(mu, best, xi, minimise)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = delta / sd
    return np.where(sd > 0, norm.cdf(z), (delta > 0).astype(float))


def ei(mu: np.ndarray, sd: np.ndarray, best: float, xi: float, minimise: bool) -> np.ndarray:
    """Expected improvement, with the point-mass limit where sd = 0."""
    delta = improvement(mu, best, xi, minimise)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = delta / sd
        value = delta * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(sd > 0, np.maximum(value, 0.0), np.maximum(delta, 0.0))


def ucb(mu: np.ndarray, sd: np.ndarray, beta: float, minimise: bool) -> np.ndarray:
    """The confidence bound, negated lower bound when minimising."""
    return (-mu if minimise else mu) + np.sqrt(beta) * sd


def mes(mu: np.ndarray, sd: np.ndarray, samples: np.ndarray, minimise: bool) -> np.ndarray:
    """Max-value entropy search for given samples of the optimum; 0 where sd = 0."""
    m = -mu if minimise else mu
    ys = -samples if minimise else samples
    out = np.zeros_like(mu)
    for i in range(len(mu)):
        if sd[i] == 0:
            continue
        g = (ys - m[i]) / sd[i]
        out[i] = max(np.mean(g * np.exp(norm.logpdf(g) - norm.logcdf(g)) / 2 - norm.logcdf(g)), 0.0)
    return out


def gumbel(mu: np.ndarray, sd: np.ndarray, minimise: bool) -> dict[str, float]:
    """The Gumbel location and scale matched to the product of the marginal cdfs at its quartiles and median."""
    m = -mu if minimise else mu

    def log_cdf(y: float) -> float:
        return float(np.sum(norm.logcdf((y - m) / sd)))

    lo, hi = np.max(m - 5 * sd), np.max(m + 5 * sd)
    q = [
        brentq(lambda y, p=p: log_cdf(y) - np.log(p), lo - 10, hi + 10, xtol=1e-14, rtol=1e-15)
        for p in (0.25, 0.5, 0.75)
    ]
    scale = (q[2] - q[0]) / (np.log(-np.log(0.25)) - np.log(-np.log(0.75)))
    location = q[1] + scale * np.log(-np.log(0.5))
    return {"minimise": minimise, "location": location, "scale": scale, "quantiles": q}


def cases() -> dict[str, object]:
    acquisitions = []
    for minimise in (True, False):
        for xi in (0.0, 0.25):
            acquisitions.append(
                {
                    "minimise": minimise,
                    "xi": xi,
                    "pi": pi(MU, SD, BEST, xi, minimise),
                    "ei": ei(MU, SD, BEST, xi, minimise),
                }
            )
    bounds = [
        {"minimise": minimise, "beta": beta, "ucb": ucb(MU, SD, beta, minimise)}
        for minimise in (True, False)
        for beta in (0.0, 1.0, 4.0, 9.5)
    ]
    betas = [
        {"t": t, "candidates": c, "delta": d, "beta": 2 * np.log(c * t**2 * np.pi**2 / (6 * d))}
        for t, c, d in ((1, 201, 0.1), (10, 201, 0.1), (100, 1681, 0.05))
    ]
    samples = np.array([-2.1, -1.7, -1.55, -2.6])
    entropy = [
        {"minimise": True, "samples": samples, "mes": mes(MU, SD, samples, True)},
        {"minimise": False, "samples": -samples + 1, "mes": mes(MU, SD, -samples + 1, False)},
    ]
    grid_mu = np.sin(np.linspace(0, 3, 31)) + 0.2 * np.linspace(0, 3, 31)
    grid_sd = 0.1 + 0.3 * np.abs(np.cos(np.linspace(0, 5, 31)))
    gumbels = [gumbel(grid_mu, grid_sd, minimise) for minimise in (True, False)]
    return {
        "mu": MU,
        "sd": SD,
        "best": BEST,
        "acquisitions": acquisitions,
        "bounds": bounds,
        "betas": betas,
        "entropy": entropy,
        "gumbel": {"mu": grid_mu, "sd": grid_sd, "fits": gumbels},
    }
