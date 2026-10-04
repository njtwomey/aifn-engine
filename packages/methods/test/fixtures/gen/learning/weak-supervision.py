"""Reference values for aifn-methods/learning/weak-supervision: Dawid-Skene expectation-maximisation written directly
in NumPy (majority-vote start, pseudo-count smoothing 0.01, the M-step before the E-step), on random crowd votes with
abstentions; the posteriors, priors, confusion matrices and log-likelihood after a fixed number of steps. Also the
anchor-point tests for class-conditional label noise (Poyiadzi et al. 2022; Yang et al. 2024): the logistic MLE and
its inverse Fisher information from statsmodels' Logit, and local likelihood logistic fits from statsmodels' Binomial
GLM with kernel variance weights and the HC0 sandwich covariance."""

from typing import Any

import numpy as np
import statsmodels.api as sm  # pyright: ignore[reportMissingTypeStubs]


def majority(votes: np.ndarray, k: int) -> np.ndarray:
    n = votes.shape[0]
    t = np.zeros((n, k))
    for i in range(n):
        counts = np.bincount(votes[i][votes[i] >= 0], minlength=k).astype(float)
        top = counts.max()
        t[i] = 1.0 / k if top == 0 else (counts == top) / np.sum(counts == top)
    return t


def em(votes: np.ndarray, k: int, steps: int, smoothing: float = 0.01):
    n, m = votes.shape
    t = majority(votes, k)
    ll = 0.0
    priors = np.zeros(k)
    conf = np.zeros((m, k, k))
    for _ in range(steps + 1):
        priors = t.sum(0) + smoothing
        priors /= priors.sum()
        conf = np.full((m, k, k), smoothing)
        for j in range(m):
            for i in range(n):
                if votes[i, j] >= 0:
                    conf[j, :, votes[i, j]] += t[i]
        conf /= conf.sum(axis=2, keepdims=True)
        lp = np.tile(np.log(priors), (n, 1))
        for j in range(m):
            cast = votes[:, j] >= 0
            lp[cast] += np.log(conf[j][:, votes[cast, j]].T)
        top = lp.max(axis=1, keepdims=True)
        z = np.exp(lp - top).sum(axis=1, keepdims=True)
        t = np.exp(lp - top) / z
        ll = float(np.sum(top + np.log(z)))
    return t, priors, conf, ll


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(8)
    n, m, k = 30, 5, 3
    y = rng.integers(0, k, size=n)
    votes = np.where(rng.uniform(size=(n, m)) < 0.7, y[:, None], rng.integers(0, k, size=(n, m)))
    votes = np.where(rng.uniform(size=(n, m)) < 0.3, -1, votes)
    out: list[dict[str, Any]] = []
    for steps in [0, 1, 10]:
        t, priors, conf, ll = em(votes, k, steps)
        out.append({"steps": steps, "posteriors": t, "priors": priors, "confusions": conf, "logLikelihood": ll})
    return {"votes": votes, "classes": k, "majority": majority(votes, k), "runs": out, "noise": noise_cases()}


def noise_cases() -> dict[str, Any]:
    """Noisy Gaussian-pair data (N([1, 1], I) vs N([-1, -1], I), labels flipped at alpha = 0, beta = 0.15)."""
    rng = np.random.default_rng(21)
    n = 300
    y = (np.arange(n) < n // 2).astype(int)
    x = np.where(y[:, None] == 1, 1.0, -1.0) + rng.standard_normal((n, 2))
    flip = rng.uniform(size=n) < np.where(y == 1, 0.0, 0.15)
    noisy = np.where(flip, 1 - y, y)
    u = rng.uniform(-4, 4, size=4)
    anchors = np.stack([u, -u], axis=1)
    design = sm.add_constant(x)  # pyright: ignore[reportUnknownMemberType]
    logit = sm.Logit(noisy, design).fit(disp=0)  # pyright: ignore[reportUnknownMemberType]
    theta = np.asarray(logit.params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    cov = np.asarray(logit.cov_params())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    aug = np.hstack([np.ones((4, 1)), anchors])
    eta = 1 / (1 + np.exp(-aug @ theta))
    bar = aug.mean(axis=0)
    v = float(bar @ cov @ bar) / 16
    z = (eta.mean() - 0.5) / np.sqrt(v)
    local: list[dict[str, Any]] = []
    for h, degree in [(1.0, 1), (0.7, 2)]:
        for a in np.array([[0.5, -0.5], [-1.0, 1.0]]):
            d = x - a
            w = np.exp(-0.5 * np.sum((d / h) ** 2, axis=1))
            cols = [np.ones(n), d[:, 0], d[:, 1]]
            if degree == 2:
                cols += [d[:, 0] ** 2 / 2, d[:, 0] * d[:, 1], d[:, 1] ** 2 / 2]
            basis = np.stack(cols, axis=1)
            glm = sm.GLM(noisy, basis, family=sm.families.Binomial(), var_weights=w)  # pyright: ignore[reportUnknownMemberType]
            fit = glm.fit(cov_type="HC0", tol=1e-12)  # pyright: ignore[reportUnknownMemberType]
            beta = np.asarray(fit.params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            sandwich = np.asarray(fit.cov_params())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            local.append({"bandwidth": h, "degree": degree, "at": a, "coefficients": beta, "covariance": sandwich})
    return {
        "x": x,
        "y": noisy,
        "anchors": anchors,
        "theta": theta,
        "fisherInverse": cov,
        "estimates": eta,
        "variance": v,
        "z": float(z),
        "local": local,
    }
