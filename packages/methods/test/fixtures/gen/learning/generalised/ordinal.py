"""Golden values for aifn-methods/learning/generalised/ordinal (and the GP ordinal model of
aifn-methods/learning/gaussian-processes), each written out directly in numpy/scipy:

- maximum-likelihood fits of the cumulative-link (logit and probit), continuation-ratio and adjacent-category models,
  minimised by scipy's BFGS in (β, θ); the cumulative model's θ is kept increasing through θ₁ = u₁,
  θ_k = θ_{k−1} + exp(u_k), the other two models' thresholds are free;
- threshold-loss fits (Rennie & Srebro, 2005): Σ loss(xᵀw, y, θ) + ½λ‖w‖² by BFGS with free θ. The all-threshold
  optimum is ordered (Chu & Keerthi, 2005), so it equals aifn's fit with ordered thresholds; the immediate-threshold
  optimum is not on these data (class 1 is sparse), and is compared with aifn's fit with `ordered: false`;
- Frank & Hall's binary decomposition with scikit-learn's L2 logistic regression (C = 1/λ) per split;
- GP ordinal regression (Chu & Ghahramani, 2005) by Laplace at fixed hyperparameters: Newton for the mode with the
  closed-form gradient and curvature of the cumulative probit likelihood, the evidence, and predictions."""

from typing import Any

import numpy as np
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression

CDF = {"logit": special.expit, "probit": stats.norm.cdf}


def probabilities(model: str, link: str, eta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    F = CDF[link]
    z = theta[None, :] - eta[:, None]
    if model == "cumulative":
        c = np.column_stack([np.zeros(len(eta)), F(z), np.ones(len(eta))])
        return np.diff(c, axis=1)
    if model == "continuation-ratio":
        stop = F(z)
        go = np.cumprod(np.column_stack([np.ones(len(eta)), 1 - stop]), axis=1)
        return np.column_stack([stop, np.ones(len(eta))]) * go
    # adjacent-category: log P(k) ∝ Σ_{j<k} (η − θ_j)
    s = np.column_stack([np.zeros(len(eta)), np.cumsum(-z, axis=1)])
    return special.softmax(s, axis=1)


def fit(model: str, link: str, x: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    d = x.shape[1]

    def unpack(w):
        beta = w[:d]
        u = w[d:]
        theta = np.cumsum(np.r_[u[0], np.exp(u[1:])]) if model == "cumulative" else u
        return beta, theta

    def nll(w):
        beta, theta = unpack(w)
        p = probabilities(model, link, x @ beta, theta)
        with np.errstate(divide="ignore"):
            return -np.sum(np.log(p[np.arange(len(y)), y]))

    w0 = np.r_[np.zeros(d), -1.0, np.zeros(k - 2)]
    res = optimize.minimize(nll, w0, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
    beta, theta = unpack(res.x)
    return {"beta": beta, "theta": theta, "loglik": -res.fun}


def penalty(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "logistic":
        return np.logaddexp(0.0, -z)
    if kind == "smooth-hinge":
        return np.where(z >= 1, 0.0, np.where(z <= 0, 0.5 - z, 0.5 * (1 - z) ** 2))
    if kind == "hinge":
        return np.maximum(0.0, 1 - z)
    return np.maximum(0.0, 1 - z) ** 2


def threshold_fit(construction: str, kind: str, x: np.ndarray, y: np.ndarray, k: int, l2: float) -> dict[str, object]:
    n, d = x.shape
    ls = np.arange(k - 1)[None, :]
    sign = np.where(ls < y[:, None], 1.0, -1.0)
    mask = np.ones((n, k - 1)) if construction == "all" else ((ls == y[:, None] - 1) | (ls == y[:, None])) * 1.0

    def objective(v):
        w, theta = v[:d], v[d:]
        z = sign * ((x @ w)[:, None] - theta[None, :])
        return np.sum(mask * penalty(z, kind)) + 0.5 * l2 * w @ w

    v0 = np.r_[np.zeros(d), np.linspace(-1, 1, k - 1)]
    res = optimize.minimize(objective, v0, method="BFGS", options={"gtol": 1e-11, "maxiter": 20000})
    w, theta = res.x[:d], res.x[d:]
    assert construction == "immediate" or np.all(np.diff(theta) > 0)
    return {"w": w, "theta": theta, "objective": res.fun}


def frank_hall(x: np.ndarray, y: np.ndarray, k: int, xq: np.ndarray, l2: float) -> dict[str, object]:
    q = np.column_stack(
        [
            LogisticRegression(C=1 / l2, tol=1e-12, max_iter=10000).fit(x, (y > j).astype(int)).predict_proba(xq)[:, 1]
            for j in range(k - 1)
        ]
    )
    c = np.column_stack([np.ones(len(xq)), q, np.zeros(len(xq))])
    p = np.maximum(c[:, :-1] - c[:, 1:], 0)
    return {"q": q, "proba": p / p.sum(axis=1, keepdims=True)}


def gp_ordinal(x: np.ndarray, y: np.ndarray, xq: np.ndarray, theta: np.ndarray, sigma: float) -> dict[str, object]:
    """Laplace for the cumulative probit likelihood (Chu & Ghahramani, 2005, eqs. 9-14); unit RBF kernel."""
    edges = np.r_[-np.inf, theta, np.inf]

    def rbf(a, b):
        return np.exp(-0.5 * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

    K = rbf(x, x)
    n = len(y)

    def terms(f):
        z1 = (edges[y + 1] - f) / sigma
        z2 = (edges[y] - f) / sigma
        logp = np.log(stats.norm.cdf(z1) - stats.norm.cdf(z2))
        n1 = np.where(np.isfinite(z1), stats.norm.pdf(z1), 0.0)
        n2 = np.where(np.isfinite(z2), stats.norm.pdf(z2), 0.0)
        zn1 = np.where(np.isfinite(z1), z1, 0.0) * n1
        zn2 = np.where(np.isfinite(z2), z2, 0.0) * n2
        Z = np.exp(logp)
        r = (n1 - n2) / Z
        grad = -r / sigma
        W = (r**2 + (zn1 - zn2) / Z) / sigma**2
        return logp.sum(), grad, W

    f = np.zeros(n)
    obj = -np.inf
    for _ in range(200):
        _, g, W = terms(f)
        sW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
        b = W * f + g
        a = b - sW * np.linalg.solve(L.T, np.linalg.solve(L, sW * (K @ b)))
        f = K @ a
        ll, g, W = terms(f)
        new = ll - 0.5 * a @ f
        if abs(new - obj) < 1e-14:
            break
        obj = new
    ll, g, W = terms(f)
    sW = np.sqrt(W)
    L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
    log_marginal = ll - 0.5 * f @ np.linalg.solve(K, f) - np.log(np.diag(L)).sum()
    Ks = rbf(x, xq)
    mean = Ks.T @ g
    v = np.linalg.solve(L, sW[:, None] * Ks)
    var = 1.0 - (v**2).sum(0)
    s = np.sqrt(sigma**2 + var)
    c = stats.norm.cdf((edges[None, :] - mean[:, None]) / s[:, None])
    return {"mode": f, "logMarginal": log_marginal, "mean": mean, "variance": var, "proba": np.diff(c, axis=1)}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    n, k = 80, 4
    x = rng.normal(size=(n, 2))
    latent = x @ np.array([1.2, -0.7]) + rng.logistic(size=n)
    y = np.digitize(latent, [-1.0, 0.3, 1.5])
    xq = rng.normal(size=(5, 2))
    out: dict[str, object] = {"x": x, "y": y, "k": k, "xq": xq}
    for model, link in [
        ("cumulative", "logit"),
        ("cumulative", "probit"),
        ("continuation-ratio", "logit"),
        ("adjacent-category", "logit"),
    ]:
        f = fit(model, link, x, y, k)
        f["proba"] = probabilities(model, link, xq @ f["beta"], f["theta"])
        out[f"{model}/{link}"] = f
    for construction, kind in [("all", "logistic"), ("immediate", "logistic"), ("all", "smooth-hinge")]:
        out[f"threshold/{construction}/{kind}"] = threshold_fit(construction, kind, x, y, k, 1.0)
    out["frank-hall"] = frank_hall(x, y, k, xq, 1.0)
    out["gp"] = {
        "n": 40,
        "theta": [-1.0, 0.3, 1.5],
        "sigma": 0.5,
        **gp_ordinal(x[:40], y[:40], xq, np.array([-1.0, 0.3, 1.5]), 0.5),
    }
    return out
