"""Reference fits for aifn-compute/learning/calibration: isotonic regression from scikit-learn's IsotonicRegression
(ties in x
pooled to their weighted mean, increasing and decreasing, with weights) and scipy.optimize.isotonic_regression (on an
ordered sequence); Platt scaling from scikit-learn's _sigmoid_calibration (Platt's smoothed targets); temperature
scaling by scipy.optimize.minimize_scalar on the log loss; beta and Dirichlet calibration as unpenalised scikit-learn
LogisticRegression on [ln s, -ln(1 - s)] and on ln q."""

from typing import Any

import numpy as np
from scipy.optimize import isotonic_regression, minimize_scalar
from scipy.special import log_softmax
from sklearn.calibration import _sigmoid_calibration  # pyright: ignore[reportPrivateUsage]
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def cases() -> dict:
    rng = np.random.default_rng(7)
    out: dict = {"isotonicRegression": [], "poolAdjacentViolatorsSteps": []}
    for n, increasing, weighted, ties in [
        (12, True, False, False),
        (30, True, True, False),
        (25, False, True, False),
        (40, True, True, True),
    ]:
        x = np.sort(rng.uniform(0, 10, n)) if not ties else np.round(rng.uniform(0, 10, n))
        sign = 1 if increasing else -1
        y = np.sin(x / 3) * sign + 0.4 * rng.standard_normal(n) + x / 10 * sign
        w = rng.uniform(0.2, 2.0, n) if weighted else np.ones(n)
        fit = IsotonicRegression(increasing=increasing).fit(x, y, sample_weight=w).predict(x)
        out["isotonicRegression"].append({"x": x, "y": y, "weights": w, "increasing": increasing, "fit": fit})
    for n in (8, 50):
        y = rng.standard_normal(n).cumsum() * 0.3 + rng.standard_normal(n)
        w = rng.uniform(0.5, 1.5, n)
        res = isotonic_regression(y, weights=w, increasing=True)
        out["poolAdjacentViolatorsSteps"].append({"y": y, "weights": w, "fit": res.x, "blocks": res.blocks})
    out.update(maps(rng))
    return out


def beta_fit(s: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Kull et al.'s procedure: an unpenalised logistic regression on [ln s, -ln(1 - s)], refitted without a feature
    whose coefficient comes out negative."""
    features = np.column_stack([np.log(s), -np.log(1 - s)])
    free = [True, True]
    lr = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-12).fit(features, y)
    coef = list(lr.coef_[0])
    if coef[0] < 0 or coef[1] < 0:
        free = [coef[0] >= 0, coef[1] >= 0]
    if not all(free):
        cols = [j for j in range(2) if free[j]]
        if cols:
            lr = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-12).fit(features[:, cols], y)
            coef = [0.0, 0.0]
            for c, j in enumerate(cols):
                coef[j] = float(lr.coef_[0, c])
            return {"a": coef[0], "b": coef[1], "c": float(lr.intercept_[0])}
        rate = float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
        return {"a": 0.0, "b": 0.0, "c": float(np.log(rate / (1 - rate)))}
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(lr.intercept_[0])}


def maps(rng: np.random.Generator) -> dict[str, Any]:
    """Platt, temperature, beta and Dirichlet references on overconfident synthetic outputs."""
    out: dict[str, Any] = {
        "plattScaling": [],
        "temperatureScaling": [],
        "betaCalibration": [],
        "dirichletCalibration": [],
    }
    for n, k, temperature in [(200, 3, 2.5), (300, 4, 0.6)]:
        z = 1.5 * rng.standard_normal((n, k))
        p = np.exp(log_softmax(z, axis=1))
        y = np.array([rng.choice(k, p=row) for row in p])
        logits = temperature * z

        def nll(log_t: float, logits: np.ndarray = logits, y: np.ndarray = y) -> float:
            lp = log_softmax(logits / np.exp(log_t), axis=1)
            return float(-lp[np.arange(len(y)), y].mean())

        res = minimize_scalar(nll, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10})
        out["temperatureScaling"].append({"logits": logits, "labels": y, "temperature": float(np.exp(res.x))})
        q = np.exp(log_softmax(logits, axis=1))
        lr = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-12).fit(np.log(q), y)
        out["dirichletCalibration"].append({"probabilities": q, "labels": y, "calibrated": lr.predict_proba(np.log(q))})
    for n, a_true, b_true in [(200, 2.0, 0.7), (300, 0.5, 1.5)]:
        y = (rng.uniform(size=n) < 0.4).astype(int)
        s = np.where(y == 1, rng.beta(2 * a_true, 1, n), rng.beta(1, 2 * b_true, n))
        s = np.clip(s, 1e-6, 1 - 1e-6)
        out["betaCalibration"].append({"scores": s, "labels": y, **beta_fit(s, y)})
        f = 3 * (s - 0.5) + 0.3 * rng.standard_normal(n)
        a, b = _sigmoid_calibration(f, y)
        out["plattScaling"].append({"scores": f, "labels": y, "A": a, "B": b})
    return out
