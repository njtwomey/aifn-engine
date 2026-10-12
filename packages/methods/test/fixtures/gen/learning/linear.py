"""Golden values for aifn-methods learning/linear and learning/generalised/glm (logistic regression).

From scikit-learn's LinearRegression, Ridge and LogisticRegression; and, under `diabetes`, the regularised paths on the
diabetes data: `ElasticNet` and `enet_path` (coordinate descent), `Ridge` with the effective degrees of freedom from
the singular values of the centred design, `lars_path` (least-angle regression and its lasso modification) and
`SGDRegressor` over twenty seeds. The diabetes data are scikit-learn's raw copy (`load_diabetes(scaled=False)`), the
same values as the pinned source vendored by `scripts/diabetes.py`. As in the field notes, the features are
standardised with the population standard deviation (`StandardScaler`) and, for the paths, the target is centred.
Coordinate descent runs to a duality gap of 1e-12 of the target's sum of squares, tighter than the engine's default,
so the fixture is the optimum and not one solver's stopping point.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
    enet_path,
    lars_path,
)
from sklearn.preprocessing import StandardScaler

TOL = 1e-12
MAX_ITER = 1_000_000


def events(coefs: np.ndarray) -> list[dict[str, object]]:
    """The knots at which each feature joins (zero there, nonzero after) or leaves (nonzero before, zero there)."""
    out: list[dict[str, object]] = []
    d, k = coefs.shape
    for t in range(k):
        for j in range(d):
            before = t > 0 and coefs[j, t - 1] != 0
            here = coefs[j, t] != 0
            after = t + 1 < k and coefs[j, t + 1] != 0
            if not here and after and not before:
                out.append({"knot": t, "feature": j, "kind": "enter"})
            elif before and not here:
                out.append({"knot": t, "feature": j, "kind": "drop"})
    return out


def diabetes() -> dict[str, object]:
    """The diabetes cases: the regularised paths of the field notes, against scikit-learn."""
    raw, target = load_diabetes(scaled=False, return_X_y=True)
    n = raw.shape[0]
    x = StandardScaler().fit_transform(raw)
    y = target - target.mean()

    fits = []
    for alpha, l1_ratio, standardised in [(1.0, 1.0, True), (0.5, 0.5, True), (0.1, 0.9, True), (2.0, 0.1, True),
                                          (1.0, 1.0, False)]:  # fmt: skip
        design = x if standardised else raw
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=TOL, max_iter=MAX_ITER).fit(design, target)
        r = target - m.predict(design)
        objective = (r @ r) / (2 * n) + alpha * l1_ratio * np.abs(m.coef_).sum()
        objective += 0.5 * alpha * (1 - l1_ratio) * (m.coef_ @ m.coef_)
        fits.append({"alpha": alpha, "l1_ratio": l1_ratio, "standardised": standardised, "coef": m.coef_,
                     "intercept": m.intercept_, "objective": objective})  # fmt: skip

    paths = []
    for l1_ratio in [0.1, 0.5, 0.9, 1.0]:
        alphas, coefs, _ = enet_path(x, y, l1_ratio=l1_ratio, eps=1e-4, alphas=81, tol=TOL, max_iter=MAX_ITER)
        paths.append({"l1_ratio": l1_ratio, "alphas": alphas, "coef": coefs, "nonzero": (coefs != 0).sum(axis=0)})

    lambdas = np.logspace(5, -2, 71)
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    ridge = {
        "lambdas": lambdas,
        "coef": np.array([Ridge(alpha=lam).fit(x, target).coef_ for lam in lambdas]).T,
        "intercept": [Ridge(alpha=lam).fit(x, target).intercept_ for lam in lambdas],
        "dof": [(singular**2 / (singular**2 + lam)).sum() for lam in lambdas],
    }

    lars = {}
    for method in ["lar", "lasso"]:
        alphas, active, coefs = lars_path(x, y, method=method)
        lars[method] = {
            "alphas": alphas,
            "active": active,
            "coef": coefs,
            "l1_norm": np.abs(coefs).sum(axis=0),
            "events": events(coefs),
        }

    sgd = []
    for penalty, alpha, l1_ratio in [("l2", 0.1, 0.15), ("l1", 1.0, 1.0), ("elasticnet", 0.5, 0.5)]:
        runs = [
            SGDRegressor(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, max_iter=100, tol=None, random_state=s).fit(
                x, y
            )
            for s in range(20)
        ]
        w = np.array([r.coef_ for r in runs])
        b = np.array([r.intercept_[0] for r in runs])
        if penalty == "l2":
            optimum = Ridge(alpha=n * alpha).fit(x, y).coef_
        else:
            rho = 1.0 if penalty == "l1" else l1_ratio
            optimum = ElasticNet(alpha=alpha, l1_ratio=rho, tol=TOL, max_iter=MAX_ITER).fit(x, y).coef_
        sgd.append({"penalty": penalty, "alpha": alpha, "l1_ratio": l1_ratio, "epochs": 100, "mean": w.mean(axis=0),
                    "sd": w.std(axis=0), "intercept_mean": b.mean(), "intercept_sd": b.std(),
                    "zeros": (w == 0).sum(axis=1).min(), "optimum": optimum})  # fmt: skip

    return {
        "data": {
            "mean": raw.mean(axis=0),
            "sd": raw.std(axis=0),
            "target_mean": target.mean(),
            "first": raw[0],
            "first_target": target[0],
        },
        "elastic_net": fits,
        "paths": paths,
        "ridge": ridge,
        "lars": lars,
        "sgd": sgd,
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    n, d = 40, 3
    x = rng.normal(size=(n, d))
    y = x @ np.array([1.5, -2.0, 0.5]) + 0.7 + rng.normal(scale=0.3, size=n)

    ols = LinearRegression().fit(x, y)
    residuals = y - ols.predict(x)
    ridge = Ridge(alpha=2.5).fit(x, y)
    x_test = rng.normal(size=(5, d))

    # Rank-deficient design: the third column is the sum of the first two.
    x_deficient = np.column_stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 1]])
    deficient = LinearRegression().fit(x_deficient, y)

    logits = x @ np.array([1.0, -1.5, 0.5]) - 0.3
    y_binary = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    binary = LogisticRegression(C=1 / 0.5, tol=1e-12, max_iter=100000).fit(x, y_binary)

    scores = x @ np.array([[1.0, -1.0, 0.0], [0.5, 0.5, -1.0], [-1.0, 0.0, 1.0]])
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    y_multi = np.array([rng.choice(3, p=p) for p in probs])
    multi = LogisticRegression(C=1 / 0.8, tol=1e-12, max_iter=100000).fit(x, y_multi)

    return {
        "x": x,
        "y": y,
        "x_test": x_test,
        "ols": {
            "coef": ols.coef_,
            "intercept": ols.intercept_,
            "predict": ols.predict(x_test),
            "noise_sd": np.sqrt((residuals**2).sum() / (n - d - 1)),
        },
        "ridge": {"alpha": 2.5, "coef": ridge.coef_, "intercept": ridge.intercept_, "predict": ridge.predict(x_test)},
        "deficient": {"x": x_deficient, "coef": deficient.coef_, "intercept": deficient.intercept_},
        "binary": {
            "l2": 0.5,
            "y": y_binary,
            "coef": binary.coef_[0],
            "intercept": binary.intercept_[0],
            "proba": binary.predict_proba(x_test)[:, 1],
            "predict": binary.predict(x_test),
        },
        "multinomial": {
            "l2": 0.8,
            "y": y_multi,
            "coef": multi.coef_.T,
            "intercept": multi.intercept_,
            "proba": multi.predict_proba(x_test),
            "predict": multi.predict(x_test),
        },
        "diabetes": diabetes(),
    }
