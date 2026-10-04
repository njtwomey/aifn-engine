"""Golden values for aifn-compute/optim/second-order: lasso and elastic-net solutions from scikit-learn's coordinate
descent,
which `owlqn` must reach on the same objectives."""

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    n, d = 40, 8
    x = rng.normal(size=(n, d))
    w = np.array([2.0, 0.0, -1.5, 0.0, 0.0, 0.8, 0.0, 0.0])
    y = x @ w + 0.3 * rng.normal(size=n)
    out: dict[str, object] = {"x": x, "y": y, "lasso": [], "elastic_net": []}
    # sklearn's objective: (1/(2n))‖y − Xw‖² + α‖w‖₁ (no intercept here).
    for alpha in [0.01, 0.1, 0.5]:
        m = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=100000).fit(x, y)
        out["lasso"].append({"alpha": alpha, "coef": m.coef_})  # pyright: ignore[reportAttributeAccessIssue]
    # (1/(2n))‖y − Xw‖² + α·ρ‖w‖₁ + ½α(1 − ρ)‖w‖².
    for alpha, rho in [(0.1, 0.5), (0.3, 0.2)]:
        m = ElasticNet(alpha=alpha, l1_ratio=rho, fit_intercept=False, tol=1e-12, max_iter=100000).fit(x, y)
        out["elastic_net"].append({"alpha": alpha, "l1Ratio": rho, "coef": m.coef_})  # pyright: ignore[reportAttributeAccessIssue]
    return out
