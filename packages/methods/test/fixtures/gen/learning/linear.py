"""Golden values for aifn-methods learning/linear and learning/generalised/glm (logistic regression).

From scikit-learn's LinearRegression, Ridge and LogisticRegression; and, under `diabetes`, the regularised paths on the
diabetes data: `ElasticNet` and `enet_path` (coordinate descent), `Ridge` with the effective degrees of freedom from
the singular values of the centred design, `lars_path` (least-angle regression and its lasso modification) and
`SGDRegressor` over twenty seeds. The diabetes data are scikit-learn's raw copy (`load_diabetes(scaled=False)`), the
same values as the pinned source vendored by `scripts/diabetes.py`. As in the field notes, the features are
standardised with the population standard deviation (`StandardScaler`) and, for the paths, the target is centred.
Coordinate descent runs to a duality gap of 1e-12 of the target's sum of squares, tighter than the engine's default,
so the fixture is the optimum and not one solver's stopping point.

Under `evidence`, `subsets` and `group_lasso`, on the same standardised diabetes data: `BayesianRidge` with its gamma
hyperpriors switched off (MacKay's updates, with the log evidence at each step in `scores_`), and the log evidence at
fixed precisions as the log density of the marginal N(0, I / beta + X X^T / alpha); best-subset selection by brute
force over all 1024 subsets with `lstsq`, stepwise selection and incremental forward stagewise regression written out
in numpy; and the group lasso by proximal gradient (FISTA with restarts) polished by Newton's method on the active
groups.
"""

import itertools

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    BayesianRidge,
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


def standardised_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """The diabetes design standardised with the population standard deviation, and the raw target."""
    raw, target = load_diabetes(scaled=False, return_X_y=True)
    return StandardScaler().fit_transform(raw), target


def log_evidence(x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> float:
    """The log density of y under its marginal N(0, I / beta + X X^T / alpha), the evidence by its definition."""
    cov = np.eye(len(y)) / beta + x @ x.T / alpha
    return float(multivariate_normal(mean=np.zeros(len(y)), cov=cov).logpdf(y))


def evidence() -> dict[str, object]:
    """MacKay's updates as BayesianRidge (hyperpriors off), and the evidence at fixed precisions by its definition."""
    x, y = standardised_diabetes()
    runs = []
    for name, rows, cols in [("diabetes", slice(None), slice(None)), ("wide", slice(0, 8), slice(None))]:
        xs, ys = x[rows, cols], y[rows]
        m = BayesianRidge(max_iter=1000, tol=1e-8, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0, compute_score=True)
        m.fit(xs, ys)
        runs.append({"name": name, "rows": len(ys), "scores": m.scores_, "n_iter": m.n_iter_, "alpha": m.lambda_,
                     "beta": m.alpha_, "coef": m.coef_, "intercept": m.intercept_, "sigma": m.sigma_})  # fmt: skip
    fixed = []
    xc, yc = x - x.mean(axis=0), y - y.mean()
    # Bishop's polynomial basis on ten points, uncentred, and the centred diabetes data.
    t = np.linspace(0, 1, 10)
    target = np.sin(2 * np.pi * t) + 0.3 * np.cos(17 * t)
    for degree, alpha, beta in [(0, 5e-3, 11.1), (3, 5e-3, 11.1), (9, 5e-3, 11.1), (3, 0.5, 2.0)]:
        design = np.vander(t, degree + 1, increasing=True)
        fixed.append({"case": "polynomial", "degree": degree, "alpha": alpha, "beta": beta,
                      "log_evidence": log_evidence(design, target, alpha, beta)})  # fmt: skip
    for alpha, beta in [(1e-3, 3e-4), (1.0, 1e-3)]:
        fixed.append({"case": "diabetes", "degree": -1, "alpha": alpha, "beta": beta,
                      "log_evidence": log_evidence(xc, yc, alpha, beta)})  # fmt: skip
    return {"runs": runs, "polynomial": {"t": t, "y": target}, "fixed": fixed}


def rss_of(x: np.ndarray, y: np.ndarray, subset: tuple[int, ...] | list[int]) -> float:
    """The residual sum of squares of least squares on the columns of a subset (x and y centred)."""
    if len(subset) == 0:
        return float(y @ y)
    cols = x[:, list(subset)]
    w = np.linalg.lstsq(cols, y, rcond=None)[0]
    r = y - cols @ w
    return float(r @ r)


def stepwise_reference(x: np.ndarray, y: np.ndarray, direction: str, criterion: str) -> list[dict[str, object]]:
    """Greedy stepwise selection, ties to the lowest column, R's AIC and BIC for a linear model with an intercept."""
    n, d = x.shape

    def score(subset: list[int]) -> tuple[float, float]:
        rss = rss_of(x, y, subset)
        if criterion == "rss":
            return rss, rss
        k = len(subset) + 1
        return rss, n * np.log(rss / n) + (2 if criterion == "aic" else np.log(n)) * k

    active: list[int] = [] if direction in ("forward", "both") else list(range(d))
    rss, value = score(active)
    steps: list[dict[str, object]] = [{"action": "start", "feature": -1, "active": list(active), "rss": rss,
                                       "criterion": value}]  # fmt: skip
    while True:
        moves = []
        if direction != "backward":
            moves += [("add", j, sorted([*active, j])) for j in range(d) if j not in active]
        if direction != "forward":
            moves += [("remove", j, [k for k in active if k != j]) for j in active]
        if not moves:
            break
        best = None
        for action, j, subset in moves:
            r, v = score(subset)
            if best is None or v < best[4]:
                best = (action, j, subset, r, v)
        assert best is not None
        if criterion != "rss" and not best[4] < value:
            break
        active, value = best[2], best[4]
        steps.append({"action": best[0], "feature": best[1], "active": best[2], "rss": best[3], "criterion": best[4]})
    return steps


def stagewise_reference(x: np.ndarray, y: np.ndarray, epsilon: float, steps: int) -> dict[str, object]:
    """Incremental forward stagewise regression on centred data, stopping when no epsilon step lowers the RSS."""
    d = x.shape[1]
    w = np.zeros(d)
    r = y.copy()
    path = [w.copy()]
    features = []
    for _ in range(steps):
        c = x.T @ r
        j = int(np.argmax(np.abs(c)))
        if np.abs(c[j]) < epsilon * (x[:, j] @ x[:, j]) / 2:
            break
        delta = epsilon * np.sign(c[j])
        w[j] += delta
        r -= delta * x[:, j]
        path.append(w.copy())
        features.append(j)
    coef = np.array(path).T
    return {"epsilon": epsilon, "steps": steps, "coef": coef, "features": features,
            "rss": [float(np.sum((y - x @ col) ** 2)) for col in coef.T]}  # fmt: skip


def subsets() -> dict[str, object]:
    """Best subsets by brute force, stepwise selection and forward stagewise on the standardised diabetes data."""
    x, y = standardised_diabetes()
    xc, yc = x - x.mean(axis=0), y - y.mean()
    d = x.shape[1]
    sizes, rss, sets = [], [], []
    for k in range(d + 1):
        for subset in itertools.combinations(range(d), k):
            sizes.append(k)
            sets.append(list(subset))
            rss.append(rss_of(xc, yc, subset))
    rss_arr = np.array(rss)
    best = []
    for k in range(d + 1):
        idx = [i for i, s in enumerate(sizes) if s == k]
        i = idx[int(np.argmin(rss_arr[idx]))]
        cols = sets[i]
        w = np.zeros(d)
        if cols:
            w[cols] = np.linalg.lstsq(xc[:, cols], yc, rcond=None)[0]
        best.append({"size": k, "subset": cols, "rss": rss[i], "coef": w, "intercept": y.mean() - x.mean(axis=0) @ w})
    runs = [("forward", "rss"), ("backward", "rss"), ("forward", "bic"), ("backward", "aic"), ("both", "aic")]
    stepwise = [
        {"direction": direction, "criterion": criterion, "steps": stepwise_reference(xc, yc, direction, criterion)}
        for direction, criterion in runs
    ]
    return {
        "all": {"sizes": sizes, "subsets": sets, "rss": rss},
        "best": best,
        "stepwise": stepwise,
        "stagewise": stagewise_reference(xc, yc, 0.5, 2000),
    }


def group_lasso_reference(x: np.ndarray, y: np.ndarray, groups: list[int], alpha: float) -> np.ndarray:
    """The group lasso (1/(2n) loss, penalty alpha sum sqrt(p_g) |w_g|) by FISTA with restarts, run to convergence."""
    n, d = x.shape
    labels = sorted(set(groups))
    members = [np.array([j for j in range(d) if groups[j] == g]) for g in labels]
    lipschitz = np.linalg.eigvalsh(x.T @ x / n).max()
    step = 1 / lipschitz

    def prox(v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        for cols in members:
            norm = np.linalg.norm(v[cols])
            threshold = step * alpha * np.sqrt(len(cols))
            if norm > threshold:
                out[cols] = (1 - threshold / norm) * v[cols]
        return out

    def objective(w: np.ndarray) -> float:
        r = y - x @ w
        return r @ r / (2 * n) + alpha * sum(np.sqrt(len(c)) * np.linalg.norm(w[c]) for c in members)

    w = np.zeros(d)
    z = w.copy()
    t = 1.0
    for _ in range(20_000):
        grad = x.T @ (x @ z - y) / n
        nxt = prox(z - step * grad)
        if objective(nxt) > objective(w):
            # Restart the momentum when the objective rises.
            z, t = w.copy(), 1.0
            continue
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = nxt + (t - 1) / t_next * (nxt - w)
        done = np.max(np.abs(nxt - w)) <= 1e-15 * max(1.0, np.max(np.abs(nxt)))
        w, t = nxt, t_next
        if done:
            break
    # Polish by Newton's method on the active groups, where the objective is smooth: FISTA leaves errors of order
    # 1e-7 along the poorly conditioned directions of these data, Newton removes them.
    active = [cols for cols in members if np.linalg.norm(w[cols]) > 0]
    idx = np.concatenate(active)
    for _ in range(50):
        grad = x.T @ (x @ w - y) / n
        hess = x.T @ x / n
        for cols in active:
            norm = np.linalg.norm(w[cols])
            scale = alpha * np.sqrt(len(cols))
            grad[cols] += scale * w[cols] / norm
            hess[np.ix_(cols, cols)] += scale * (np.eye(len(cols)) / norm - np.outer(w[cols], w[cols]) / norm**3)
        delta = np.linalg.solve(hess[np.ix_(idx, idx)], grad[idx])
        w[idx] -= delta
        if np.max(np.abs(delta)) <= 1e-15 * np.max(np.abs(w)):
            break
    return w


def group_lasso() -> dict[str, object]:
    """The group lasso on the standardised diabetes data, against proximal gradient."""
    x, y = standardised_diabetes()
    n = x.shape[0]
    xc, yc = x - x.mean(axis=0), y - y.mean()
    out = []
    for groups in [[0, 0, 1, 1, 2, 2, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]]:
        members = [[j for j in range(10) if groups[j] == g] for g in sorted(set(groups))]
        top = max(np.linalg.norm(xc[:, c].T @ yc) / (n * np.sqrt(len(c))) for c in members)
        fits = []
        for fraction in [0.5, 0.1, 0.01]:
            alpha = top * fraction
            w = group_lasso_reference(xc, yc, groups, alpha)
            r = yc - xc @ w
            penalty = sum(np.sqrt(len(c)) * np.linalg.norm(w[c]) for c in members)
            fits.append({"alpha": alpha, "coef": w, "intercept": y.mean() - x.mean(axis=0) @ w,
                         "objective": r @ r / (2 * n) + alpha * penalty})  # fmt: skip
        out.append({"groups": groups, "alpha_max": top, "fits": fits})
    return {"cases": out}


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
        "evidence": evidence(),
        "subsets": subsets(),
        "group_lasso": group_lasso(),
    }
