"""Reference values for aifn-compute/learning/selection: scikit-learn's RidgeCV (exact leave-one-out squared errors,
`store_cv_results`, both `gcv_mode`s, with and without an intercept, and `alpha_per_target` on two targets) checked
against brute-force refits, and generalised cross-validation from the explicit hat matrix; a Nadaraya-Watson smoother
with leave-one-out by refitting; the one-standard-error rule on scikit-learn's GridSearchCV fold scores (diabetes,
ridge); statsmodels' OLS log-likelihood, AIC and BIC and its `aicc` over a polynomial-degree sweep, with Mallows' Cp;
soft-thresholding SURE in closed form; the .632 and .632+ bootstrap estimates of Efron and Tibshirani (1997) for least
squares and one-nearest-neighbour (scikit-learn) on fixed bootstrap samples; arviz-stats' PSIS-LOO (`loo` with
`reff=1`, and `psislw`) and WAIC as in Vehtari, Gelman and Gabry (2017) on fixed log-likelihood draws; nested
cross-validation (Bates, Hastie and Tibshirani, 2024, Algorithm 1, with the rescaling, clamping and the bias of their
equation 15) for least squares by a direct numpy implementation on fixed fold assignments; stacking weights (Yao et al.,
2018) checked against a one-dimensional root of the gradient for two models and against scipy's SLSQP and arviz-stats'
`compare` for three, with pseudo-BMA and the Bayesian-bootstrap pseudo-BMA+ (checked against `compare` on its own
Dirichlet draws)."""

from typing import Any

import arviz_stats
import numpy as np
import statsmodels.api as sm
from arviz_base import from_dict
from arviz_stats.base import array_stats
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, softmax
from scipy.stats import dirichlet, norm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tools.eval_measures import aicc


def ridge_hat(x: np.ndarray, lam: float, intercept: bool) -> np.ndarray:
    """The ridge hat matrix with an unpenalised intercept column, from the normal equations."""
    n = x.shape[0]
    a = np.column_stack([np.ones(n), x]) if intercept else x
    penalty = np.eye(a.shape[1]) * lam
    if intercept:
        penalty[0, 0] = 0
    return a @ np.linalg.solve(a.T @ a + penalty, a.T)


def ridge_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for n, d, intercept in [(25, 4, True), (30, 6, False), (12, 20, True)]:
        x = rng.standard_normal((n, d)) * rng.uniform(0.5, 3, d)
        y = x[:, : min(d, 3)] @ rng.standard_normal(min(d, 3)) + 0.7 * rng.standard_normal(n) + 2
        lambdas = np.logspace(-2, 3, 9)
        sklearn = {}
        for mode in ("svd", "eigen"):
            fit = RidgeCV(alphas=lambdas, fit_intercept=intercept, gcv_mode=mode, store_cv_results=True).fit(x, y)
            sklearn[mode] = {"cvResults": fit.cv_results_, "alpha": fit.alpha_, "bestScore": fit.best_score_}
        np.testing.assert_allclose(sklearn["svd"]["cvResults"], sklearn["eigen"]["cvResults"], rtol=1e-8, atol=1e-10)
        # Brute force: leave each row out and refit.
        brute = np.array(
            [
                [
                    y[i]
                    - Ridge(alpha=lam, fit_intercept=intercept)
                    .fit(np.delete(x, i, 0), np.delete(y, i))
                    .predict(x[i : i + 1])[0]
                    for lam in lambdas
                ]
                for i in range(n)
            ]
        )
        np.testing.assert_allclose(brute**2, sklearn["svd"]["cvResults"], rtol=1e-7, atol=1e-10)
        gcv, df = [], []
        for lam in lambdas:
            s = ridge_hat(x, lam, intercept)
            r = y - s @ y
            df.append(np.trace(s))
            gcv.append(np.mean(r**2) / (1 - np.trace(s) / n) ** 2)
        out.append(
            {
                "x": x,
                "y": y,
                "intercept": intercept,
                "lambdas": lambdas,
                "looResiduals": brute.T,
                "sklearn": sklearn,
                "gcv": gcv,
                "df": df,
                "smoother": ridge_hat(x, float(lambdas[4]), intercept),
            }
        )
    return out


def alpha_per_target(rng: np.random.Generator) -> dict[str, Any]:
    """Two targets, one smooth and one noisy, so that RidgeCV picks a different alpha for each."""
    n, d = 40, 5
    x = rng.standard_normal((n, d))
    y = np.column_stack([x @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n), 3 * rng.standard_normal(n)])
    lambdas = np.logspace(-3, 3, 13)
    fit = RidgeCV(alphas=lambdas, alpha_per_target=True, store_cv_results=True).fit(x, y)
    return {"x": x, "y": y.T, "lambdas": lambdas, "alpha": fit.alpha_, "cvResults": fit.cv_results_.transpose(1, 2, 0)}


def nadaraya_watson(rng: np.random.Generator) -> dict[str, Any]:
    n = 20
    t = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * t) + 0.3 * rng.standard_normal(n)
    k = np.exp(-0.5 * ((t[:, None] - t[None, :]) / 0.1) ** 2)
    s = k / k.sum(1, keepdims=True)
    loo = np.array([y[i] - np.delete(k[i], i) @ np.delete(y, i) / np.delete(k[i], i).sum() for i in range(n)])
    r = y - s @ y
    return {
        "smoother": s,
        "y": y,
        "looResiduals": loo,
        "loo": np.mean(loo**2),
        "gcv": np.mean(r**2) / (1 - np.trace(s) / n) ** 2,
        "df": np.trace(s),
    }


def one_standard_error() -> dict[str, Any]:
    """GridSearchCV over ridge penalties on the diabetes data; larger penalties are simpler."""
    x, y = load_diabetes(return_X_y=True)
    lambdas = np.logspace(-4, 1, 11)
    k = 10
    search = GridSearchCV(
        Ridge(), {"alpha": lambdas}, cv=KFold(k, shuffle=True, random_state=0), scoring="neg_mean_squared_error"
    ).fit(x, y)
    scores = np.array([search.cv_results_[f"split{f}_test_score"] for f in range(k)]).T
    mean = scores.mean(1)
    se = scores.std(1, ddof=1) / np.sqrt(k)
    best = int(np.argmax(mean))
    threshold = mean[best] - se[best]
    eligible = mean >= threshold
    chosen = int(np.max(np.flatnonzero(eligible)))  # the largest eligible penalty
    return {
        "lambdas": lambdas,
        "scores": scores,
        "mean": mean,
        "se": se,
        "best": best,
        "threshold": threshold,
        "eligible": eligible.astype(int),
        "chosen": chosen,
    }


def information_criteria(rng: np.random.Generator) -> dict[str, Any]:
    n = 40
    t = np.sort(rng.uniform(-1, 1, n))
    y = 1 + 2 * t - 1.5 * t**3 + 0.3 * rng.standard_normal(n)
    rows = []
    for degree in range(7):
        a = np.vander(t, degree + 1, increasing=True)
        fit = sm.OLS(y, a).fit()
        k = degree + 1
        rows.append(
            {"llf": fit.llf, "k": k, "aic": fit.aic, "bic": fit.bic, "aicc": aicc(fit.llf, n, k), "rss": fit.ssr}
        )
    sigma2 = sm.OLS(y, np.vander(t, 7, increasing=True)).fit().mse_resid
    for r in rows:
        r["cp"] = r["rss"] / n + 2 * r["k"] * sigma2 / n
        r["cpClassical"] = r["rss"] / sigma2 - n + 2 * r["k"]
    return {"n": n, "sigma2": sigma2, "rows": rows}


def soft_threshold_sure(rng: np.random.Generator) -> dict[str, Any]:
    mu = np.concatenate([np.zeros(30), rng.uniform(2, 5, 10)])
    sigma2 = 1.0
    y = mu + rng.standard_normal(mu.size)
    lambdas = np.linspace(0, 3, 13)
    sure = [
        -y.size * sigma2 + np.sum(np.minimum(y**2, lam**2)) + 2 * sigma2 * np.sum(np.abs(y) > lam) for lam in lambdas
    ]
    return {"y": y, "sigma2": sigma2, "lambdas": lambdas, "sure": sure}


def bootstrap_summary(loss_full: np.ndarray, loss_oob: list[list[float]], gamma: float) -> dict[str, Any]:
    """Efron and Tibshirani (1997): the apparent error, Err^(1), .632 and .632+ from per-point out-of-bag losses."""
    err = float(np.mean(loss_full))
    counts = np.array([len(v) for v in loss_oob])
    per_point = np.array([np.mean(v) if v else np.nan for v in loss_oob])
    err1 = float(np.nanmean(per_point))
    err1p = min(err1, gamma)
    r = (err1p - err) / (gamma - err) if err1 > err and gamma > err else 0.0
    e632 = 0.368 * err + 0.632 * err1
    plus = e632 + (err1p - err) * (0.368 * 0.632 * r) / (1 - 0.368 * r)
    return {
        "apparent": err,
        "looBootstrap": err1,
        "noInformation": gamma,
        "relativeOverfitting": r,
        "err632": e632,
        "err632plus": plus,
        "outOfBagCount": counts,
        "perPoint": per_point,
    }


def bootstrap_cases(rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Least squares with an intercept, squared error.
    n, d, b = 30, 3, 40
    x = rng.standard_normal((n, d))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(n)
    idx = rng.integers(0, n, (b, n))

    def ols(rows: np.ndarray) -> np.ndarray:
        a = np.column_stack([np.ones(rows.size), x[rows]])
        return np.linalg.lstsq(a, y[rows], rcond=None)[0]

    def predict(w: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return w[0] + x[rows] @ w[1:]

    full = predict(ols(np.arange(n)), np.arange(n))
    oob: list[list[float]] = [[] for _ in range(n)]
    for rows in idx:
        w = ols(rows)
        for i in np.setdiff1d(np.arange(n), rows):
            oob[i].append(float((y[i] - predict(w, np.array([i]))[0]) ** 2))
    gamma = float(np.mean((y[:, None] - full[None, :]) ** 2))
    out["regression"] = {"x": x, "y": y, "indices": idx, **bootstrap_summary((y - full) ** 2, oob, gamma)}
    # One-nearest-neighbour, 0-1 loss: the apparent error is 0.
    n, b = 40, 50
    labels = (rng.uniform(size=n) < 0.5).astype(int)
    x = rng.standard_normal((n, 2)) + labels[:, None] * 0.8
    idx = rng.integers(0, n, (b, n))
    knn = KNeighborsClassifier(n_neighbors=1)
    full = knn.fit(x, labels).predict(x)
    oob = [[] for _ in range(n)]
    for rows in idx:
        fit = KNeighborsClassifier(n_neighbors=1).fit(x[rows], labels[rows])
        out_rows = np.setdiff1d(np.arange(n), rows)
        for i, p in zip(out_rows, fit.predict(x[out_rows]), strict=True):
            oob[i].append(float(p != labels[i]))
    p1 = labels.mean()
    q1 = full.mean()
    gamma = float(p1 * (1 - q1) + (1 - p1) * q1)
    out["classification"] = {"x": x, "y": labels, "indices": idx, **bootstrap_summary(full != labels, oob, gamma)}
    return out


def log_lik_draws(rng: np.random.Generator, s: int, n: int) -> np.ndarray:
    """Pointwise normal log-likelihood draws of a linear model, the last observation an outlier (a large k-hat)."""
    t = np.linspace(-1, 1, n)
    y = 0.5 + 1.2 * t + 0.4 * rng.standard_normal(n)
    y[-1] += 3
    a = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    sigma = np.sqrt(np.sum((y - a @ coef) ** 2) / (n - 2))
    cov = sigma**2 * np.linalg.inv(a.T @ a)
    draws = rng.multivariate_normal(coef, cov, size=s)
    sigmas = sigma * np.sqrt((n - 2) / rng.chisquare(n - 2, size=s))
    mu = draws @ a.T
    return -0.5 * np.log(2 * np.pi * sigmas[:, None] ** 2) - 0.5 * ((y[None, :] - mu) / sigmas[:, None]) ** 2


def predictive(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for s, n in [(400, 16), (200, 12), (100, 10)]:
        ll = log_lik_draws(rng, s, n)
        tree = from_dict({"posterior": {"mu": rng.standard_normal((1, s))}, "log_likelihood": {"y": ll[None]}})
        loo = arviz_stats.loo(tree, pointwise=True, reff=1.0)
        # Smoothed log weights of the first and the last (outlying) observation.
        log_weights, khat = array_stats.psislw(ll[:, [0, n - 1]].T, r_eff=1.0)
        lppd = logsumexp(ll, axis=0) - np.log(s)
        p_waic = ll.var(axis=0, ddof=1)
        elpd_waic = lppd - p_waic
        out.append(
            {
                "logLik": ll,
                "loo": {
                    "elpd": loo.elpd,
                    "se": loo.se,
                    "p": loo.p,
                    "elpdPointwise": loo.elpd_i.values,
                    "khat": loo.pareto_k.values,
                    "goodK": loo.good_k,
                },
                "logWeights": log_weights,
                "khat": khat,
                "waic": {
                    "elpd": elpd_waic.sum(),
                    "p": p_waic.sum(),
                    "se": np.sqrt(n * elpd_waic.var(ddof=1)),
                    "elpdPointwise": elpd_waic,
                    "pPointwise": p_waic,
                    "lppd": lppd.sum(),
                },
            }
        )
    return out


def nested_cv(rng: np.random.Generator) -> dict[str, Any]:
    """Algorithm 1 of Bates, Hastie and Tibshirani (2024) for least squares with an intercept, squared loss."""
    n, k, reps = 23, 4, 3
    x = rng.standard_normal((n, 2))
    y = 1 + x @ np.array([1.5, -0.5]) + 0.8 * rng.standard_normal(n)
    fold_ids = np.array([rng.permutation(np.arange(n) % k) for _ in range(reps)])

    def losses(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        a = np.column_stack([np.ones(train.sum()), x[train]])
        coef, *_ = np.linalg.lstsq(a, y[train], rcond=None)
        return (y[test] - np.column_stack([np.ones(test.sum()), x[test]]) @ coef) ** 2

    a_list, b_list, es, outs = [], [], [], []
    for fold in fold_ids:
        for kk in range(k):
            e_in = np.concatenate([losses((fold != j) & (fold != kk), fold == j) for j in range(k) if j != kk])
            e_out = losses(fold != kk, fold == kk)
            a_list.append((e_in.mean() - e_out.mean()) ** 2)
            b_list.append(e_out.var(ddof=1) / len(e_out))
            es.append(e_in)
            outs.append(e_out)
    es_all, out_all = np.concatenate(es), np.concatenate(outs)
    ncv, cv = es_all.mean(), out_all.mean()
    mse_hat = np.mean(a_list) - np.mean(b_list)
    se_in = es_all.std(ddof=1) / np.sqrt(n)
    se = np.clip(np.sqrt(max(0.0, (k - 1) / k * mse_hat)), se_in, np.sqrt(k) * se_in)
    bias = (1 + (k - 2) / k) * (ncv - cv)
    z = norm.ppf(0.95)
    naive_se = out_all.std(ddof=1) / np.sqrt(n)
    return {
        "x": x,
        "y": y,
        "foldIds": fold_ids,
        "level": 0.9,
        "a": a_list,
        "b": b_list,
        "ncv": ncv,
        "cv": cv,
        "mseNested": mse_hat,
        "se": se,
        "bias": bias,
        "estimate": ncv - bias,
        "interval": [ncv - bias - z * se, ncv - bias + z * se],
        "naiveSe": naive_se,
        "naiveInterval": [cv - z * naive_se, cv + z * naive_se],
    }


def stacking_objective(lpd: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(logsumexp(lpd, axis=1, b=w[None, :])))


def stacking(rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Two models, each better on part of the data: the optimum is interior, a root of the derivative in w.
    lpd2 = np.column_stack([rng.normal(-1, 0.6, 30), rng.normal(-1.2, 0.8, 30)])
    shift = lpd2.max(1, keepdims=True)
    p = np.exp(lpd2 - shift)
    w1 = brentq(lambda w: np.sum((p[:, 0] - p[:, 1]) / (w * p[:, 0] + (1 - w) * p[:, 1])), 1e-12, 1 - 1e-12, xtol=1e-16)
    w2 = np.array([w1, 1 - w1])
    out["two"] = {"lpd": lpd2, "weights": w2, "objective": stacking_objective(lpd2, w2)}
    # Three models from posterior draws of normal models; PSIS-LOO pointwise elpd, as arviz-stats' compare sees it.
    n, s = 25, 400
    yobs = np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 0.5, 5)])
    trees, lpd = {}, []
    for name, mu, sigma in [("wide", 0.6, 1.6), ("centred", 0.0, 1.0), ("shifted", 2.5, 0.8)]:
        mus = mu + 0.1 * rng.standard_normal(s)
        ll = norm.logpdf(yobs[None, :], mus[:, None], sigma)
        tree = from_dict({"posterior": {"mu": mus[None]}, "log_likelihood": {"y": ll[None]}})
        trees[name] = tree
        lpd.append(arviz_stats.loo(tree, pointwise=True, reff=1.0).elpd_i.values)
    lpd3 = np.column_stack(lpd)
    names = list(trees)
    weights = {}
    for method in ("stacking", "pseudo-bma", "bb-pseudo-bma"):
        table = arviz_stats.compare(trees, method=method, round_to="none")
        weights[method] = np.array([table.loc[m, "weight"] for m in names])
    # The arviz-stats Bayesian bootstrap, reproduced from its own draws.
    draws = dirichlet.rvs(alpha=[1] * n, size=1000, random_state=124)
    bb = np.mean([softmax(draws[b] @ (lpd3 * n)) for b in range(1000)], axis=0)
    np.testing.assert_allclose(bb, weights["bb-pseudo-bma"], rtol=1e-12)
    pseudo = softmax(lpd3.sum(0))
    np.testing.assert_allclose(pseudo, weights["pseudo-bma"], rtol=1e-12)
    # A tight optimum by SLSQP on the simplex, checked by the optimality conditions.
    fit = minimize(
        lambda w: -stacking_objective(lpd3, np.maximum(w, 0)),
        np.full(3, 1 / 3),
        method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-16, "maxiter": 1000},
    )
    w3 = np.maximum(fit.x, 0) / np.maximum(fit.x, 0).sum()
    small = rng.dirichlet(np.ones(n), size=40)
    out["three"] = {
        "lpd": lpd3,
        "weights": w3,
        "objective": stacking_objective(lpd3, w3),
        "arviz": weights,
        "pseudoBma": pseudo,
        "elpd": lpd3.sum(0),
        "draws": small,
        "pseudoBmaPlus": np.mean([softmax(small[b] @ (lpd3 * n)) for b in range(40)], axis=0),
        "pseudoBmaPlusSe": np.array([small[b] @ (lpd3 * n) for b in range(40)]).std(0),
    }
    return out


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(20261013)
    return {
        "ridgeCv": ridge_cases(rng),
        "alphaPerTarget": alpha_per_target(rng),
        "nadarayaWatson": nadaraya_watson(rng),
        "oneStandardError": one_standard_error(),
        "informationCriteria": information_criteria(rng),
        "softThresholdSure": soft_threshold_sure(rng),
        "bootstrap": bootstrap_cases(rng),
        "predictive": predictive(rng),
        "nestedCv": nested_cv(np.random.default_rng(2024)),
        "stacking": stacking(np.random.default_rng(2018)),
    }
