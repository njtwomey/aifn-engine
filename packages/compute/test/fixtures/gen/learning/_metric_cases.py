"""Reference cases for every registered metric of aifn-compute/learning/metrics, keyed by registry key: each case is the
metric's positional arguments (options last) and the value. Values come from scikit-learn and scipy where they define
the metric, from published worked examples where they exist (Krippendorff 2011, Shrout and Fleiss 1979, Fleiss 1971),
and otherwise from a direct numpy computation of the formula in the metric's docstring."""

import itertools
from typing import Protocol, cast

import numpy as np
from scipy import stats
from scipy.spatial import distance, procrustes
from scipy.stats import contingency
from sklearn import metrics as sk
from sklearn.preprocessing import label_binarize


class _HasStatistic(Protocol):
    statistic: float


def statistic(result: object) -> float:
    """The statistic of a scipy.stats result, whose result classes Pyright cannot see into (scipy is unstubbed)."""
    return cast(_HasStatistic, result).statistic


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def confusion_counts(y, p, labels):
    """Per class one-vs-rest TP, FP, FN, TN."""
    out = []
    for c in labels:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        tn = np.sum((y != c) & (p != c))
        out.append((tp, fp, fn, tn))
    return np.array(out, dtype=float)


RATE = {
    "specificity": lambda tp, fp, fn, tn: tn / (tn + fp),
    "negativePredictiveValue": lambda tp, fp, fn, tn: tn / (tn + fn),
    "falsePositiveRate": lambda tp, fp, fn, tn: fp / (fp + tn),
    "falseNegativeRate": lambda tp, fp, fn, tn: fn / (fn + tp),
    "informedness": lambda tp, fp, fn, tn: tp / (tp + fn) + tn / (tn + fp) - 1,
    "markedness": lambda tp, fp, fn, tn: tp / (tp + fp) + tn / (tn + fn) - 1,
}


def classification(rng, cases):
    n, K = 60, 4
    y = rng.integers(0, K, n)
    p = np.where(rng.uniform(size=n) < 0.6, y, rng.integers(0, K, n))
    yb = rng.integers(0, 2, n)
    pb = np.where(rng.uniform(size=n) < 0.7, yb, 1 - yb)
    w = rng.uniform(0.2, 2, n)
    labels = list(range(K))

    cases["accuracy"] = [
        {"args": [y, p], "value": sk.accuracy_score(y, p)},
        {"args": [y, p, {"sampleWeight": w}], "value": sk.accuracy_score(y, p, sample_weight=w)},
        {"args": [[str(v) for v in y], [str(v) for v in p]], "value": sk.accuracy_score(y, p)},
    ]
    cases["errorRate"] = [{"args": [y, p], "value": 1 - sk.accuracy_score(y, p)}]
    cases["balancedAccuracy"] = [
        {"args": [y, p], "value": sk.balanced_accuracy_score(y, p)},
        {"args": [y, p, {"adjusted": True}], "value": sk.balanced_accuracy_score(y, p, adjusted=True)},
        {"args": [y, p, {"sampleWeight": w}], "value": sk.balanced_accuracy_score(y, p, sample_weight=w)},
    ]
    prf = {
        "precision": sk.precision_score,
        "recall": sk.recall_score,
        "f1": sk.f1_score,
        "jaccardScore": sk.jaccard_score,
    }
    for key, f in prf.items():
        cs = [{"args": [yb, pb, {"average": "binary", "positive": 1}], "value": f(yb, pb)}]
        cs.append({"args": [yb, pb, {"average": "binary", "positive": 0}], "value": f(yb, pb, pos_label=0)})
        for avg in ("macro", "micro", "weighted"):
            cs.append({"args": [y, p, {"average": avg}], "value": f(y, p, average=avg)})
        cs.append(
            {
                "args": [y, p, {"average": "macro", "sampleWeight": w}],
                "value": f(y, p, average="macro", sample_weight=w),
            }
        )
        cases[key] = cs
    cases["fBeta"] = [
        {"args": [yb, pb, {"beta": 2, "average": "binary", "positive": 1}], "value": sk.fbeta_score(yb, pb, beta=2)},
        {"args": [y, p, {"beta": 0.5, "average": "macro"}], "value": sk.fbeta_score(y, p, beta=0.5, average="macro")},
        {"args": [y, p, {"beta": 3, "average": "weighted"}], "value": sk.fbeta_score(y, p, beta=3, average="weighted")},
    ]
    # Multi-label sets: samples averaging.
    Y = (rng.uniform(size=(25, 5)) < 0.4).astype(int)
    P = np.where(rng.uniform(size=Y.shape) < 0.8, Y, 1 - Y)
    for key, f in prf.items():
        cases[key].append(
            {
                "args": [Y, P, {"average": "samples", "zeroDivision": 0}],
                "value": f(Y, P, average="samples", zero_division=0),  # pyright: ignore[reportArgumentType]  # sklearn is unstubbed: zero_division inferred as str
            }
        )
        cases[key].append({"args": [Y, P, {"average": "micro"}], "value": f(Y, P, average="micro")})
    cases["hammingLoss"] = [
        {"args": [Y, P], "value": sk.hamming_loss(Y, P)},
        {"args": [y, p], "value": sk.hamming_loss(y, p)},
    ]
    cases["exactMatch"] = [{"args": [Y, P], "value": sk.accuracy_score(Y, P)}]

    cb = confusion_counts(yb, pb, [1])[0]
    cm = confusion_counts(y, p, labels)
    for key, f in RATE.items():
        cases[key] = [
            {"args": [yb, pb, {"average": "binary", "positive": 1}], "value": f(*cb)},
            {"args": [y, p, {"average": "macro"}], "value": float(np.mean([f(*c) for c in cm]))},
        ]
    cases["matthewsCorrelation"] = [
        {"args": [yb, pb], "value": sk.matthews_corrcoef(yb, pb)},
        {"args": [y, p], "value": sk.matthews_corrcoef(y, p)},
        {"args": [y, p, {"sampleWeight": w}], "value": sk.matthews_corrcoef(y, p, sample_weight=w)},
    ]
    cases["cohensKappa"] = [
        {"args": [y, p], "value": sk.cohen_kappa_score(y, p)},
        {"args": [y, p, {"weights": "linear"}], "value": sk.cohen_kappa_score(y, p, weights="linear")},
        {"args": [y, p, {"weights": "quadratic"}], "value": sk.cohen_kappa_score(y, p, weights="quadratic")},
    ]
    lr_pos, lr_neg = sk.class_likelihood_ratios(yb, pb)
    cases["positiveLikelihoodRatio"] = [{"args": [yb, pb, {"positive": 1}], "value": lr_pos}]
    cases["negativeLikelihoodRatio"] = [{"args": [yb, pb, {"positive": 1}], "value": lr_neg}]
    cases["diagnosticOddsRatio"] = [{"args": [yb, pb, {"positive": 1}], "value": lr_pos / lr_neg}]


def ordinal(rng, cases):
    n, K = 50, 5
    y = rng.integers(0, K, n)
    p = np.clip(y + rng.integers(-2, 3, n), 0, K - 1)
    labels = list(range(K))
    cases["quadraticWeightedKappa"] = [{"args": [y, p], "value": sk.cohen_kappa_score(y, p, weights="quadratic")}]
    cases["ordinalMeanAbsoluteError"] = [{"args": [y, p], "value": np.mean(np.abs(y - p))}]
    present = sorted(set(y))
    cases["macroMeanAbsoluteError"] = [
        {"args": [y, p], "value": np.mean([np.mean(np.abs(y[y == c] - p[y == c])) for c in present])}
    ]
    cases["withinToleranceAccuracy"] = [
        {"args": [y, p, {"tolerance": 1, "labels": labels}], "value": np.mean(np.abs(y - p) <= 1)},
        {"args": [y, p, {"tolerance": 0, "labels": labels}], "value": np.mean(y == p)},
    ]
    P = softmax(rng.normal(size=(n, K)) + np.eye(K)[y])
    cdf = np.cumsum(P, axis=1)[:, :-1]
    obs = (y[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    cases["rankedProbabilityScore"] = [{"args": [y, P], "value": np.mean(np.sum((cdf - obs) ** 2, axis=1) / (K - 1))}]
    score = np.round(0.5 * y + rng.normal(size=n), 1)
    cases["ordinalConcordanceIndex"] = [{"args": [y, score], "value": (stats.somersd(y, score).statistic + 1) / 2}]


def regression(rng, cases):
    n = 40
    y = rng.uniform(0.5, 5, n)
    q = y + rng.normal(0, 0.6, n)
    q = np.where(q <= 0.05, 0.05, q)
    w = rng.uniform(0.2, 2, n)
    sk_plain = {
        "meanSquaredError": sk.mean_squared_error,
        "rootMeanSquaredError": sk.root_mean_squared_error,
        "meanAbsoluteError": sk.mean_absolute_error,
        "r2Score": sk.r2_score,
        "explainedVariance": sk.explained_variance_score,
        "meanSquaredLogError": sk.mean_squared_log_error,
        "poissonDeviance": sk.mean_poisson_deviance,
        "gammaDeviance": sk.mean_gamma_deviance,
        "meanAbsolutePercentageError": sk.mean_absolute_percentage_error,
    }
    for key, f in sk_plain.items():
        cases[key] = [
            {"args": [y, q], "value": f(y, q)},
            {"args": [y, q, {"sampleWeight": w}], "value": f(y, q, sample_weight=w)},
        ]
    cases["medianAbsoluteError"] = [{"args": [y, q], "value": sk.median_absolute_error(y, q)}]
    cases["maxError"] = [{"args": [y, q], "value": sk.max_error(y, q)}]
    cases["pinballLoss"] = [
        {"args": [y, q, {"tau": tau}], "value": sk.mean_pinball_loss(y, q, alpha=tau)} for tau in (0.1, 0.5, 0.9)
    ]
    cases["tweedieDeviance"] = [
        # sklearn is unstubbed: power is inferred as int from its default of 0.
        {"args": [y, q, {"power": pw}], "value": sk.mean_tweedie_deviance(y, q, power=pw)}  # pyright: ignore[reportArgumentType]
        for pw in (0, 1, 1.5, 2, 3)
    ]
    r2 = sk.r2_score(y, q)
    cases["adjustedR2Score"] = [{"args": [y, q, {"predictors": 3}], "value": 1 - (1 - r2) * (n - 1) / (n - 3 - 1)}]
    e = y - q
    cases["logCoshError"] = [{"args": [y, q], "value": np.mean(np.log(np.cosh(e)))}]
    cases["huberLoss"] = [
        {
            "args": [y, q, {"delta": d}],
            "value": np.mean(np.where(np.abs(e) <= d, 0.5 * e**2, d * (np.abs(e) - 0.5 * d))),
        }
        for d in (0.3, 1.0)
    ]
    rmse = np.sqrt(np.mean(e**2))
    cases["normalisedRootMeanSquaredError"] = [
        {"args": [y, q, {"by": "std"}], "value": rmse / np.std(y)},
        {"args": [y, q, {"by": "mean"}], "value": rmse / np.mean(y)},
        {"args": [y, q, {"by": "range"}], "value": rmse / np.ptp(y)},
    ]
    cases["symmetricMeanAbsolutePercentageError"] = [
        {"args": [y, q], "value": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(q)))}
    ]
    cases["weightedMeanAbsolutePercentageError"] = [{"args": [y, q], "value": np.sum(np.abs(e)) / np.sum(np.abs(y))}]
    train = np.sin(np.arange(30) * 2 * np.pi / 7) * 3 + rng.normal(0, 0.3, 30) + 5
    for m in (1, 7):
        scale_abs = np.mean(np.abs(train[m:] - train[:-m]))
        scale_sq = np.mean((train[m:] - train[:-m]) ** 2)
        cases.setdefault("meanAbsoluteScaledError", []).append(
            {"args": [y, q, {"train": train, "season": m}], "value": np.mean(np.abs(e)) / scale_abs}
        )
        cases.setdefault("rootMeanSquaredScaledError", []).append(
            {"args": [y, q, {"train": train, "season": m}], "value": np.sqrt(np.mean(e**2) / scale_sq)}
        )


def ece(y, p, bins, strategy, norm):
    n = len(p)
    if strategy == "uniform":
        b = np.minimum(bins - 1, np.floor(p * bins).astype(int))
    else:
        order = np.lexsort((np.arange(n), p))
        b = np.empty(n, int)
        b[order] = (np.arange(n) * bins) // n
    total = 0.0
    for m in range(bins):
        sel = b == m
        c = sel.sum()
        if c == 0:
            continue
        gap = abs(y[sel].mean() - p[sel].mean())
        total = max(total, gap) if norm == "max" else total + c / n * gap**norm
    return np.sqrt(total) if norm == 2 else total


def probabilistic(rng, cases):
    n, K = 80, 3
    y = rng.integers(0, K, n)
    P = softmax(rng.normal(size=(n, K)) + 1.2 * np.eye(K)[y])
    yb = rng.integers(0, 2, n)
    pb = np.clip(0.5 + 0.35 * (2 * yb - 1) * rng.uniform(0, 1, n) + rng.normal(0, 0.15, n), 0.01, 0.99)
    cases["logLoss"] = [
        {"args": [yb, pb, {"positive": 1}], "value": sk.log_loss(yb, pb)},
        {"args": [y, P], "value": sk.log_loss(y, P)},
    ]
    cases["brierScore"] = [
        {"args": [yb, pb, {"positive": 1}], "value": sk.brier_score_loss(yb, pb)},
        {"args": [y, P], "value": np.mean(np.sum((P - np.eye(K)[y]) ** 2, axis=1))},
    ]
    cases["sphericalScore"] = [
        {"args": [y, P], "value": np.mean(1 - P[np.arange(n), y] / np.linalg.norm(P, axis=1))},
        {
            "args": [yb, pb, {"positive": 1}],
            "value": np.mean(1 - np.where(yb == 1, pb, 1 - pb) / np.sqrt(pb**2 + (1 - pb) ** 2)),
        },
    ]
    yf = yb.astype(float)
    for strategy in ("uniform", "quantile"):
        for bins in (5, 10):
            o = {"bins": bins, "strategy": strategy, "positive": 1}
            cases.setdefault("expectedCalibrationError", []).append(
                {"args": [yb, pb, o], "value": ece(yf, pb, bins, strategy, 1)}
            )
            cases.setdefault("rmsCalibrationError", []).append(
                {"args": [yb, pb, o], "value": ece(yf, pb, bins, strategy, 2)}
            )
            cases.setdefault("maximumCalibrationError", []).append(
                {"args": [yb, pb, o], "value": ece(yf, pb, bins, strategy, "max")}
            )
    # The uniform ECE also from scikit-learn's calibration_curve (no prediction lies on a bin edge).
    prob_true, prob_pred = __import__("sklearn.calibration", fromlist=["calibration_curve"]).calibration_curve(
        yb, pb, n_bins=10
    )
    counts = np.histogram(pb, bins=np.linspace(0, 1, 11))[0]
    counts = counts[counts > 0]
    assert np.isclose(np.sum(counts / n * np.abs(prob_true - prob_pred)), ece(yf, pb, 10, "uniform", 1))

    def debiased(bins):
        order = np.lexsort((np.arange(n), pb))
        b = np.empty(n, int)
        b[order] = (np.arange(n) * bins) // n
        t = 0.0
        for m in range(bins):
            sel = b == m
            c = sel.sum()
            if c == 0:
                continue
            fy, mp = yf[sel].mean(), pb[sel].mean()
            t += c / n * ((fy - mp) ** 2 - (fy * (1 - fy) / (c - 1) if c > 1 else 0))
        return t

    cases["debiasedSquaredCalibrationError"] = [
        {"args": [yb, pb, {"bins": b, "positive": 1}], "value": debiased(b)} for b in (5, 10)
    ]

    def sweep():
        best = 1
        for m in range(1, n + 1):
            order = np.lexsort((np.arange(n), pb))
            b = np.empty(n, int)
            b[order] = (np.arange(n) * m) // n
            freq = [yf[b == k].mean() for k in range(m)]
            if any(freq[k] < freq[k - 1] for k in range(1, m)):
                break
            best = m
        return ece(yf, pb, best, "quantile", 1)

    cases["sweepCalibrationError"] = [{"args": [yb, pb, {"positive": 1}], "value": sweep()}]
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    cases["confidenceCalibrationError"] = [
        {"args": [y, P, {"bins": 10}], "value": ece(correct, conf, 10, "uniform", 1)},
        {"args": [y, P, {"bins": 5, "strategy": "quantile"}], "value": ece(correct, conf, 5, "quantile", 1)},
    ]
    cases["classwiseCalibrationError"] = [
        {
            "args": [y, P, {"bins": 10}],
            "value": np.mean([ece((y == k).astype(float), P[:, k], 10, "uniform", 1) for k in range(K)]),
        }
    ]

    yr = rng.normal(size=30)
    mu = yr + rng.normal(0, 0.5, 30)
    sd = rng.uniform(0.3, 1.5, 30)
    z = (yr - mu) / sd
    crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    cases["crpsGaussian"] = [{"args": [yr, {"mean": mu, "sd": sd}], "value": np.mean(crps)}]
    cases["gaussianLogScore"] = [
        {"args": [yr, {"mean": mu, "sd": sd}], "value": -np.mean(stats.norm.logpdf(yr, mu, sd))}
    ]
    # logScore takes a predictive distribution: the test builds Normal(mean, sd) from these.
    cases["logScore"] = [{"args": [yr, {"mean": mu, "sd": sd}], "value": -np.mean(stats.norm.logpdf(yr, mu, sd))}]
    S = mu[:, None] + sd[:, None] * rng.normal(size=(30, 20))
    m = S.shape[1]
    term1 = np.mean(np.abs(S - yr[:, None]), axis=1)
    pair = np.abs(S[:, :, None] - S[:, None, :]).sum(axis=(1, 2))
    cases["crpsEnsemble"] = [
        {"args": [yr, S], "value": np.mean(term1 - pair / (2 * m * m))},
        {"args": [yr, S, {"fair": True}], "value": np.mean(term1 - pair / (2 * m * (m - 1)))},
    ]
    alpha = 0.2
    lo, hi = mu - 1.2816 * sd, mu + 1.2816 * sd
    iscore = (hi - lo) + 2 / alpha * (lo - yr) * (yr < lo) + 2 / alpha * (yr - hi) * (yr > hi)
    cases["intervalScore"] = [{"args": [yr, {"lower": lo, "upper": hi}, {"alpha": alpha}], "value": np.mean(iscore)}]
    cases["coverage"] = [{"args": [yr, {"lower": lo, "upper": hi}], "value": np.mean((yr >= lo) & (yr <= hi))}]
    logp = np.log(rng.uniform(0.01, 1, 50))
    cases["perplexity"] = [{"args": [logp], "value": np.exp(-np.mean(logp))}]


def curves(rng, cases):
    n = 120
    y = rng.integers(0, 2, n)
    s = y * 0.9 + rng.normal(size=n)
    st = np.round(s, 1)  # with ties
    cases["auroc"] = [
        {"args": [y, s, {"positive": 1}], "value": sk.roc_auc_score(y, s)},
        {"args": [y, st, {"positive": 1}], "value": sk.roc_auc_score(y, st)},
    ]
    K = 4
    yk = rng.integers(0, K, n)
    Pk = softmax(rng.normal(size=(n, K)) + np.eye(K)[yk])
    cases["auroc"] += [
        {"args": [yk, Pk], "value": sk.roc_auc_score(yk, Pk, multi_class="ovr")},
        {
            "args": [yk, Pk, {"average": "weighted"}],
            "value": sk.roc_auc_score(yk, Pk, multi_class="ovr", average="weighted"),
        },
        {"args": [yk, Pk, {"multiClass": "ovo"}], "value": sk.roc_auc_score(yk, Pk, multi_class="ovo")},
    ]
    cs = []
    for mf in (0.1, 0.3, 0.75):
        std = sk.roc_auc_score(y, s, max_fpr=mf)
        lo, hi = 0.5 * mf**2, mf
        cs.append({"args": [y, s, {"maxFpr": mf, "standardised": True}], "value": std})
        cs.append({"args": [y, s, {"maxFpr": mf, "standardised": False}], "value": lo + (2 * std - 1) * (hi - lo)})
    cases["partialAuroc"] = cs
    Yk = label_binarize(yk, classes=list(range(K)))
    cases["averagePrecision"] = [
        {"args": [y, s, {"positive": 1}], "value": sk.average_precision_score(y, s)},
        {"args": [y, st, {"positive": 1}], "value": sk.average_precision_score(y, st)},
        {"args": [yk, Pk], "value": sk.average_precision_score(Yk, Pk, average="macro")},
        {"args": [yk, Pk, {"average": "weighted"}], "value": sk.average_precision_score(Yk, Pk, average="weighted")},
    ]
    fpr, tpr, _ = sk.roc_curve(y, s, drop_intermediate=False)
    d = 1 - tpr - fpr
    i = int(np.argmax(d <= 0))
    t = d[i - 1] / (d[i - 1] - d[i])
    cases["eer"] = [{"args": [y, s, {"positive": 1}], "value": fpr[i - 1] + t * (fpr[i] - fpr[i - 1])}]
    cases["specificityAtSensitivity"] = [
        {"args": [y, s, {"sensitivity": v, "positive": 1}], "value": 1 - fpr[tpr >= v].min()} for v in (0.5, 0.8, 0.95)
    ]
    cases["tprAtFpr"] = [
        {"args": [y, s, {"fpr": v, "positive": 1}], "value": tpr[fpr <= v].max()} for v in (0.05, 0.2, 0.5)
    ]
    prec, rec, _ = sk.precision_recall_curve(y, s, drop_intermediate=False)
    ok = rec > 0
    cases["recallAtPrecision"] = [
        {"args": [y, s, {"precision": v, "positive": 1}], "value": rec[ok & (prec >= v)].max()}
        for v in (0.6, 0.75, 0.9)
    ]


def ranking(rng, cases):
    Q, L = 5, 10
    rel = rng.integers(0, 4, size=(Q, L)).astype(float)
    rel[2] = 0
    rel[2, 7] = 2
    scores = rng.normal(size=(Q, L)) + 0.4 * rel
    order = np.argsort(-scores, axis=1, kind="stable")
    g = np.take_along_axis(rel, order, axis=1)
    b = (g > 0).astype(float)
    for k in (3, 5, L):
        cases.setdefault("precisionAtK", []).append(
            {"args": [rel, scores, {"k": k}], "value": np.mean(b[:, :k].sum(1) / k)}
        )
        cases.setdefault("recallAtK", []).append(
            {"args": [rel, scores, {"k": k}], "value": np.mean(b[:, :k].sum(1) / b.sum(1))}
        )
        cases.setdefault("hitRate", []).append({"args": [rel, scores, {"k": k}], "value": np.mean(b[:, :k].sum(1) > 0)})
        rr = [1 / (np.argmax(r[:k]) + 1) if r[:k].any() else 0.0 for r in b]
        cases.setdefault("meanReciprocalRank", []).append({"args": [rel, scores, {"k": k}], "value": np.mean(rr)})
        ap = [np.sum(np.cumsum(r[:k]) / np.arange(1, k + 1) * r[:k]) / r.sum() for r in b]
        cases.setdefault("meanAveragePrecision", []).append({"args": [rel, scores, {"k": k}], "value": np.mean(ap)})
        cases.setdefault("dcg", []).append(
            {
                "args": [rel, scores, {"k": k, "gain": "linear"}],
                "value": np.mean([sk.dcg_score([rel[q]], [scores[q]], k=k) for q in range(Q)]),
            }
        )
        cases.setdefault("ndcg", []).append(
            {"args": [rel, scores, {"k": k, "gain": "linear"}], "value": sk.ndcg_score(rel, scores, k=k)}
        )
        disc = 1 / np.log2(np.arange(2, k + 2))
        exp_dcg = ((2 ** g[:, :k] - 1) * disc).sum(1)
        ideal = np.sort(rel, axis=1)[:, ::-1]
        exp_idcg = ((2 ** ideal[:, :k] - 1) * disc).sum(1)
        cases["dcg"].append({"args": [rel, scores, {"k": k}], "value": np.mean(exp_dcg)})
        cases["ndcg"].append({"args": [rel, scores, {"k": k}], "value": np.mean(exp_dcg / exp_idcg)})
        err = []
        for r in g:
            gmax = r.max()
            stop, total = 1.0, 0.0
            for i in range(k):
                R = (2 ** r[i] - 1) / 2**gmax
                total += stop * R / (i + 1)
                stop *= 1 - R
            err.append(total)
        cases.setdefault("expectedReciprocalRank", []).append({"args": [rel, scores, {"k": k}], "value": np.mean(err)})
    cases["rPrecision"] = [{"args": [rel, scores], "value": np.mean([r[: int(r.sum())].sum() / r.sum() for r in b])}]
    # Ties in the scores: scikit-learn's DCG gives each tied item the mean gain of its group.
    st = np.round(scores, 0)
    cases["dcg"].append(
        {"args": [rel, st, {"gain": "linear"}], "value": np.mean([sk.dcg_score([rel[q]], [st[q]]) for q in range(Q)])}
    )
    cases["ndcg"].append({"args": [rel, st, {"gain": "linear"}], "value": sk.ndcg_score(rel, st)})


def clustering(rng, cases):
    n = 90
    a = rng.integers(0, 4, n)
    b = np.where(rng.uniform(size=n) < 0.6, a, rng.integers(0, 5, n))
    sk_pairs = {
        "randIndex": sk.rand_score,
        "adjustedRandIndex": sk.adjusted_rand_score,
        "fowlkesMallows": sk.fowlkes_mallows_score,
        "mutualInformationScore": sk.mutual_info_score,
        "homogeneity": sk.homogeneity_score,
        "completeness": sk.completeness_score,
        "vMeasure": sk.v_measure_score,
    }
    for key, f in sk_pairs.items():
        cases[key] = [{"args": [a, b], "value": f(a, b)}]
    cases["vMeasure"].append({"args": [a, b, {"beta": 2}], "value": sk.v_measure_score(a, b, beta=2)})
    for key, f in {
        "normalisedMutualInformation": sk.normalized_mutual_info_score,
        "adjustedMutualInformation": sk.adjusted_mutual_info_score,
    }.items():
        cases[key] = [
            {"args": [a, b, {"average": m}], "value": f(a, b, average_method=m)}
            for m in ("arithmetic", "geometric", "min", "max")
        ]
    ha = stats.entropy(np.bincount(a))
    hb = stats.entropy(np.bincount(b))
    cases["variationOfInformation"] = [{"args": [a, b], "value": ha + hb - 2 * sk.mutual_info_score(a, b)}]

    centres = np.array([[0, 0], [4, 0], [0, 4]])
    lab = rng.integers(0, 3, 60)
    X = centres[lab] + rng.normal(size=(60, 2))
    cases["silhouetteScore"] = [{"args": [X, lab], "value": sk.silhouette_score(X, lab)}]
    cases["calinskiHarabasz"] = [{"args": [X, lab], "value": sk.calinski_harabasz_score(X, lab)}]
    cases["daviesBouldin"] = [{"args": [X, lab], "value": sk.davies_bouldin_score(X, lab)}]
    D = distance.squareform(distance.pdist(X))
    same = lab[:, None] == lab[None, :]
    cases["dunnIndex"] = [{"args": [X, lab], "value": D[~same].min() / D[same].max()}]


def krippendorff(data, level):
    """Krippendorff's α from a raters × units matrix with NaN for missing (Krippendorff 2011)."""
    values = np.unique(data[~np.isnan(data)])
    index = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V))
    for u in data.T:
        r = u[~np.isnan(u)]
        m = len(r)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[index[r[i]], index[r[j]]] += 1 / (m - 1)
    nc = o.sum(axis=1)
    N = nc.sum()

    def delta(c, k):
        vc, vk = values[c], values[k]
        if level == "nominal":
            return float(c != k)
        if level == "interval":
            return (vc - vk) ** 2
        if level == "ratio":
            return ((vc - vk) / (vc + vk)) ** 2
        lo, hi = min(c, k), max(c, k)
        return (nc[lo : hi + 1].sum() - (nc[c] + nc[k]) / 2) ** 2

    d = np.array([[delta(c, k) for k in range(V)] for c in range(V)])
    Do = (o * d).sum()
    De = (np.outer(nc, nc) * d).sum() / (N - 1)
    return 1 - Do / De


def agreement(rng, cases):
    x = rng.normal(size=40)
    yv = 0.6 * x + rng.normal(0, 0.8, 40)
    yt = np.round(yv, 0)  # ties for the rank correlations
    cases["pearsonCorrelation"] = [{"args": [x, yv], "value": statistic(stats.pearsonr(x, yv))}]
    cases["spearmanCorrelation"] = [
        {"args": [x, yv], "value": statistic(stats.spearmanr(x, yv))},
        {"args": [x, yt], "value": statistic(stats.spearmanr(x, yt))},
    ]
    cases["kendallCorrelation"] = [
        {"args": [x, yv], "value": stats.kendalltau(x, yv).statistic},
        {"args": [x, yt], "value": stats.kendalltau(x, yt).statistic},
    ]
    mx, my = x.mean(), yv.mean()
    cov = np.mean((x - mx) * (yv - my))
    cases["concordanceCorrelation"] = [{"args": [x, yv], "value": 2 * cov / (x.var() + yv.var() + (mx - my) ** 2)}]

    # Fleiss (1971) via Wikipedia's worked example: 10 items, 14 raters, 5 categories; κ = 0.210.
    fleiss = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    N, nr = fleiss.shape[0], fleiss[0].sum()
    pj = fleiss.sum(0) / (N * nr)
    Pi = (np.sum(fleiss**2, axis=1) - nr) / (nr * (nr - 1))
    kappa = (Pi.mean() - np.sum(pj**2)) / (1 - np.sum(pj**2))
    assert round(kappa, 3) == 0.210
    cases["fleissKappa"] = [{"args": [fleiss], "value": kappa}]

    # Krippendorff (2011), section C: 4 observers × 12 units with missing values; published α to three places.
    nan = np.nan
    kd = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
            [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
        ]
    )
    published = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}
    kcases = []
    for level, value in published.items():
        alpha = krippendorff(kd, level)
        assert round(alpha, 3) == value, (level, alpha)
        kcases.append({"args": [kd, {"level": level}], "value": alpha})
    cases["krippendorffAlpha"] = kcases

    # Shrout and Fleiss (1979), Table 2: 6 targets × 4 judges; published ICCs to two places.
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    n, k = sf.shape
    g = sf.mean()
    ssr = k * np.sum((sf.mean(1) - g) ** 2)
    ssc = n * np.sum((sf.mean(0) - g) ** 2)
    sse = np.sum((sf - g) ** 2) - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    msw = (ssc + sse) / (n * (k - 1))
    icc = {
        "ICC1": (msr - msw) / (msr + (k - 1) * msw),
        "ICC2": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
        "ICC3": (msr - mse) / (msr + (k - 1) * mse),
        "ICC1k": (msr - msw) / msr,
        "ICC2k": (msr - mse) / (msr + (msc - mse) / n),
        "ICC3k": (msr - mse) / msr,
    }
    for form, value in {"ICC1": 0.17, "ICC2": 0.29, "ICC3": 0.71, "ICC1k": 0.44, "ICC2k": 0.62, "ICC3k": 0.91}.items():
        assert round(icc[form], 2) == value, (form, icc[form])
    cases["intraclassCorrelation"] = [{"args": [sf, {"form": f}], "value": v} for f, v in icc.items()]

    table = rng.integers(1, 20, size=(3, 4))
    cases["cramersV"] = [{"args": [table], "value": contingency.association(table, method="cramer")}]
    cases["tschuprowT"] = [{"args": [table], "value": contingency.association(table, method="tschuprow")}]
    cases["contingencyCoefficient"] = [{"args": [table], "value": contingency.association(table, method="pearson")}]
    pxy = table / table.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    mi = np.sum(pxy * np.log(pxy / np.outer(px, py)))
    cases["theilsU"] = [
        {"args": [table], "value": mi / stats.entropy(py)},
        {"args": [table, {"of": "rows"}], "value": mi / stats.entropy(px)},
    ]


def distances(rng, cases):
    u = rng.normal(size=6)
    v = rng.normal(size=6)
    cases["minkowskiDistance"] = [{"args": [u, v, {"p": p}], "value": distance.minkowski(u, v, p)} for p in (1, 1.5, 3)]
    cases["euclideanDistance"] = [{"args": [u, v], "value": distance.euclidean(u, v)}]
    cases["manhattanDistance"] = [{"args": [u, v], "value": distance.cityblock(u, v)}]
    cases["chebyshevDistance"] = [{"args": [u, v], "value": distance.chebyshev(u, v)}]
    cases["cosineDistance"] = [{"args": [u, v], "value": distance.cosine(u, v)}]
    cases["cosineSimilarity"] = [{"args": [u, v], "value": 1 - distance.cosine(u, v)}]
    cases["angularDistance"] = [{"args": [u, v], "value": np.arccos(1 - distance.cosine(u, v)) / np.pi}]
    M = rng.normal(size=(6, 6))
    C = M @ M.T + np.eye(6)
    cases["mahalanobisDistance"] = [
        {"args": [u, v, {"covariance": C}], "value": distance.mahalanobis(u, v, np.linalg.inv(C))}
    ]
    A = rng.normal(size=(10, 3))
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    B = 2.5 * A @ R + 1 + 0.1 * rng.normal(size=(10, 3))
    cases["procrustesDisparity"] = [{"args": [A, B], "value": procrustes(A, B)[2]}]
    Y = rng.normal(size=(7, 3)) + 0.5
    h = max(distance.directed_hausdorff(A, Y)[0], distance.directed_hausdorff(Y, A)[0])
    cases["hausdorffDistance"] = [{"args": [A, Y], "value": h}]


def representation(rng, cases):
    """Wang & Isola (2020)'s reference code: (x - y).norm(dim=1).pow(alpha).mean() and
    pdist(x).pow(2).mul(-t).exp().mean().log(), on unit-norm embeddings."""
    x = rng.normal(size=(9, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = x + 0.3 * rng.normal(size=(9, 3))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    gap = np.linalg.norm(x - y, axis=1)
    cases["alignment"] = [
        {"args": [x, y], "value": np.mean(gap**2)},
        {"args": [x, y, {"alpha": 1}], "value": np.mean(gap)},
    ]
    d2 = distance.pdist(x) ** 2
    cases["uniformity"] = [
        {"args": [x], "value": np.log(np.mean(np.exp(-2 * d2)))},
        {"args": [x, {"t": 0.5}], "value": np.log(np.mean(np.exp(-0.5 * d2)))},
    ]


def metric_cases() -> dict[str, list[dict[str, object]]]:
    rng = np.random.default_rng(20261001)
    cases: dict[str, list[dict[str, object]]] = {}
    for part in (
        classification,
        ordinal,
        regression,
        probabilistic,
        curves,
        ranking,
        clustering,
        agreement,
        distances,
        representation,
    ):
        part(rng, cases)
    return cases
