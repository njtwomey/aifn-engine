"""Golden values for aifn-compute/learning/metrics (ordinal metrics on scores and probabilities): the ordinal C-index as
(Somers' D + 1)/2 from scipy, the binary AUROC from scikit-learn, and the ranked probability score computed directly in
numpy; and, under `metrics`, reference cases for every registered metric (gen/learning/_metric_cases.py)."""

import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _metric_cases import metric_cases


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    n, k = 40, 4
    y = rng.integers(0, k, size=n)
    # Scores rounded to one decimal so that ties occur.
    score = np.round(0.6 * y + rng.normal(size=n), 1)
    somers = stats.somersd(y, score).statistic  # D(score | y): rows of the table are y
    logits = rng.normal(size=(n, k)) + np.eye(k)[y] * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)[:, :-1]
    obs = (y[:, None] <= np.arange(k - 1)[None, :]).astype(float)
    rps = np.mean(np.sum((cdf - obs) ** 2, axis=1) / (k - 1))
    yb = (y >= 2).astype(int)
    pb = rng.uniform(size=n)
    return {
        "y": y,
        "score": score,
        "c_index": (somers + 1) / 2,
        "probabilities": p,
        "rps": rps,
        "binary": {
            "y": yb,
            "score": score,
            "auroc": roc_auc_score(yb, score),
            "p": pb,
            "brier": brier_score_loss(yb, pb),
        },
        "metrics": metric_cases(),
    }
