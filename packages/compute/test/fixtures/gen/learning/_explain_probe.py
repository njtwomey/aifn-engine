"""Reference values for aifn-compute/learning/explain's linear probe (the classifier behind concept activation vectors):
scikit-learn's L2-regularised logistic regression with an unpenalised intercept, whose objective
C Σ loss + ½‖w‖² equals aifn's mean loss + (λ/2)‖w‖² at C = 1/(nλ)."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def probe_cases() -> dict[str, Any]:
    rng = np.random.default_rng(31)
    positive = rng.normal(size=(40, 5)) + np.array([0.8, -0.4, 0.0, 0.3, 0.1])
    negative = rng.normal(size=(50, 5))
    x = np.vstack([positive, negative])
    y = np.r_[np.ones(40), np.zeros(50)]
    l2 = 0.05
    model = LogisticRegression(C=1 / (len(y) * l2), tol=1e-12, max_iter=10000).fit(x, y)
    return {
        "positive": positive,
        "negative": negative,
        "l2": l2,
        "weights": model.coef_[0],
        "bias": model.intercept_[0],
    }
