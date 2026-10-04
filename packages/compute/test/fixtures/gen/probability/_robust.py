"""Reference values for the robust estimators of aifn-compute/probability/stats from scikit-learn's MinCovDet (FastMCD
with
the consistency correction and reweighting): the raw and reweighted location and covariance, the support, the squared
Mahalanobis distances and the raw log determinant, on contaminated Gaussian samples in two and three dimensions."""

from typing import Any

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.covariance._robust_covariance import _consistency_factor  # pyright: ignore[reportPrivateUsage]


def mcd_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(3)
    out: list[dict[str, Any]] = []
    for name, p, n, k in [("plane", 2, 60, 8), ("space", 3, 80, 12)]:
        a = rng.normal(size=(p, p))
        x = rng.normal(size=(n, p)) @ a + rng.normal(size=p)
        x[:k] = rng.normal(6, 1, size=(k, p))
        m = MinCovDet(random_state=0).fit(x)
        out.append(
            {
                "name": name,
                "x": x,
                "location": m.location_,
                "covariance": m.covariance_,
                "rawLocation": m.raw_location_,
                # scikit-learn keeps the raw covariance uncorrected; aifn reports it with the consistency factor.
                "rawCovariance": m.raw_covariance_ * _consistency_factor(p, len(np.flatnonzero(m.raw_support_)) / n),
                "rawLogDeterminant": np.linalg.slogdet(m.raw_covariance_)[1],
                "support": np.flatnonzero(m.raw_support_),
                "distances": m.dist_,
            }
        )
    return out
