"""Golden values for aifn-methods/learning/preprocessing's imbalanced-class resamplers (imbalanced-learn): the
borderline-SMOTE danger set, ADASYN's per-row synthetic counts, Tomek links and the class counts after resampling,
on seeded overlapping Gaussian classes. Random draws differ between libraries, so only the deterministic parts and
the counts are compared."""

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.neighbors import NearestNeighbors


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 1, size=(150, 2)), rng.normal(1.2, 0.8, size=(30, 2))])
    y = np.array([0] * 150 + [1] * 30)
    minority = np.flatnonzero(y == 1)
    # Borderline-SMOTE: m = 10 nearest neighbours in all the data, excluding the point itself.
    nn = NearestNeighbors(n_neighbors=11).fit(x)
    idx = nn.kneighbors(x[minority], return_distance=False)[:, 1:]
    other = (y[idx] != 1).sum(axis=1)
    danger = (other >= 5) & (other < 10)
    noise = other == 10
    # ADASYN: k = 5 in all the data; counts rint(ratio * G).
    nn5 = NearestNeighbors(n_neighbors=6).fit(x)
    idx5 = nn5.kneighbors(x[minority], return_distance=False)[:, 1:]
    ratio = (y[idx5] != 1).sum(axis=1) / 5
    ratio = ratio / ratio.sum()
    counts = np.rint(ratio * (150 - 30)).astype(int)
    tl = TomekLinks()
    tl.fit_resample(x, y)
    kept = tl.sample_indices_
    resampled = {
        "smote": np.bincount(SMOTE(random_state=0).fit_resample(x, y)[1]),
        "borderline": np.bincount(BorderlineSMOTE(random_state=0).fit_resample(x, y)[1]),
        "adasyn": np.bincount(ADASYN(random_state=0).fit_resample(x, y)[1]),
    }
    return {
        "x": x,
        "y": y,
        "danger": danger.astype(int),
        "noise": noise.astype(int),
        "adasyn_counts": counts,
        "tomek_kept": kept,
        "counts": resampled,
    }
