"""Golden values for aifn-compute/signal/similarity: the matrix profile and MASS distance profiles from stumpy (`stump`,
`mass`), and plain-numpy references for banded DTW (the O(nm) recursion), the Keogh envelope and LB_Keogh, PAA, the
SAX breakpoints (scipy's normal quantiles), SAX words and MINDIST."""

import numpy as np
import stumpy
from scipy.stats import norm


def planted(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    """A random walk with one shape planted twice (a motif) and one spike (a discord)."""
    x = np.cumsum(rng.normal(size=n)) * 0.3
    shape = np.sin(np.linspace(0, 3 * np.pi, m)) * 3
    for at in (80, 400):
        x[at : at + m] += shape
    x[260:290] += np.sin(np.linspace(0, 12 * np.pi, 30)) * 2.5
    return x


def dtw(a: np.ndarray, b: np.ndarray, window: int, cost: str) -> tuple[float, np.ndarray]:
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            c = (a[i - 1] - b[j - 1]) ** 2 if cost == "squared" else abs(a[i - 1] - b[j - 1])
            d[i, j] = c + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    total = d[n, m]
    return (float(np.sqrt(total)) if cost == "squared" else float(total)), d[1:, 1:]


def envelope(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([x[max(0, i - w) : i + w + 1].max() for i in range(len(x))])
    lo = np.array([x[max(0, i - w) : i + w + 1].min() for i in range(len(x))])
    return up, lo


def paa(x: np.ndarray, w: int) -> np.ndarray:
    # Fractional frames: repeat each sample w times, then average blocks of n.
    n = len(x)
    return np.repeat(x, w).reshape(w, n).mean(axis=1)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261002)
    m = 40
    x = planted(rng, 600, m)
    mp = stumpy.stump(x, m)
    query = x[80 : 80 + m] + 0.1 * rng.normal(size=m)
    out: dict[str, object] = {
        "series": x,
        "m": m,
        "profile": mp[:, 0].astype(float),
        "index": mp[:, 1].astype(int),
        "query": query,
        "mass": stumpy.mass(query, x),
        "massRaw": stumpy.mass(query, x, normalize=False),
    }
    a = np.sin(np.linspace(0, 6, 50)) + 0.1 * rng.normal(size=50)
    b = np.sin(np.linspace(0.5, 6.5, 50)) + 0.1 * rng.normal(size=50)
    c = np.cos(np.linspace(0, 5, 43))
    dtw_cases = []
    for p, q, w, cost in ((a, b, 5, "squared"), (a, b, 50, "absolute"), (a, c, 10, "squared"), (a, b, 0, "squared")):
        dist, table = dtw(p, q, w, cost)
        dtw_cases.append({"x": p, "y": q, "window": w, "cost": cost, "distance": dist, "accumulated": table})
    out["dtw"] = dtw_cases
    up, lo = envelope(b, 5)
    lbk = float(np.sqrt(np.sum(np.where(a > up, (a - up) ** 2, np.where(a < lo, (a - lo) ** 2, 0.0)))))
    out["keogh"] = {"x": a, "y": b, "window": 5, "upper": up, "lower": lo, "lbKeogh": lbk}
    z = (a - a.mean()) / a.std()
    beta = norm.ppf(np.arange(1, 6) / 6)
    word = np.searchsorted(beta, paa(z, 8), side="right")
    z2 = (b - b.mean()) / b.std()
    word2 = np.searchsorted(beta, paa(z2, 8), side="right")
    dist = np.zeros((6, 6))
    for r in range(6):
        for s in range(6):
            if abs(r - s) > 1:
                dist[r, s] = beta[max(r, s) - 1] - beta[min(r, s)]
    mindist = float(np.sqrt(50 / 8) * np.sqrt(np.sum(dist[word, word2] ** 2)))
    out["sax"] = {
        "x": a,
        "y": b,
        "paa7": paa(a, 7),
        "paa10": paa(a, 10),
        "breakpoints6": beta,
        "word": word,
        "word2": word2,
        "mindist": mindist,
        "euclidean": float(np.linalg.norm(z - z2)),
    }
    return out
