"""Reference values for aifn-methods/optimisation/multi-objective: Pareto fronts, hypervolume and EHVI.

Everything is brute force, independent of the engine's sweep, slicing and strip formulas:

- The front: every pair of points compared.
- The hypervolume: the grid of every distinct coordinate (and the reference point) cuts the box into cells; a cell is
  dominated when some point is at or below its lower corner, and the volume is the sum over dominated cells. Checked
  against inclusion and exclusion over subsets for one small case.
- The improvement: the grid volume with the new point less the volume without it.
- The expected improvement: EHVI is the integral over the non-dominated part of the box of P(y <= z), and over a cell
  [a, b] the integral of prod_j Phi((z_j - mu_j) / sd_j) factorises into prod_j (Psi_j(b_j) - Psi_j(a_j)), with
  Psi(c) = E[(c - y)^+] (Psi(-inf) = 0). This holds for any number of objectives, so it gives exact values for M = 3
  too; for M = 2 it is checked here against a Monte Carlo average over 10^6 draws.
"""

from itertools import combinations

import numpy as np
from scipy.stats import norm


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def front_indices(points: np.ndarray) -> list[int]:
    n = len(points)
    keep = [i for i in range(n) if not any(dominates(points[k], points[i]) for k in range(n) if k != i)]
    return sorted(keep, key=lambda i: (tuple(points[i]), i))


def grid_cells(points: np.ndarray, ref: np.ndarray, lower: float):
    """Each cell of the grid of distinct coordinates below the reference point: lower and upper corners."""
    m = len(ref)
    axes = []
    for j in range(m):
        v = sorted({float(x) for x in points[:, j] if x < ref[j]} | {float(ref[j])}) if len(points) else [float(ref[j])]
        axes.append([lower, *v])
    for idx in np.ndindex(*[len(a) - 1 for a in axes]):
        lo = np.array([axes[j][i] for j, i in enumerate(idx)])
        hi = np.array([axes[j][i + 1] for j, i in enumerate(idx)])
        yield lo, hi


def hv_grid(points: np.ndarray, ref: np.ndarray) -> float:
    points = np.array([p for p in points if np.all(p < ref)]).reshape(-1, len(ref))
    if len(points) == 0:
        return 0.0
    low = float(points.min()) - 1.0
    total = 0.0
    for lo, hi in grid_cells(points, ref, low):
        if np.any(np.all(points <= lo, axis=1)):
            total += float(np.prod(hi - lo))
    return total


def hv_inclusion_exclusion(points: np.ndarray, ref: np.ndarray) -> float:
    total = 0.0
    for k in range(1, len(points) + 1):
        for sub in combinations(range(len(points)), k):
            corner = points[list(sub)].max(axis=0)
            total += (-1) ** (k + 1) * float(np.prod(np.maximum(ref - corner, 0)))
    return total


def psi(c: float, mu: float, sd: float) -> float:
    if c == -np.inf:
        return 0.0
    if sd == 0:
        return max(c - mu, 0.0)
    z = (c - mu) / sd
    return (c - mu) * norm.cdf(z) + sd * norm.pdf(z)


def ehvi_cells(front: np.ndarray, ref: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    """The integral of P(y <= z) over the non-dominated cells of the box below the reference point."""
    inside = np.array([p for p in front if np.all(p < ref)]).reshape(-1, len(ref))
    total = 0.0
    for lo, hi in grid_cells(inside, ref, -np.inf):
        if len(inside) and np.any(np.all(inside <= lo, axis=1)):
            continue
        total += float(np.prod([psi(hi[j], mu[j], sd[j]) - psi(lo[j], mu[j], sd[j]) for j in range(len(ref))]))
    return total


def ehvi_monte_carlo(front: np.ndarray, ref: np.ndarray, mu: np.ndarray, sd: np.ndarray, rng, n: int) -> float:
    """For two objectives: the improvement of each draw by the vertical strips, vectorised over the draws."""
    inside = np.array(sorted(tuple(p) for p in front if np.all(p < ref))).reshape(-1, 2)
    inside = inside[[not any(dominates(q, p) for q in inside) for p in inside]]
    y = mu + sd * rng.standard_normal((n, 2))
    xs = np.concatenate([[-np.inf], inside[:, 0], [ref[0]]])
    us = np.concatenate([[ref[1]], inside[:, 1]])
    total = np.zeros(n)
    for i in range(len(us)):
        width = np.maximum(xs[i + 1] - np.maximum(xs[i], y[:, 0]), 0)
        total += width * np.maximum(us[i] - y[:, 1], 0)
    return float(total.mean())


def chebyshev(y: np.ndarray, w: np.ndarray, rho: float, ideal=None, nadir=None) -> np.ndarray:
    v = y.copy()
    if ideal is not None:
        v = v - ideal
        if nadir is not None:
            v = v / (nadir - ideal)
    return (w * v).max(axis=1) + rho * (w * v).sum(axis=1)


def cases() -> dict:
    rng = np.random.default_rng(20061001)

    # Fronts: integer points (ties and repeats) in 2-D and 3-D, and a random cloud.
    p2 = rng.integers(0, 6, size=(25, 2)).astype(float)
    p3 = rng.integers(0, 5, size=(30, 3)).astype(float)
    cloud = rng.normal(size=(40, 2))
    pareto = [{"points": p, "indices": front_indices(p)} for p in (p2, p3, cloud)]
    # Maximising the second objective of the cloud.
    flipped = cloud * np.array([1.0, -1.0])
    pareto_mixed = {"points": cloud, "indices": front_indices(flipped)}

    # Hypervolume: 2-D, 3-D and 4-D sets (with dominated points and points beyond the reference point).
    hv = []
    for m, n in ((2, 12), (3, 10), (3, 18), (4, 8)):
        pts = rng.uniform(0, 1, size=(n, m))
        pts[0] = 1.05  # beyond the reference point in every objective
        ref = np.full(m, 1.0)
        hv.append({"points": pts, "reference": ref, "hv": hv_grid(pts, ref)})
    small = rng.uniform(0, 1, size=(7, 3))
    hv_ie = {
        "points": small,
        "reference": [1.1, 1.2, 1.3],
        "hv": hv_inclusion_exclusion(small, np.array([1.1, 1.2, 1.3])),
    }
    assert abs(hv_ie["hv"] - hv_grid(small, np.array([1.1, 1.2, 1.3]))) < 1e-12

    # Improvements of candidates over a front, 2-D and 3-D.
    hvi = []
    for m, n in ((2, 6), (3, 7)):
        front = rng.uniform(0, 1, size=(n, m))
        ref = np.full(m, 1.2)
        y = rng.uniform(-0.1, 1.3, size=(6, m))
        base = hv_grid(front, ref)
        hvi.append(
            {
                "front": front,
                "reference": ref,
                "y": y,
                "hvi": [hv_grid(np.vstack([front, yi]), ref) - base for yi in y],
            }
        )

    # EHVI, two objectives: the cell integral (exact) and a Monte Carlo check.
    front2 = np.array([[0.1, 0.9], [0.3, 0.6], [0.5, 0.5], [0.7, 0.2], [0.9, 0.1], [0.6, 0.7]])
    ref2 = np.array([1.0, 1.0])
    mu2 = np.array([[0.4, 0.4], [0.2, 0.3], [0.8, 0.8], [1.1, 0.0], [0.5, 0.5]])
    sd2 = np.array([[0.1, 0.2], [0.3, 0.05], [0.2, 0.2], [0.15, 0.1], [0.0, 0.0]])
    exact2 = [ehvi_cells(front2, ref2, mu2[i], sd2[i]) for i in range(len(mu2))]
    mc2 = [ehvi_monte_carlo(front2, ref2, mu2[i], sd2[i], rng, 1_000_000) for i in range(len(mu2))]
    for e, m in zip(exact2, mc2, strict=True):
        assert abs(e - m) < 2e-3 * max(1.0, e) + 1e-4, (e, m)
    empty2 = ehvi_cells(np.zeros((0, 2)), ref2, mu2[0], sd2[0])

    # EHVI, three objectives: exact by the cell integral, for the engine's Monte Carlo estimate.
    front3 = rng.uniform(0, 1, size=(5, 3))
    ref3 = np.full(3, 1.1)
    mu3 = np.array([[0.4, 0.4, 0.4], [0.2, 0.8, 0.5]])
    sd3 = np.array([[0.2, 0.2, 0.2], [0.1, 0.3, 0.2]])
    exact3 = [ehvi_cells(front3, ref3, mu3[i], sd3[i]) for i in range(len(mu3))]

    # One objective: the expected improvement on min(r, best).
    front1 = np.array([[0.4], [0.7]])
    mu1 = np.array([[0.3], [0.6], [0.5]])
    sd1 = np.array([[0.1], [0.2], [0.0]])
    exact1 = [psi(0.4, mu1[i, 0], sd1[i, 0]) for i in range(3)]

    # Chebyshev scalarisation.
    y = rng.uniform(0, 10, size=(8, 3))
    w = np.array([0.2, 0.5, 0.3])
    ideal = np.array([0.0, 1.0, 2.0])
    nadir = np.array([10.0, 9.0, 12.0])

    return {
        "pareto": pareto,
        "paretoMixed": pareto_mixed,
        "hypervolume": hv,
        "hypervolumeInclusionExclusion": hv_ie,
        "improvement": hvi,
        "ehvi2": {"front": front2, "reference": ref2, "mean": mu2, "sd": sd2, "exact": exact2, "monteCarlo": mc2},
        "ehvi2Empty": empty2,
        "ehvi3": {"front": front3, "reference": ref3, "mean": mu3, "sd": sd3, "exact": exact3},
        "ehvi1": {"front": front1, "reference": [1.0], "mean": mu1, "sd": sd1, "exact": exact1},
        "chebyshev": {
            "y": y,
            "weights": w,
            "ideal": ideal,
            "nadir": nadir,
            "plain": chebyshev(y, w, 0.05),
            "normalised": chebyshev(y, w, 0.1, ideal, nadir),
        },
    }
