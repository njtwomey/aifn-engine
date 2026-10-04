"""Reference values for aifn-compute/probability/markov from numpy: the stationary distribution as the left eigenvector
of
eigenvalue 1, the fundamental matrix N = inv(I - Q) with B = N R and t = N 1, hitting times by numpy.linalg.solve,
the eigenvalue moduli of P, and the worst total-variation distance max_x |P^t(x, .) - pi|_TV by matrix powers."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type Matrix = NDArray[np.float64]


def random_chain(rng: np.random.Generator, n: int, sparsity: float) -> Matrix:
    """A random irreducible, aperiodic chain: a ring of positive steps plus random extra edges and self-loops."""
    p = rng.uniform(0, 1, (n, n)) * (rng.uniform(0, 1, (n, n)) > sparsity)
    for i in range(n):
        p[i, (i + 1) % n] += 0.3
        p[i, i] += 0.1
    return p / p.sum(axis=1, keepdims=True)


def stationary(p: Matrix) -> Matrix:
    values, vectors = np.linalg.eig(p.T)
    k = int(np.argmin(np.abs(values - 1)))
    v = np.real(vectors[:, k])
    return v / v.sum()


def worst_tv(p: Matrix, pi: Matrix, steps: int) -> list[float]:
    out: list[float] = []
    pt = np.eye(p.shape[0])
    for _ in range(steps + 1):
        out.append(float(0.5 * np.abs(pt - pi[None, :]).sum(axis=1).max()))
        pt = pt @ p
    return out


def gamblers_ruin(target: int, win: float) -> Matrix:
    p = np.zeros((target + 1, target + 1))
    p[0, 0] = p[target, target] = 1
    for i in range(1, target):
        p[i, i + 1] = win
        p[i, i - 1] = 1 - win
    return p


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    ergodic: list[dict[str, Any]] = []
    for n, sparsity in [(3, 0.0), (5, 0.5), (8, 0.7)]:
        p = random_chain(rng, n, sparsity)
        pi = stationary(p)
        moduli = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
        target = [n - 1]
        rest = [i for i in range(n) if i not in target]
        a = np.eye(len(rest)) - p[np.ix_(rest, rest)]
        k = np.zeros(n)
        k[rest] = np.linalg.solve(a, np.ones(len(rest)))
        ergodic.append(
            {"P": p, "stationary": pi, "moduli": moduli, "target": target, "hitting": k, "worst": worst_tv(p, pi, 12)}
        )
    absorbing: list[dict[str, Any]] = []
    for target, win in [(4, 0.5), (6, 0.45), (10, 0.6)]:
        p = gamblers_ruin(target, win)
        transient = list(range(1, target))
        q = p[np.ix_(transient, transient)]
        r = p[np.ix_(transient, [0, target])]
        fundamental = np.linalg.inv(np.eye(len(transient)) - q)
        steps = fundamental @ np.ones(len(transient))
        variance = (2 * fundamental - np.eye(len(transient))) @ steps - steps**2
        absorbing.append(
            {
                "target": target,
                "win": win,
                "P": p,
                "fundamental": fundamental,
                "probabilities": fundamental @ r,
                "expectedSteps": steps,
                "varianceSteps": variance,
            }
        )
    return {"ergodic": ergodic, "absorbing": absorbing}
