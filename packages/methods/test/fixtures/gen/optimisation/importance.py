"""Reference values for aifn-methods/optimisation/importance: the exact functional ANOVA of a tabulated surface.

Each term is computed directly with numpy from its definition (Hutter, Hoos and Leyton-Brown, 2014): the marginal
m_V is the weighted mean of the table over the axes outside V (np.tensordot with each axis's weights), and
f_U = sum over V ⊆ U of (-1)^{|U|-|V|} m_V, broadcast to U's grid. A random 3-axis table with unequal weights, and a
2-axis table with equal weights.
"""

from itertools import combinations

import numpy as np


def marginal(table: np.ndarray, weights: list[np.ndarray], keep: tuple[int, ...]) -> np.ndarray:
    """The weighted mean of the table over the axes not in `keep`, as an array over the kept axes (in order)."""
    out = table
    for j in reversed(range(table.ndim)):
        if j not in keep:
            out = np.tensordot(out, weights[j], axes=([j], [0]))
    return out


def term(table: np.ndarray, weights: list[np.ndarray], axes: tuple[int, ...]) -> np.ndarray:
    shape = [table.shape[j] for j in axes]
    total = np.zeros(shape)
    for k in range(len(axes) + 1):
        for sub in combinations(axes, k):
            m = marginal(table, weights, sub)
            # Broadcast m (over `sub`) onto the grid of `axes`.
            expand = [table.shape[j] if j in sub else 1 for j in axes]
            total = total + (-1) ** (len(axes) - k) * np.broadcast_to(m.reshape(expand), shape)
    return total


def weighted_var(values: np.ndarray, weights: list[np.ndarray], axes: tuple[int, ...]) -> float:
    w = np.ones(())
    for j in axes:
        w = np.multiply.outer(w, weights[j])
    return float(np.sum(w * values**2))


def decompose(table: np.ndarray, raw: list[np.ndarray]) -> dict:
    weights = [w / w.sum() for w in raw]
    d = table.ndim
    every = tuple(range(d))
    mean = float(marginal(table, weights, ()))
    variance = weighted_var(table - mean, weights, every)
    terms = []
    for k in range(1, d + 1):
        for axes in combinations(range(d), k):
            values = term(table, weights, axes)
            v = weighted_var(values, weights, axes)
            terms.append(
                {
                    "axes": list(axes),
                    "values": values.ravel(),
                    "marginal": np.asarray(marginal(table, weights, axes)).ravel(),
                    "variance": v,
                    "fraction": v / variance,
                }
            )
    return {
        "table": table,
        "weights": [w for w in raw],
        "mean": mean,
        "variance": variance,
        "terms": terms,
    }


def cases() -> dict:
    rng = np.random.default_rng(20140601)
    table3 = rng.normal(size=(4, 3, 5))
    a, b, c = np.meshgrid(np.arange(4), np.arange(3), np.arange(5), indexing="ij")
    table3 = table3 + 2.0 * a - 1.5 * (b - 1) ** 2 + 0.5 * a * c
    raw3 = [rng.uniform(0.2, 1.0, size=n) for n in table3.shape]
    x = np.linspace(-1, 1, 6)
    y = np.linspace(0, 2, 4)
    table2 = np.sin(3 * x)[:, None] + y[None, :] ** 2 + 0.7 * np.outer(x, y)
    raw2 = [np.ones(6), np.ones(4)]
    return {"three": decompose(table3, raw3), "two": decompose(table2, raw2)}
