"""Golden values for aifn-compute/tensor: broadcasting, reductions, matmul, einsum and views, from numpy and scipy."""

import numpy as np
from scipy.special import logsumexp


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    a = rng.normal(size=(3, 1, 4))
    b = rng.normal(size=(2, 1))
    positive = rng.uniform(0.5, 2.0, size=(3, 1, 4))
    broadcasting = {
        "a": a,
        "b": b,
        "positive": positive,
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "div": a / b,
        "pow": positive**b,
        "maximum": np.maximum(a, b),
        "where": np.where(a > 0, a, b),
        "rowPlusColumn": np.arange(3.0)[:, None] + np.arange(4.0)[None, :],
    }

    x = rng.normal(size=(3, 4, 5))
    axes: list[tuple[str, int | tuple[int, ...] | None]] = [
        ("all", None),
        ("0", 0),
        ("1", 1),
        ("-1", -1),
        ("0,2", (0, 2)),
    ]
    reductions: dict[str, object] = {"x": x}
    for name, axis in axes:
        ax = axis
        reductions[name] = {
            "sum": np.sum(x, axis=ax),
            "mean": np.mean(x, axis=ax),
            "max": np.max(x, axis=ax),
            "min": np.min(x, axis=ax),
            "prod": np.prod(x, axis=ax),
            "variance": np.var(x, axis=ax),
            "varianceDdof1": np.var(x, axis=ax, ddof=1),
            "std": np.std(x, axis=ax),
            "logsumexp": logsumexp(x, axis=ax),
            "sumKeep": np.sum(x, axis=ax, keepdims=True),
        }
    reductions["argmax1"] = np.argmax(x, axis=1)
    reductions["argmin2"] = np.argmin(x, axis=2)
    reductions["argmaxAll"] = int(np.argmax(x))
    reductions["norm1Axis1"] = np.linalg.norm(x, ord=1, axis=1)
    reductions["norm2Axis1"] = np.linalg.norm(x, ord=2, axis=1)
    reductions["normInfAxis1"] = np.linalg.norm(x, ord=np.inf, axis=1)
    reductions["norm3Axis1"] = np.sum(np.abs(x) ** 3, axis=1) ** (1 / 3)
    reductions["normAll"] = float(np.sqrt(np.sum(x**2)))

    m1 = rng.normal(size=(2, 3))
    m2 = rng.normal(size=(3, 4))
    ba = rng.normal(size=(2, 1, 3, 4))
    bb = rng.normal(size=(5, 4, 2))
    v3 = rng.normal(size=3)
    v4 = rng.normal(size=4)
    s3 = rng.normal(size=(4, 2, 3))
    matmul = {
        "m1": m1,
        "m2": m2,
        "m1m2": m1 @ m2,
        "ba": ba,
        "bb": bb,
        "babb": ba @ bb,
        "v3": v3,
        "v4": v4,
        "v3m2": v3 @ m2,
        "m2v4": m2 @ v4,
        "v3v3": float(v3 @ v3),
        "s3": s3,
        "s3v3": s3 @ v3,
        "outer": np.outer(v3, v4),
    }

    e1 = rng.normal(size=(3, 3))
    e2 = rng.normal(size=(3, 4))
    e3 = rng.normal(size=(2, 3, 4))
    e4 = rng.normal(size=(2, 4, 5))
    e5 = rng.normal(size=(4, 5))
    einsum = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
        "v3": v3,
        "cases": [
            {"spec": "ij,jk->ik", "operands": ["e1", "e2"], "result": np.einsum("ij,jk->ik", e1, e2)},
            {"spec": "ij->ji", "operands": ["e2"], "result": np.einsum("ij->ji", e2)},
            {"spec": "i,i->", "operands": ["v3", "v3"], "result": np.einsum("i,i->", v3, v3)},
            {"spec": "ii->", "operands": ["e1"], "result": np.einsum("ii->", e1)},
            {"spec": "ii->i", "operands": ["e1"], "result": np.einsum("ii->i", e1)},
            {"spec": "bij,bjk->bik", "operands": ["e3", "e4"], "result": np.einsum("bij,bjk->bik", e3, e4)},
            {"spec": "ij,ij->i", "operands": ["e2", "e2"], "result": np.einsum("ij,ij->i", e2, e2)},
            {"spec": "ijk->kj", "operands": ["e3"], "result": np.einsum("ijk->kj", e3)},
            {"spec": "ij,jk", "operands": ["e1", "e2"], "result": np.einsum("ij,jk", e1, e2)},
            {"spec": "i,j->ij", "operands": ["v3", "v3"], "result": np.einsum("i,j->ij", v3, v3)},
            {"spec": "ij,jk,kl->il", "operands": ["e1", "e2", "e5"], "result": np.einsum("ij,jk,kl->il", e1, e2, e5)},
            {"spec": "bij->", "operands": ["e3"], "result": np.einsum("bij->", e3)},
        ],
    }

    grid = np.arange(60.0).reshape(3, 4, 5)
    t = grid.transpose(2, 0, 1)
    views = {
        "grid": grid,
        "row1": grid[1],
        "evenColumns": grid[:, ::2],
        "mixed": grid[::-1, 1:3, -1],
        "reversedLast": grid[:, :, ::-2],
        "negativeRange": grid[-2:, -3:-1, 1:4:2],
        "transposed": t,
        "transposedSlice": t[1:4, :, ::3],
        "transposedReshape": t.reshape(5, 12),
        "empty": grid[2:1],
        "concat1": np.concatenate([grid, grid[:, :2]], axis=1),
        "stack2": np.stack([grid[0], grid[1]], axis=2),
    }
    return {
        "broadcasting": broadcasting,
        "reductions": reductions,
        "matmul": matmul,
        "einsum": einsum,
        "views": views,
        "linspace": np.linspace(-1.0, 2.0, 7),
        "linspaceOpen": np.linspace(-1.0, 2.0, 7, endpoint=False),
        "arange": np.arange(2.0, 3.0, 0.25),
    }
