"""Golden values for aifn-compute/ot: scipy's wasserstein_distance, exact transport by scipy's linprog and
linear_sum_assignment, log-domain Sinkhorn written directly in numpy, and POT (Python Optimal Transport): exact costs
by `ot.emd2`, converged entropic plans by `ot.sinkhorn` (log domain; POT's −H(P) regulariser and aifn's KL(P ‖ a bᵀ)
give the same plan under fixed marginals) and entropic Gromov–Wasserstein by `ot.gromov`."""

import numpy as np
from scipy import optimize, special, stats


def sinkhorn_log(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float, iters: int) -> dict[str, object]:
    """f then g soft c-transforms, the plan relative to a bᵀ (the same update order as aifn)."""
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    la, lb = np.log(a), np.log(b)
    for _ in range(iters):
        # logsumexp's declared return includes the (value, sign) pair of return_sign=True.
        f = -eps * np.asarray(special.logsumexp((g[None, :] - c) / eps + lb[None, :], axis=1))
        g = -eps * np.asarray(special.logsumexp((f[:, None] - c) / eps + la[:, None], axis=0))
    plan = a[:, None] * b[None, :] * np.exp((f[:, None] + g[None, :] - c) / eps)
    return {"f": f, "g": g, "plan": plan, "transport": float((plan * c).sum())}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    out: dict[str, object] = {}
    u = rng.normal(size=12)
    v = rng.normal(1.0, 2.0, size=9)
    uw = rng.uniform(0.1, 1, size=12)
    vw = rng.uniform(0.1, 1, size=9)
    out["w1"] = {
        "u": u,
        "v": v,
        "uw": uw,
        "vw": vw,
        "plain": stats.wasserstein_distance(u, v),
        "weighted": stats.wasserstein_distance(u, v, uw, vw),
    }
    x = rng.normal(size=(6, 2))
    y = rng.normal(size=(6, 2)) + 1
    c = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    rows, cols = optimize.linear_sum_assignment(c)
    out["assignment"] = {"x": x, "y": y, "cost": c, "cols": cols, "value": float(c[rows, cols].sum() / 6)}
    a = rng.uniform(0.2, 1, size=5)
    a /= a.sum()
    b = rng.uniform(0.2, 1, size=4)
    b /= b.sum()
    c2 = rng.uniform(0, 3, size=(5, 4))
    aeq = np.zeros((9, 20))
    for i in range(5):
        aeq[i, i * 4 : (i + 1) * 4] = 1
    for j in range(4):
        aeq[5 + j, j::4] = 1
    res = optimize.linprog(c2.ravel(), A_eq=aeq, b_eq=np.concatenate([a, b]), method="highs")
    out["exact"] = {"a": a, "b": b, "cost": c2, "value": res.fun}
    out["sinkhorn"] = {"a": a, "b": b, "cost": c2, "eps": 0.1, "iters": 50, **sinkhorn_log(a, b, c2, 0.1, 50)}
    out["pot"] = pot_cases(rng)
    return out


def pot_cases(rng: np.random.Generator) -> dict[str, object]:
    import ot

    cases = []
    for n, m, eps in [(5, 4, 0.5), (7, 9, 0.1), (12, 10, 0.05)]:
        a = rng.uniform(0.2, 1, size=n)
        a /= a.sum()
        b = rng.uniform(0.2, 1, size=m)
        b /= b.sum()
        x = rng.normal(size=(n, 2))
        y = rng.normal(size=(m, 2)) + 0.5
        c = ot.dist(x, y)
        plan = ot.sinkhorn(a, b, c, eps, method="sinkhorn_log", numItermax=200000, stopThr=1e-15)
        cases.append({"a": a, "b": b, "cost": c, "eps": eps, "emd": ot.emd2(a, b, c), "plan": plan})
    # Entropic Gromov-Wasserstein (square loss): POT's linearised cost is twice aifn's, so POT's ε is twice aifn's.
    x = rng.normal(size=(6, 2))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    y = x[[3, 0, 5, 1, 4, 2]] @ rot.T + 0.01 * rng.normal(size=(6, 2))
    cx, cy = ot.dist(x, x), ot.dist(y, y)
    p = np.full(6, 1 / 6)
    gw = ot.gromov.entropic_gromov_wasserstein(
        cx, cy, p, p, "square_loss", epsilon=2 * 0.05, max_iter=1000, tol=1e-12, solver="PGD", verbose=False
    )
    loss = ot.gromov.gromov_wasserstein2(cx, cy, p, p, "square_loss")
    return {"sinkhorn": cases, "gromov": {"cx": cx, "cy": cy, "eps": 0.05, "plan": gw, "exactLoss": loss}}
