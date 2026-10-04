"""Golden values for aifn-compute/nn/graph from plain torch in float64: graph convolution (three normalisations), GAT
and GATv2
(multi-head, concatenated and averaged), GraphSAGE (mean, sum, max, max-pooling, normalised) and a message-passing
layer, written from the papers' formulas on dense adjacency matrices. Parameters are in aifn's layout (weights
[in, out], applied as x W + b). Gradients are of sum(r * y) with respect to the features and the main weight."""

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

V, FEAT = 8, 3
# An undirected graph on 8 nodes with node 7 isolated; weights for the message-passing layer.
EDGES = [(0, 1, 0.5), (0, 2, 1.0), (1, 2, 2.0), (2, 3, 1.5), (3, 4, 0.7), (4, 5, 1.2), (5, 6, 0.9), (1, 4, 0.3)]


def t(a: np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def adjacency(weighted: bool) -> torch.Tensor:
    a = torch.zeros(V, V)
    for u, v, w in EDGES:
        a[u, v] = a[v, u] = w if weighted else 1.0
    return a


def grads(y: torch.Tensor, r: np.ndarray, *xs: torch.Tensor) -> list[np.ndarray]:
    g = torch.autograd.grad((y * t(r)).sum(), xs)
    return [x.detach().numpy() for x in g]


def gcn(rng: np.random.Generator, h: np.ndarray) -> dict[str, object]:
    w = rng.normal(size=(FEAT, 4)) / 2
    b = 0.1 * rng.normal(size=4)
    out: dict[str, object] = {"weight": w, "bias": b}
    a = adjacency(False) + torch.eye(V)
    d = a.sum(1)
    for norm in ("symmetric", "random-walk", "none"):
        hh, ww = t(h, True), t(w, True)
        if norm == "symmetric":
            ahat = a / torch.sqrt(d[:, None] * d[None, :])
        elif norm == "random-walk":
            ahat = a / d[:, None]
        else:
            ahat = a
        y = ahat @ (hh @ ww) + t(b)
        r = rng.normal(size=y.shape)
        gh, gw = grads(y, r, hh, ww)
        out[norm] = {"y": y.detach().numpy(), "r": r, "gradH": gh, "gradW": gw}
    return out


def gat(rng: np.random.Generator, h: np.ndarray, heads: int, width: int, concat: bool, v2: bool) -> dict[str, object]:
    w = rng.normal(size=(FEAT, heads * width)) / 2
    a_src = rng.normal(size=(heads, width))
    a_dst = rng.normal(size=(heads, width))
    b = 0.1 * rng.normal(size=heads * width if concat else width)
    hh, ww = t(h, True), t(w, True)
    z = (hh @ ww).reshape(V, heads, width)
    mask = (adjacency(False) + torch.eye(V)) > 0  # mask[v, u]: u sends to v (itself included)
    if v2:
        # e[v, u, k] = a_k . LeakyReLU(z_u + z_v)
        s = F.leaky_relu(z[None, :, :, :] + z[:, None, :, :], 0.2)
        e = (s * t(a_src)[None, None]).sum(-1)
    else:
        es = (z * t(a_src)[None]).sum(-1)  # [V, H] source part
        ed = (z * t(a_dst)[None]).sum(-1)  # [V, H] destination part
        e = F.leaky_relu(ed[:, None, :] + es[None, :, :], 0.2)  # [v, u, H]
    e = e.masked_fill(~mask[:, :, None], float("-inf"))
    alpha = torch.softmax(e, dim=1)  # over sources u
    y = torch.einsum("vuk,ukg->vkg", alpha, z)
    y = y.reshape(V, heads * width) if concat else y.mean(1)
    y = y + t(b)
    r = rng.normal(size=y.shape)
    gh, gw = grads(y, r, hh, ww)
    # Attention per message edge in aifn's order: both directions of each edge, then the self-loops.
    order = [(u, v) for u, v, _ in EDGES for (u, v) in ((u, v), (v, u))] + [(i, i) for i in range(V)]
    att = np.array([[alpha[v, u, k].item() for k in range(heads)] for u, v in order])
    return {
        "weight": w,
        "attSource": a_src,
        "attTarget": a_dst,
        "bias": b,
        "y": y.detach().numpy(),
        "r": r,
        "gradH": gh,
        "gradW": gw,
        "attention": att,
    }


def sage(rng: np.random.Generator, h: np.ndarray, aggregate: str, normalise: bool) -> dict[str, object]:
    ws = rng.normal(size=(FEAT, 4)) / 2
    bs = 0.1 * rng.normal(size=4)
    wn = rng.normal(size=(FEAT, 4)) / 2
    wp = rng.normal(size=(FEAT, FEAT)) / 2
    bp = 0.1 * rng.normal(size=FEAT)
    hh, wwn = t(h, True), t(wn, True)
    a = adjacency(False)
    msgs = torch.relu(hh @ t(wp) + t(bp)) if aggregate == "pool" else hh
    rows = []
    for v in range(V):
        nb = [u for u in range(V) if a[v, u] > 0]
        if not nb:
            rows.append(torch.zeros(msgs.shape[1]))
        elif aggregate == "mean":
            rows.append(msgs[nb].mean(0))
        elif aggregate == "sum":
            rows.append(msgs[nb].sum(0))
        else:
            rows.append(msgs[nb].max(0).values)
    agg = torch.stack(rows)
    y = hh @ t(ws) + t(bs) + agg @ wwn
    if normalise:
        y = y / torch.clamp(y.norm(dim=1, keepdim=True), min=1e-12)
    r = rng.normal(size=y.shape)
    gh, gw = grads(y, r, hh, wwn)
    out: dict[str, object] = {"self": {"weight": ws, "bias": bs}, "neighbour": {"weight": wn}}
    if aggregate == "pool":
        out["pool"] = {"weight": wp, "bias": bp}
    return {"params": out, "y": y.detach().numpy(), "r": r, "gradH": gh, "gradW": gw}


def mpnn(rng: np.random.Generator, h: np.ndarray) -> dict[str, object]:
    m_dim, g_dim = 5, 4
    wm = rng.normal(size=(2 * FEAT + 1, m_dim)) / 2
    bm = 0.1 * rng.normal(size=m_dim)
    wu = rng.normal(size=(FEAT + m_dim, g_dim)) / 2
    bu = 0.1 * rng.normal(size=g_dim)
    hh, wwm = t(h, True), t(wm, True)
    a = adjacency(True)
    rows = []
    for v in range(V):
        acc = torch.zeros(m_dim)
        for u in range(V):
            if a[v, u] > 0:
                x = torch.cat([hh[u], hh[v], a[v, u].reshape(1)])
                acc = acc + torch.tanh(x @ wwm + t(bm))
        rows.append(acc)
    agg = torch.stack(rows)
    y = torch.cat([hh, agg], 1) @ t(wu) + t(bu)
    r = rng.normal(size=y.shape)
    gh, gw = grads(y, r, hh, wwm)
    return {
        "message": {"weight": wm, "bias": bm},
        "update": {"weight": wu, "bias": bu},
        "y": y.detach().numpy(),
        "r": r,
        "gradH": gh,
        "gradW": gw,
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261002)
    h = rng.normal(size=(V, FEAT))
    return {
        "nodes": V,
        "edges": [[u, v, w] for u, v, w in EDGES],
        "h": h,
        "gcn": gcn(rng, h),
        "gat": gat(rng, h, 3, 2, True, False),
        "gatMean": gat(rng, h, 2, 3, False, False),
        "gatv2": gat(rng, h, 2, 2, True, True),
        "sage": {
            "mean": sage(rng, h, "mean", False),
            "sum": sage(rng, h, "sum", False),
            "max": sage(rng, h, "max", False),
            "pool": sage(rng, h, "pool", False),
            "meanNormalised": sage(rng, h, "mean", True),
        },
        "mpnn": mpnn(rng, h),
    }
