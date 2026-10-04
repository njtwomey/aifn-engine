"""Golden values for aifn-compute/nn/sequence, from scipy, numpy and torch in float64.

- Discretisation: `scipy.signal.cont2discrete` (zero-order hold and bilinear) of HiPPO-LegS and a singular system.
- HiPPO-LegS: a transcription of S4's `transition('legs', N)`.
- S4 kernel and recurrence: numpy powers Ā^k and a sequential loop; outputs by `scipy.signal.dlsim`.
- Selective scan: a transcription of Mamba's `selective_scan_ref` (torch, batch first, channels before time), with
  gradients.
- Linear attention: the causal cumulative-sum form of Katharopoulos et al. (2020), with gradients.
- Bahdanau and Luong attention: torch transcriptions, with gradients.
- Backpropagation through time: torch's GRUCell unrolled, with ∂L/∂h_t retained at every step.

Parameters are in aifn's layout: Linear weights [in, out]; recurrent weights [in, gates * H].
"""

from typing import cast

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def t(a, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float64, requires_grad=grad)


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().numpy()


def grad_of(p: torch.Tensor) -> torch.Tensor:
    assert p.grad is not None, "backward() has not reached this tensor"
    return p.grad


def legs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """S4's `transition('legs', N)`."""
    q = np.arange(n, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    return A, B


def selective_scan_ref(u, delta, A, B, C, D, z):
    """Mamba's `selective_scan_ref` with variable B and C, delta_softplus=False."""
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = u.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    ys = []
    for i in range(u.shape[2]):
        x = delta_a[:, :, i] * x + delta_b_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y + u * D[..., None]
    return out * F.silu(z)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(7)
    out: dict[str, object] = {}

    # Discretisation.
    A, B = legs(4)
    C = rng.normal(size=(1, 4))
    disc = {"A": A, "B": B, "C": C, "dt": 0.1}
    for method, name in [("zoh", "zoh"), ("bilinear", "bilinear")]:
        Ad, Bd, *_ = cast(
            tuple[np.ndarray, ...], scipy.signal.cont2discrete((A, B, C, np.zeros((1, 1))), 0.1, method=method)
        )
        disc[name] = {"A": Ad, "B": Bd}
    S = np.array([[0.0, 1.0], [0.0, -0.5]])  # singular: ZOH must not invert A
    Sb = np.array([[0.0], [1.0]])
    Ad, Bd, *_ = cast(
        tuple[np.ndarray, ...], scipy.signal.cont2discrete((S, Sb, np.eye(2), np.zeros((2, 1))), 0.3, method="zoh")
    )
    disc["singular"] = {"A": S, "B": Sb, "dt": 0.3, "Ad": Ad, "Bd": Bd}
    out["discretise"] = disc
    out["hippo"] = {str(n): {"A": legs(n)[0], "B": legs(n)[1]} for n in [1, 3, 6]}

    # S4 kernel and the two modes.
    Ad, Bd = disc["zoh"]["A"], disc["zoh"]["B"]
    L = 12
    K = np.array([(C @ np.linalg.matrix_power(Ad, k) @ Bd).item() for k in range(L)])
    u = rng.normal(size=L)
    # A state-space system makes dlsim return (t, y, x); its declared return also allows (t, y).
    _, y, x = cast(tuple[np.ndarray, np.ndarray, np.ndarray], scipy.signal.dlsim((Ad, Bd, C, np.zeros((1, 1)), 1), u))
    # dlsim's state x_k is before input u_k; aifn's state after it (x_k = Ā x_{k−1} + B̄ u_k, y_k = C x_k).
    states = x[1:] if len(x) > L else np.vstack([x[1:], (Ad @ x[-1] + Bd[:, 0] * u[-1])[None]])
    out["s4"] = {"kernel": K, "u": u, "y": np.convolve(K, u)[:L], "states": states}

    # Selective scan.
    b, d, n, L = 2, 3, 4, 7
    u = rng.normal(size=(b, d, L))
    delta = np.log1p(np.exp(rng.normal(size=(b, d, L)) - 1))
    A = -np.exp(rng.normal(size=(d, n)) * 0.5)
    Bv = rng.normal(size=(b, n, L))
    Cv = rng.normal(size=(b, n, L))
    D = rng.normal(size=d)
    z = rng.normal(size=(b, d, L))
    ins = [t(a, True) for a in (u, delta, A, Bv, Cv, D, z)]
    y = selective_scan_ref(*ins)
    r = rng.normal(size=y.shape)
    grads = torch.autograd.grad((t(r) * y).sum(), ins)
    # Time first for aifn: x [b, L, d], B and C [b, L, n].
    tf = lambda a: np.swapaxes(a, 1, 2)  # noqa: E731
    out["selective"] = {
        "x": tf(u),
        "delta": tf(delta),
        "A": A,
        "B": tf(Bv),
        "C": tf(Cv),
        "D": D,
        "z": tf(z),
        "y": tf(arr(y)),
        "r": tf(r),
        "gx": tf(arr(grads[0])),
        "gdelta": tf(arr(grads[1])),
        "gA": arr(grads[2]),
        "gB": tf(arr(grads[3])),
        "gC": tf(arr(grads[4])),
    }

    # Causal linear attention (elu + 1 features), cumulative-sum form.
    T, dk, dv = 6, 3, 2
    q, k, v = rng.normal(size=(T, dk)), rng.normal(size=(T, dk)), rng.normal(size=(T, dv))
    qt, kt, vt = t(q, True), t(k, True), t(v, True)
    fq, fk = F.elu(qt) + 1, F.elu(kt) + 1
    S = torch.cumsum(fk[:, :, None] * vt[:, None, :], 0)
    Z = torch.cumsum(fk, 0)
    o = torch.einsum("td,tde->te", fq, S) / (fq * Z).sum(-1, keepdim=True)
    r = rng.normal(size=o.shape)
    gq, gk, gv = torch.autograd.grad((t(r) * o).sum(), (qt, kt, vt))
    out["linearAttention"] = {"q": q, "k": k, "v": v, "y": arr(o), "r": r, "gq": arr(gq), "gk": arr(gk), "gv": arr(gv)}

    # Bahdanau and Luong attention over a batch of encoder states.
    Bt, Ts, dq, dkk, hid = 2, 5, 3, 4, 6
    s = rng.normal(size=(Bt, dq))
    h = rng.normal(size=(Bt, Ts, dkk))
    mask = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=float)
    p = {
        "query": {"weight": rng.normal(size=(dq, hid)) / 2},
        "key": {"weight": rng.normal(size=(dkk, hid)) / 2, "bias": 0.1 * rng.normal(size=hid)},
        "v": rng.normal(size=(hid, 1)),
    }
    st, ht = t(s, True), t(h, True)
    e = (
        torch.tanh((st @ t(p["query"]["weight"]))[:, None, :] + ht @ t(p["key"]["weight"]) + t(p["key"]["bias"]))
        @ t(p["v"])
    )[..., 0]
    e = e.masked_fill(t(mask) == 0, float("-inf"))
    a = torch.softmax(e, -1)
    c = (a[:, :, None] * ht).sum(1)
    r = rng.normal(size=c.shape)
    gs, gh = torch.autograd.grad((t(r) * c).sum(), (st, ht))
    out["bahdanau"] = {
        "params": p,
        "s": s,
        "h": h,
        "mask": mask,
        "context": arr(c),
        "weights": arr(a),
        "r": r,
        "gs": arr(gs),
        "gh": arr(gh),
    }

    luong = {}
    for score in ["dot", "general", "concat"]:
        q_dim = dkk if score == "dot" else dq
        s = rng.normal(size=(Bt, q_dim))
        st, ht = t(s, True), t(h, True)
        if score == "dot":
            lp = {}
            e = (ht @ st[:, :, None])[..., 0]
        elif score == "general":
            lp = {"weight": rng.normal(size=(dkk, q_dim)) / 2}
            e = ((ht @ t(lp["weight"])) @ st[:, :, None])[..., 0]
        else:
            lp = {"concat": {"weight": rng.normal(size=(q_dim + dkk, hid)) / 2}, "v": rng.normal(size=(hid, 1))}
            cat = torch.cat([st[:, None, :].expand(Bt, Ts, q_dim), ht], -1)
            e = (torch.tanh(cat @ t(lp["concat"]["weight"])) @ t(lp["v"]))[..., 0]
        a = torch.softmax(e, -1)
        c = (a[:, :, None] * ht).sum(1)
        r = rng.normal(size=c.shape)
        gs, gh = torch.autograd.grad((t(r) * c).sum(), (st, ht))
        luong[score] = {
            "params": lp,
            "s": s,
            "h": h,
            "context": arr(c),
            "weights": arr(a),
            "r": r,
            "gs": arr(gs),
            "gh": arr(gh),
        }
    out["luong"] = luong

    # Backpropagation through time: a GRU unrolled over 12 steps, loss on the last hidden state.
    torch.manual_seed(7)
    n_in, H, T = 2, 3, 12
    gru = torch.nn.GRUCell(n_in, H)
    xs = rng.normal(size=(T, n_in))
    w = rng.normal(size=H)
    hstate = torch.zeros(H)
    hs = []
    for step in range(T):
        hstate = gru(t(xs[step]), hstate)
        hstate.retain_grad()
        hs.append(hstate)
    loss = (t(w) * hs[-1]).sum() + 0.1 * sum((hh**2).sum() for hh in hs)
    loss.backward()
    out["bptt"] = {
        "x": xs,
        "w": w,
        "inputWeight": arr(gru.weight_ih).T,
        "hiddenWeight": arr(gru.weight_hh).T,
        "bias": arr(gru.bias_ih),
        "hiddenBias": arr(gru.bias_hh),
        "loss": loss.item(),
        "hiddenGrads": np.stack([hh.grad.numpy() for hh in hs]),
        "gInputWeight": arr(grad_of(gru.weight_ih)).T,
    }
    return out
