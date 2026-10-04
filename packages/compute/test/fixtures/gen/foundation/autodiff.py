"""Golden gradients, Hessians and Hessian-vector products for aifn-compute/autodiff, from torch.autograd in float64 (and
scipy for the polygamma values, where torch is less accurate).

Each case gives its inputs, the function's value and torch's derivatives; the tests compute the same function with
aifn primitives and compare.
"""

from typing import cast

import numpy as np
import torch
from scipy import special as sp

torch.set_default_dtype(torch.float64)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=True)


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().numpy()


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    out: dict[str, object] = {}

    # softplus: f(x) = Σ w·softplus(x), including large |x|.
    x = np.concatenate([rng.normal(size=5) * 3, [-40.0, 40.0]])
    w = rng.uniform(0.5, 1.5, size=x.size)
    xt = t(x)
    y = (torch.tensor(w) * torch.nn.functional.softplus(xt)).sum()
    (g,) = torch.autograd.grad(y, xt)
    out["softplus"] = {"x": x, "w": w, "value": y.item(), "grad": arr(g)}

    # logsumexp along the last axis: f(X) = Σᵢ wᵢ logsumexp(Xᵢ,:), X 3×4.
    xm = rng.normal(size=(3, 4)) * 2
    wl = rng.uniform(0.5, 1.5, size=3)
    xt = t(xm)
    y = (torch.tensor(wl) * torch.logsumexp(xt, dim=1)).sum()
    (g,) = torch.autograd.grad(y, xt)
    hess = torch.autograd.functional.hessian(lambda z: (torch.tensor(wl) * torch.logsumexp(z, dim=1)).sum(), xt)
    out["logsumexp"] = {"x": xm, "w": wl, "value": y.item(), "grad": arr(g), "hessian": arr(hess).reshape(12, 12)}

    # matmul: f(A, B) = Σ W ⊙ tanh(A B), A 3×4, B 4×2.
    a = rng.normal(size=(3, 4))
    b = rng.normal(size=(4, 2))
    wm = rng.normal(size=(3, 2))
    at, bt = t(a), t(b)
    y = (torch.tensor(wm) * torch.tanh(at @ bt)).sum()
    ga, gb = torch.autograd.grad(y, [at, bt])
    out["matmul"] = {"a": a, "b": b, "w": wm, "value": y.item(), "gradA": arr(ga), "gradB": arr(gb)}

    # Cholesky log-determinant of A = C Cᵀ + n I, differentiated with respect to C (so that torch's symmetric and
    # aifn's lower-triangular conventions for ∂/∂A agree).
    n = 4
    c = rng.normal(size=(n, n))
    ct = t(c)
    am = ct @ ct.T + n * torch.eye(n)
    y = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(am))).sum()
    (g,) = torch.autograd.grad(y, ct)
    out["choleskyLogDet"] = {"c": c, "value": y.item(), "grad": arr(g)}

    # Normal log-likelihood of data under N(μ, σ²) with σ = exp(s): gradient and Hessian in (μ, s), and an HVP.
    data = rng.normal(loc=1.0, scale=2.0, size=20)

    def loglik(p: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(p[0], torch.exp(p[1])).log_prob(torch.tensor(data)).sum()

    p = np.array([0.3, 0.2])
    pt = t(p)
    y = loglik(pt)
    (g,) = torch.autograd.grad(y, pt)
    # hessian and hvp are typed for tuples of inputs; one tensor in gives one tensor out.
    h = cast(torch.Tensor, torch.autograd.functional.hessian(loglik, pt))
    v = np.array([0.7, -1.3])
    hv = cast(torch.Tensor, torch.autograd.functional.hvp(loglik, pt, torch.tensor(v))[1])
    out["normalLogLikelihood"] = {
        "data": data,
        "params": p,
        "value": y.item(),
        "grad": arr(g),
        "hessian": arr(h),
        "v": v,
        "hvp": arr(hv),
    }

    # Derivatives of log Γ: ψ, ψ₁ (trigamma) and ψ₂.
    xs = np.array([0.1, 0.5, 1.0, 2.5, 7.0, 30.0])
    # torch.polygamma(1, x) is off by up to 3e-10 relative, so scipy is the reference for these.
    out["logGamma"] = {
        "x": xs,
        "d1": sp.digamma(xs),
        "d2": sp.polygamma(1, xs),
        "d3": sp.polygamma(2, xs),
    }

    # A mixed pytree: f(p) = Σ softplus(W x + b) · u + ‖W‖² / 2, p = {W, b, u}, with x fixed.
    wp = rng.normal(size=(3, 2))
    bp = rng.normal(size=3)
    up = rng.normal(size=3)
    xp = rng.normal(size=2)
    wt, bt2, ut = t(wp), t(bp), t(up)
    y = (torch.nn.functional.softplus(wt @ torch.tensor(xp) + bt2) * ut).sum() + 0.5 * (wt**2).sum()
    gw, gb2, gu = torch.autograd.grad(y, [wt, bt2, ut])
    out["pytree"] = {
        "W": wp,
        "b": bp,
        "u": up,
        "x": xp,
        "value": y.item(),
        "gradW": arr(gw),
        "gradB": arr(gb2),
        "gradU": arr(gu),
    }
    return out
