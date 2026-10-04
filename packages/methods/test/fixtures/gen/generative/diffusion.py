"""Golden noise schedules for aifn-compute/diffusion, transcribed from the papers' reference code in numpy.

- Linear schedule and posterior coefficients: Ho et al. (2020), diffusion_utils_2.py (betas linspace 1e-4 → 0.02,
  alphas_cumprod, posterior_variance, posterior_mean_coef1/2).
- Cosine schedule: Nichol & Dhariwal (2021), improved-diffusion's betas_for_alpha_bar with max_beta 0.999.
- VP SDE marginal: m(t) = exp(-1/2 ∫ β) by quadrature (Song et al., 2021).
- Sub-VP SDE sampling in torch, transcribed from score_sde's `subVPSDE` (sde, marginal_prob) and its reverse-time
  Euler–Maruyama and probability-flow updates (Song et al., 2021, sde_lib.py and sampling.py), driven by the exact score
  of a two-component Gaussian mixture (each noised marginal is a mixture of N(m μₖ, (m²σₖ² + s²)I), the score by
  autograd of its log density): coefficients at several times, Euler and RK4 probability-flow paths from fixed starts,
  and the deterministic part x − [f x − g²∇log p_t]h of reverse-SDE steps with the noise scale g√h.
"""

import math

import numpy as np
import torch
from scipy import integrate

BETA_0, BETA_1 = 0.1, 20.0
MIXTURE = {"weights": [0.3, 0.7], "means": [[-1.0, 0.5], [1.2, -0.3]], "stds": [0.4, 0.6]}


def sub_vp_sde(t: float) -> tuple[float, float]:
    """score_sde subVPSDE.sde: the drift coefficient −β(t)/2 and the diffusion √(β(t)(1 − e^{−2∫β}))."""
    tt = torch.tensor(t, dtype=torch.float64)
    beta_t = BETA_0 + tt * (BETA_1 - BETA_0)
    discount = 1.0 - torch.exp(-2 * BETA_0 * tt - (BETA_1 - BETA_0) * tt**2)
    return float(-0.5 * beta_t), float(torch.sqrt(beta_t * discount))


def sub_vp_marginal(t: float) -> tuple[float, float]:
    """score_sde subVPSDE.marginal_prob: the mean scale and the std 1 − exp(2 log m)."""
    log_mean_coeff = -0.25 * t**2 * (BETA_1 - BETA_0) - 0.5 * t * BETA_0
    return math.exp(log_mean_coeff), 1.0 - math.exp(2.0 * log_mean_coeff)


def score(x: torch.Tensor, t: float) -> torch.Tensor:
    m, s = sub_vp_marginal(t)
    x = x.detach().requires_grad_(True)
    w = torch.tensor(MIXTURE["weights"], dtype=torch.float64)
    mu = torch.tensor(MIXTURE["means"], dtype=torch.float64) * m
    var = torch.tensor(MIXTURE["stds"], dtype=torch.float64) ** 2 * m * m + s * s
    d = x.shape[1]
    sq = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    logp = torch.logsumexp(torch.log(w) - 0.5 * sq / var - 0.5 * d * torch.log(2 * math.pi * var), dim=1)
    (g,) = torch.autograd.grad(logp.sum(), x)
    return g


def flow(x: torch.Tensor, t: float) -> torch.Tensor:
    f, g = sub_vp_sde(t)
    return f * x - 0.5 * g * g * score(x, t)


def probability_flow(x0: np.ndarray, steps: int, method: str, end: float = 1e-3) -> np.ndarray:
    h = (1 - end) / steps
    x = torch.tensor(x0, dtype=torch.float64)
    t = 1.0
    for k in range(steps):
        if method == "euler":
            x = x - h * flow(x, t)
        else:
            k1 = flow(x, t)
            k2 = flow(x - 0.5 * h * k1, t - 0.5 * h)
            k3 = flow(x - 0.5 * h * k2, t - 0.5 * h)
            k4 = flow(x - h * k3, t - h)
            x = x - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = end if k + 1 == steps else 1 - (k + 1) * h
    return x.detach().numpy()


def sub_vp_cases() -> dict[str, object]:
    times = [1e-3, 0.05, 0.3, 0.7, 1.0]
    coefficients = [(*sub_vp_sde(t), *sub_vp_marginal(t)) for t in times]
    x0 = np.array([[0.3, -1.1], [-0.8, 0.2], [1.5, 0.9], [0.0, 0.0]])
    steps, end = 40, 1e-3
    h = (1 - end) / steps
    reverse = []
    for t in [1.0, 0.5, 0.1]:
        f, g = sub_vp_sde(t)
        x = torch.tensor(x0, dtype=torch.float64)
        mean = x - (f * x - g * g * score(x, t)) * h
        reverse.append({"time": t, "mean": mean.detach().numpy(), "noiseScale": g * math.sqrt(h)})
    return {
        "mixture": MIXTURE,
        "times": times,
        "drift": [c[0] for c in coefficients],
        "diffusion": [c[1] for c in coefficients],
        "meanScale": [c[2] for c in coefficients],
        "std": [c[3] for c in coefficients],
        "x0": x0,
        "steps": steps,
        "end": end,
        "euler": probability_flow(x0, steps, "euler", end),
        "rk4": probability_flow(x0, 20, "rk4", end),
        "reverse": reverse,
    }


def cases() -> dict[str, object]:
    T = 1000
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
    alphas_cumprod = np.cumprod(1.0 - betas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    cos_betas = np.array([min(1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), 0.999) for i in range(T)])

    beta_min, beta_max = 0.1, 20.0
    times = [0.0, 0.1, 0.37, 0.8, 1.0]
    vp_mean = [math.exp(-0.5 * integrate.quad(lambda s: beta_min + s * (beta_max - beta_min), 0, t)[0]) for t in times]
    pick = [0, 1, 9, 99, 499, 998, 999]
    return {
        "linear": {
            "betas": betas[pick],
            "alphaBars": alphas_cumprod[pick],
            "posteriorVariance": posterior_variance[pick],
            "coef1": coef1[pick],
            "coef2": coef2[pick],
            "steps": [p + 1 for p in pick],
        },
        "cosine": {
            "betas": cos_betas[pick],
            "alphaBars": np.cumprod(1 - cos_betas)[pick],
            "steps": [p + 1 for p in pick],
        },
        "vp": {"times": times, "meanScale": vp_mean},
        "subVp": sub_vp_cases(),
    }
