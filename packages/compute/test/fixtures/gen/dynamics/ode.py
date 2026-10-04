"""Golden values for aifn-compute/ode from scipy: solve_ivp's RK45 step sequence, expm and general eigenvalues."""

from typing import cast

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp


def lotka_volterra(_t: float, x: np.ndarray) -> np.ndarray:
    return np.array([1.5 * x[0] - x[0] * x[1], -3.0 * x[1] + x[0] * x[1]])


def robertson(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def robertson_jac(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0.0, 6e7 * y[1], 0.0],
        ]
    )


MU = 1000.0


def van_der_pol(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]])


def van_der_pol_jac(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array([[0.0, 1.0], [-2 * MU * y[0] * y[1] - 1, MU * (1 - y[0] ** 2)]])


def bdf_cases() -> dict[str, object]:
    """scipy BDF (analytic Jacobian) step sequences, and tight Radau references at the end time."""
    problems = {
        "robertson": (robertson, robertson_jac, (0.0, 1e5), [1.0, 0.0, 0.0], 1e-6, [1e-8, 1e-14, 1e-8]),
        "van_der_pol": (van_der_pol, van_der_pol_jac, (0.0, 3000.0), [2.0, 0.0], 1e-3, [1e-6, 1e-6]),
        "van_der_pol_tight": (van_der_pol, van_der_pol_jac, (0.0, 3000.0), [2.0, 0.0], 1e-7, [1e-9, 1e-9]),
    }
    out = {}
    for key, (f, jac, span, y0, rtol, atol) in problems.items():
        sol = solve_ivp(f, span, y0, method="BDF", jac=jac, rtol=rtol, atol=atol)
        ref = solve_ivp(f, span, y0, method="Radau", jac=jac, rtol=1e-11, atol=[a * 1e-4 for a in atol])
        out[key] = {
            "span": span,
            "y0": y0,
            "rtol": rtol,
            "atol": atol,
            "t": sol.t,
            "x": sol.y.T,
            "nfev": sol.nfev,
            "njev": sol.njev,
            "nlu": sol.nlu,
            "reference": ref.y[:, -1],
        }
    return out


def neural_ode_case() -> dict[str, object]:
    """torchdiffeq's adjoint (odeint_adjoint, dopri5, tight tolerances) on a tiny tanh field with fixed weights.

    f(x) = tanh(x W1 + b) W2 on a batch x of shape [2, 2], θ = [W1 (2×3), b (3), W2 (3×2)] flattened row-major, and
    L = Σ x(1) ⊙ M. Returns x(1), ∂L/∂θ and ∂L/∂x(0), all in float64.
    """
    import torch
    from torchdiffeq import odeint_adjoint  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    theta = [0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.2, 0.05, 0.7, -0.4, 0.3, 0.9, -0.5, 0.2]
    x0 = [[0.3, -0.7], [1.1, 0.4]]
    weights = [[1.0, -2.0], [0.5, 1.0]]

    class Field(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))

        def forward(self, _t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            w1 = self.theta[0:6].reshape(2, 3)
            b = self.theta[6:9]
            w2 = self.theta[9:15].reshape(3, 2)
            return torch.tanh(x @ w1 + b) @ w2

    field = Field()
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xt = cast(torch.Tensor, odeint_adjoint(field, x, t, method="dopri5", rtol=1e-11, atol=1e-12))[-1]
    loss = (xt * torch.tensor(weights, dtype=torch.float64)).sum()  # pyright: ignore[reportUnknownMemberType]
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    return {
        "theta": theta,
        "x0": x0,
        "weights": weights,
        "x1": xt.detach().numpy(),  # pyright: ignore[reportUnknownMemberType]
        "grad_theta": field.theta.grad.numpy(),  # pyright: ignore[reportOptionalMemberAccess]
        "grad_x0": x.grad.numpy(),  # pyright: ignore[reportOptionalMemberAccess]
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    rk45 = {}
    for rtol, atol in [(1e-3, 1e-6), (1e-8, 1e-10)]:
        sol = solve_ivp(lotka_volterra, (0.0, 10.0), [10.0, 5.0], method="RK45", rtol=rtol, atol=atol)
        rk45[f"{rtol:g}"] = {
            "rtol": rtol,
            "atol": atol,
            "t": sol.t,
            "x": sol.y.T,
            "nfev": sol.nfev,
        }
    reference = solve_ivp(lotka_volterra, (0.0, 10.0), [10.0, 5.0], method="DOP853", rtol=1e-13, atol=1e-13)

    matrices = {
        "rotation": np.array([[0.0, 1.0], [-1.0, 0.0]]),
        "random": rng.normal(size=(4, 4)),
        "large": 8.0 * rng.normal(size=(5, 5)),
        "nilpotent": np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]),
        "stiff": np.array([[-1000.0, 1.0], [0.0, -0.5]]),
    }
    expm = {k: {"a": a, "expm": sla.expm(a)} for k, a in matrices.items()}

    eig = {}
    for k, a in {
        "random": rng.normal(size=(6, 6)),
        "companion": np.array([[0, 1, 0], [0, 0, 1], [6, -11, 6.0]]),
    }.items():
        w = np.linalg.eigvals(a)
        order = np.lexsort((-np.imag(w), -np.real(w)))
        eig[k] = {"a": a, "real": np.real(w)[order], "imag": np.imag(w)[order]}

    return {
        "rk45": rk45,
        "lotka_volterra_x10": reference.y[:, -1],
        "expm": expm,
        "eig": eig,
        "bdf": bdf_cases(),
        "neural_ode": neural_ode_case(),
    }
