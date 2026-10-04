"""Golden values for aifn-compute/optim/first-order from torch.optim: the pytree update rules on a quadratic.

Each case runs a torch optimiser for six steps on f(w, b) = ½ wᵀAw − cᵀw + ½(b − 1)², with parameters a pytree
{w: [3], b: scalar}, and records the parameters after every step.
"""

import numpy as np
import torch

A = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.0]])
C = np.array([1.0, -2.0, 0.5])
W0 = [0.5, -1.0, 2.0]
B0 = 3.0
STEPS = 6


def run(make: object) -> dict[str, list[list[float]]]:
    w = torch.tensor(W0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(B0, dtype=torch.float64, requires_grad=True)
    opt = make([w, b])  # type: ignore[operator]
    a = torch.tensor(A)
    c = torch.tensor(C)
    ws: list[list[float]] = []
    bs: list[float] = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = 0.5 * w @ a @ w - c @ w + 0.5 * (b - 1) ** 2
        loss.backward()
        opt.step()
        ws.append(w.detach().tolist())
        bs.append(float(b.detach()))
    return {"w": ws, "b": [[v] for v in bs]}


def cases() -> dict[str, object]:
    return {
        "A": A,
        "c": C,
        "w0": W0,
        "b0": B0,
        "sgd": run(lambda p: torch.optim.SGD(p, lr=0.1)),
        "momentum": run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.05)),
        "nesterov": run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.8, nesterov=True)),
        "adagrad": run(lambda p: torch.optim.Adagrad(p, lr=0.3, eps=1e-10)),
        "rmsprop": run(lambda p: torch.optim.RMSprop(p, lr=0.05, alpha=0.9, eps=1e-8)),
        "adam": run(lambda p: torch.optim.Adam(p, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)),
        "adamw": run(lambda p: torch.optim.AdamW(p, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)),
    }
