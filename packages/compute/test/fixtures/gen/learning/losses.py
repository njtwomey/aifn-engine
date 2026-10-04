"""Golden values and gradients for aifn-compute/losses, from torch in float64.

Library losses come from torch.nn.functional; the rest (ranking losses, the multiclass hinges, sampled softmax, NCE,
InfoNCE, log-cosh, pinball) are written here as plain loops over torch scalars, a direct transcription of their
definitions, and differentiated by torch.autograd. Every case stores the value and the gradient with respect to the
predictions.
"""

import math
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=True)


def value_and_grad(f, x: np.ndarray) -> dict[str, object]:
    xt = t(x)
    y = f(xt)
    (g,) = torch.autograd.grad(y, xt)
    return {"value": y.item(), "grad": g.numpy()}


def gain(r: float) -> float:
    return 2.0**r - 1


def discount(pos: int) -> float:
    return 1 / math.log2(pos + 2)


def positions(s: list[float]) -> list[int]:
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    pos = [0] * len(s)
    for r, i in enumerate(order):
        pos[i] = r
    return pos


def ranking(s: torch.Tensor, rel: list[float], kind: str) -> torch.Tensor:
    n = len(rel)
    pairs = [(i, j) for i in range(n) for j in range(n) if rel[i] > rel[j]]
    loss = torch.zeros(())
    if kind == "rankNet":
        for i, j in pairs:
            loss = loss + F.softplus(-(s[i] - s[j]))
    elif kind == "pairwiseHinge":
        for i, j in pairs:
            loss = loss + torch.clamp(1 - (s[i] - s[j]), min=0)
    elif kind == "lambdaRank":
        pos = positions(s.detach().tolist())
        ideal = sum(gain(r) * discount(k) for k, r in enumerate(sorted(rel, reverse=True)))
        for i, j in pairs:
            delta = abs((gain(rel[i]) - gain(rel[j])) * (discount(pos[i]) - discount(pos[j]))) / ideal
            loss = loss + delta * F.softplus(-(s[i] - s[j]))
    elif kind == "listwiseSoftmax":
        total = sum(rel)
        target = torch.tensor([r / total for r in rel])
        loss = -(target * F.log_softmax(s, 0)).sum()
    elif kind == "listNet":
        loss = -(F.softmax(torch.tensor(rel), 0) * F.log_softmax(s, 0)).sum()
    elif kind == "listMle":
        order = sorted(range(n), key=lambda i: (-rel[i], i))
        for k in range(n):
            rest = torch.stack([s[i] for i in order[k:]])
            loss = loss + torch.logsumexp(rest, 0) - s[order[k]]
    elif kind == "pointwiseBce":
        y = torch.tensor([1.0 if r > 0 else 0.0 for r in rel])
        loss = (F.softplus(s) - y * s).sum()
    elif kind == "pointwiseSquaredError":
        loss = ((s - torch.tensor(rel)) ** 2).sum()
    elif kind == "approxNdcg":
        ideal = sum(gain(r) * discount(k) for k, r in enumerate(sorted(rel, reverse=True)))
        dcg = torch.zeros(())
        for i in range(n):
            # n ≥ 2, so the sum has a term and is a tensor.
            rank = cast(torch.Tensor, 1 + sum(torch.sigmoid(s[j] - s[i]) for j in range(n) if j != i))
            dcg = dcg + gain(rel[i]) / torch.log2(1 + rank)
        loss = -dcg / ideal
    return loss


def cases() -> dict[str, object]:
    rng = np.random.default_rng(5)
    out: dict[str, object] = {}

    # Binary cross-entropy from logits, with and without a positive weight, including large logits.
    z = np.concatenate([rng.normal(size=6) * 3, [40.0, -40.0]])
    y = np.array([1, 0, 1, 1, 0, 0, 0, 1.0])
    out["bceLogits"] = {
        "z": z,
        "y": y,
        **value_and_grad(lambda x: F.binary_cross_entropy_with_logits(x, torch.tensor(y)), z),
    }
    out["bceLogitsWeighted"] = {
        "z": z,
        "y": y,
        **value_and_grad(
            lambda x: F.binary_cross_entropy_with_logits(x, torch.tensor(y), pos_weight=torch.tensor(3.0)), z
        ),
    }
    p = rng.uniform(0.05, 0.95, size=6)
    yp = np.array([1, 0, 0.3, 1, 0, 0.8])
    out["bceProbs"] = {"p": p, "y": yp, **value_and_grad(lambda x: F.binary_cross_entropy(x, torch.tensor(yp)), p)}

    # Softmax cross-entropy: integer labels with smoothing, and probability targets.
    logits = rng.normal(size=(4, 5)) * 2
    labels = np.array([0, 3, 4, 1])
    out["softmaxCe"] = {
        "logits": logits,
        "labels": labels,
        "smoothing": 0.1,
        **value_and_grad(lambda x: F.cross_entropy(x, torch.tensor(labels), label_smoothing=0.1), logits),
    }
    soft = rng.dirichlet(np.ones(5), size=4)
    out["softmaxCeSoft"] = {
        "logits": logits,
        "targets": soft,
        **value_and_grad(lambda x: F.cross_entropy(x, torch.tensor(soft)), logits),
    }

    # Focal loss (torchvision's sigmoid_focal_loss, written out), γ = 2, α = 0.25, on the first six logits.
    y6 = y[:6]

    def focal6(x: torch.Tensor) -> torch.Tensor:
        yt = torch.tensor(y6)
        pr = torch.sigmoid(x)
        ce = F.binary_cross_entropy_with_logits(x, yt, reduction="none")
        pt = pr * yt + (1 - pr) * (1 - yt)
        return ((0.25 * yt + 0.75 * (1 - yt)) * ce * (1 - pt) ** 2).mean()

    out["focal"] = {"z": z[:6], "y": y6, **value_and_grad(focal6, z[:6])}

    # Multiclass hinges.
    scores = rng.normal(size=(4, 5))
    out["westonWatkins"] = {
        "scores": scores,
        "labels": labels,
        **value_and_grad(lambda x: F.multi_margin_loss(x, torch.tensor(labels)) * 5, scores),
    }

    def crammer(x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(())
        for i, yi in enumerate(labels):
            others = torch.stack([x[i, j] for j in range(5) if j != yi])
            total = total + torch.clamp(1 + others.max() - x[i, yi], min=0)
        return total / len(labels)

    out["crammerSinger"] = {"scores": scores, "labels": labels, **value_and_grad(crammer, scores)}

    # Regression losses.
    pred = rng.normal(size=8) * 2
    target = rng.normal(size=8) * 2
    tt = torch.tensor(target)
    out["regression"] = {
        "pred": pred,
        "target": target,
        "mse": value_and_grad(lambda x: F.mse_loss(x, tt), pred),
        "mae": value_and_grad(lambda x: F.l1_loss(x, tt), pred),
        "huber": value_and_grad(lambda x: F.huber_loss(x, tt, delta=1.5), pred),
        "logCosh": value_and_grad(lambda x: torch.log(torch.cosh(x - tt)).mean(), pred),
        "pinball": value_and_grad(lambda x: torch.maximum(0.8 * (tt - x), (0.8 - 1) * (tt - x)).mean(), pred),
    }
    counts = rng.poisson(3, size=8).astype(float)
    out["poisson"] = {
        "eta": pred / 2,
        "counts": counts,
        **value_and_grad(lambda x: F.poisson_nll_loss(x, torch.tensor(counts), log_input=True, full=False), pred / 2),
    }
    sd = rng.uniform(0.5, 2, size=8)
    out["gaussian"] = {
        "mean": pred,
        "target": target,
        "sd": sd,
        **value_and_grad(lambda x: F.gaussian_nll_loss(x, tt, torch.tensor(sd**2)), pred),
    }

    # Ranking losses on one list.
    s = rng.normal(size=6)
    rel = [2.0, 0.0, 1.0, 0.0, 3.0, 1.0]
    out["ranking"] = {"scores": s, "rel": rel}
    for kind in [
        "rankNet",
        "pairwiseHinge",
        "lambdaRank",
        "listwiseSoftmax",
        "listNet",
        "listMle",
        "pointwiseBce",
        "pointwiseSquaredError",
        "approxNdcg",
    ]:
        out["ranking"][kind] = value_and_grad(lambda x, k=kind: ranking(x, rel, k), s)  # type: ignore[index]

    # Sampled softmax with the logQ correction, NCE and negative sampling.
    B, m = 3, 4
    pos = rng.normal(size=B)
    neg = rng.normal(size=(B, m))
    logq = np.log(rng.dirichlet(np.ones(10), size=B)[:, :m])

    def sampled(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.tensor(neg) - torch.tensor(logq) - math.log(m)
        allv = torch.cat([x[:, None], shifted], 1)
        return (torch.logsumexp(allv, 1) - x).mean()

    out["sampledSoftmax"] = {"pos": pos, "neg": neg, "logQ": logq, **value_and_grad(sampled, pos)}
    logq_pos = np.log(rng.uniform(0.01, 0.2, size=B))

    def nce(x: torch.Tensor) -> torch.Tensor:
        dp = x - (torch.tensor(logq_pos) + math.log(m))
        dn = torch.tensor(neg) - (torch.tensor(logq) + math.log(m))
        return (F.softplus(-dp) + F.softplus(dn).sum(1)).mean()

    out["nce"] = {"pos": pos, "neg": neg, "logQ": logq, "logQPositive": logq_pos, **value_and_grad(nce, pos)}

    # InfoNCE with cosine similarity, τ = 0.5, symmetric.
    a = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 3))

    def info(x: torch.Tensor) -> torch.Tensor:
        logits = F.normalize(x, dim=1) @ F.normalize(torch.tensor(b), dim=1).T / 0.5
        lab = torch.arange(4)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))

    out["infoNce"] = {"a": a, "b": b, **value_and_grad(info, a)}

    # Triplet: torch adds 1e-6 inside the distances, so the reference is written without it.
    n_ = rng.normal(size=(4, 3))

    def trip(x: torch.Tensor) -> torch.Tensor:
        dp = torch.linalg.norm(x - torch.tensor(b), dim=1)
        dn = torch.linalg.norm(x - torch.tensor(n_), dim=1)
        return torch.clamp(dp - dn + 1, min=0).mean()

    out["triplet"] = {"a": a, "p": b, "n": n_, **value_and_grad(trip, a)}

    # Distillation: T² KL(softmax(t/T) ‖ softmax(s/T)).
    teacher = rng.normal(size=(3, 4))
    student = rng.normal(size=(3, 4))

    def distil(x: torch.Tensor) -> torch.Tensor:
        T = 2.0
        return (
            F.kl_div(
                F.log_softmax(x / T, 1),
                F.log_softmax(torch.tensor(teacher) / T, 1),
                reduction="batchmean",
                log_target=True,
            )
            * T**2
        )

    out["distillation"] = {"student": student, "teacher": teacher, **value_and_grad(distil, student)}
    return out
