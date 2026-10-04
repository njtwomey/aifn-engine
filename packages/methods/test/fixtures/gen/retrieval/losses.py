"""Golden values for aifn-methods/retrieval/losses' in-batch softmax (Yi et al., 2019) in torch: logits
qᵢ·vⱼ/τ − log Q_j (the logQ correction by column), torch.nn.functional.cross_entropy against the diagonal, mean over
the batch; symmetric adds the transposed logits' cross-entropy and halves. Value and gradients in both towers."""

import numpy as np
import torch
import torch.nn.functional as F


def in_batch(q: np.ndarray, v: np.ndarray, temperature: float, log_q: np.ndarray | None, symmetric: bool):
    qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    vt = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    logits = qt @ vt.T / temperature
    if log_q is not None:
        logits = logits - torch.tensor(log_q, dtype=torch.float64)[None, :]
    labels = torch.arange(len(q))
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
    loss.backward()
    assert qt.grad is not None and vt.grad is not None
    return {"value": loss.item(), "gradQueries": qt.grad.numpy(), "gradItems": vt.grad.numpy()}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    B, d = 5, 3
    q = rng.normal(size=(B, d))
    v = rng.normal(size=(B, d))
    log_q = np.log(rng.dirichlet(np.ones(B)))
    return {
        "inBatchSoftmax": {
            "queries": q,
            "items": v,
            "logQ": log_q,
            "plain": in_batch(q, v, 1.0, None, False),
            "corrected": {"temperature": 0.5, **in_batch(q, v, 0.5, log_q, False)},
            "symmetric": {"temperature": 0.5, **in_batch(q, v, 0.5, log_q, True)},
        }
    }
