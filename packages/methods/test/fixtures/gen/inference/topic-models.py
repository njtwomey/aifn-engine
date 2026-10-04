"""Reference values for aifn-methods/inference/topic-models: pLSA expectation-maximisation (Hofmann, 1999) written
directly in NumPy from given initial distributions, on a small random corpus; P(w | z), P(z | d) and the
log-likelihood after a fixed number of steps."""

from typing import Any

import numpy as np


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(12)
    d, v, k = 12, 15, 3
    docs = [list(rng.integers(0, v, size=int(rng.integers(5, 15)))) for _ in range(d)]
    n = np.zeros((d, v))
    for i, doc in enumerate(docs):
        for w in doc:
            n[i, w] += 1
    phi0 = rng.uniform(0.5, 1.5, size=(k, v))
    phi0 /= phi0.sum(1, keepdims=True)
    theta0 = rng.uniform(0.5, 1.5, size=(d, k))
    theta0 /= theta0.sum(1, keepdims=True)
    runs: list[dict[str, Any]] = []
    phi, theta = phi0.copy(), theta0.copy()
    for step in range(1, 11):
        r = theta[:, :, None] * phi[None, :, :]  # [d, k, v]
        r /= r.sum(1, keepdims=True)
        e = n[:, None, :] * r
        phi = e.sum(0)
        phi /= phi.sum(1, keepdims=True)
        theta = e.sum(2)
        theta /= theta.sum(1, keepdims=True)
        if step in (1, 10):
            ll = float(np.sum(n * np.log(theta @ phi + (n == 0))))
            runs.append({"steps": step, "topicWord": phi, "docTopic": theta, "logLikelihood": ll})
    return {
        "documents": [[int(w) for w in doc] for doc in docs],
        "topics": k,
        "vocabulary": v,
        "topicWord0": phi0,
        "docTopic0": theta0,
        "plsa": runs,
    }
