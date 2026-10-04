"""Reference values for the topic coherence of aifn-compute/text/cooccurrence, computed directly in NumPy from document
co-occurrence as gensim's CoherenceModel does with boolean documents: NPMI with epsilon 1e-12 averaged over pairs
i < j, and UMass log((D(w_i, w_j) + 1) / D(w_j)) averaged over pairs i > j."""

from itertools import combinations
from typing import Any

import numpy as np


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    vocab = 20
    docs = [list(rng.choice(vocab, size=int(rng.integers(3, 9)), replace=True)) for _ in range(40)]
    topics = [[0, 3, 5, 7], [1, 2, 9, 11, 13], [4, 6, 8]]
    has = np.zeros((len(docs), vocab), bool)
    for d, doc in enumerate(docs):
        has[d, doc] = True
    n = len(docs)

    def npmi(top: list[int]) -> float:
        vals: list[float] = []
        for j, i in combinations(range(len(top)), 2):
            wi, wj = top[i], top[j]
            pij = float(np.sum(has[:, wi] & has[:, wj])) / n + 1e-12
            pi = float(np.sum(has[:, wi])) / n
            pj = float(np.sum(has[:, wj])) / n
            vals.append(np.log(pij / (pi * pj)) / -np.log(pij))
        return float(np.mean(vals))

    def umass(top: list[int]) -> float:
        vals: list[float] = []
        for j, i in combinations(range(len(top)), 2):
            wi, wj = top[i], top[j]
            vals.append(np.log((np.sum(has[:, wi] & has[:, wj]) + 1) / np.sum(has[:, wj])))
        return float(np.mean(vals))

    return {
        "documents": [[int(w) for w in doc] for doc in docs],
        "topics": topics,
        "npmi": [npmi(t) for t in topics],
        "umass": [umass(t) for t in topics],
    }
