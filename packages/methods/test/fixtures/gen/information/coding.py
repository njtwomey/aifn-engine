"""Golden values for aifn-methods/information/coding: Huffman codes from a heapq reference, and brute-force optima."""

import heapq
import itertools
import math

import numpy as np


def huffman(p: list[float], arity: int, ties: str) -> dict[str, object]:
    """Huffman's algorithm on a binary heap, with the same queue order and digit rule as aifn's `huffmanSteps`."""
    k = len(p)
    dummies = 0 if k <= 1 else (arity - 1 - (k - 1) % (arity - 1)) % (arity - 1)
    weights = list(p) + [0.0] * dummies
    children: list[list[int]] = [[] for _ in weights]

    def key(node: int) -> tuple[float, int, int]:
        merged = len(children[node]) > 0
        if ties == "minimum-variance":
            return (round(weights[node], 12), 1 if merged else 0, node)
        return (round(weights[node], 12), 0 if merged else 1, -node if merged else node)

    heap = [(key(v), v) for v in range(len(weights))]
    heapq.heapify(heap)
    merges: list[list[int]] = []
    while len(heap) > 1:
        popped = [heapq.heappop(heap)[1] for _ in range(arity)]
        node = len(weights)
        weights.append(sum(weights[v] for v in popped))
        children.append(popped)
        merges.append(popped)
        heapq.heappush(heap, (key(node), node))
    codewords = [""] * k
    stack = [(heap[0][1], "")]
    while stack:
        v, prefix = stack.pop()
        if children[v]:
            stack.extend((c, prefix + np.base_repr(i, 36).lower()) for i, c in enumerate(children[v]))
        elif v < k:
            codewords[v] = prefix
    lengths = [len(c) for c in codewords]
    expected = sum(pi * li for pi, li in zip(p, lengths, strict=True))
    variance = sum(pi * (li - expected) ** 2 for pi, li in zip(p, lengths, strict=True))
    entropy = -sum(pi * math.log(pi, arity) for pi in p if pi > 0)
    return {
        "arity": arity,
        "ties": ties,
        "dummies": dummies,
        "merges": merges,
        "codewords": codewords,
        "lengths": lengths,
        "expectedLength": expected,
        "lengthVariance": variance,
        "entropy": entropy,
        "kraft": sum(arity**-li for li in lengths),
    }


def brute_force(p: list[float], arity: int) -> float:
    """The least expected length over every length vector that satisfies Kraft's inequality (lengths ≤ K)."""
    k = len(p)
    best = math.inf
    for lengths in itertools.product(range(1, k + 1), repeat=k):
        if sum(arity**-li for li in lengths) <= 1 + 1e-12:
            best = min(best, sum(pi * li for pi, li in zip(p, lengths, strict=True)))
    return best


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    sources: dict[str, list[float]] = {
        "textbook": [0.4, 0.2, 0.2, 0.1, 0.1],
        "dyadic": [0.5, 0.25, 0.125, 0.0625, 0.0625],
        "uniform6": [1 / 6] * 6,
        "skewed": [0.9, 0.05, 0.03, 0.02],
        "zipf8": list(1 / np.arange(1, 9) / np.sum(1 / np.arange(1, 9))),
    }
    for i in range(4):
        sources[f"dirichlet{i}"] = list(rng.dirichlet(np.ones(int(rng.integers(3, 7)))))
    out: dict[str, object] = {}
    for name, p in sources.items():
        codes = [huffman(p, arity, ties) for arity in (2, 3) for ties in ("minimum-variance", "merged-first")]
        optimum = {str(arity): brute_force(p, arity) for arity in (2, 3)} if len(p) <= 6 else None
        out[name] = {"p": p, "codes": codes, "optimum": optimum}
    return out
