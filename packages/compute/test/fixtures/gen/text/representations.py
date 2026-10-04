"""aifn-compute/text/representations and the truncated SVD, shingles and MinHash: scikit-learn's CountVectorizer and
TruncatedSVD, numpy's SVD and 3CosAdd analogies, and datasketch's MinHash estimates."""

from typing import cast

import numpy as np
from datasketch import MinHash
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

DOCS = [
    "the cat chased the mouse",
    "the dog chased the cat",
    "the cook baked the bread",
    "the baker baked the cake",
    "the bus drove to the city",
    "the car drove to the town",
]


def term_document() -> dict[str, object]:
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    x = cast(csr_matrix, cv.fit_transform(DOCS)).toarray()
    return {"vocabulary": cv.get_feature_names_out().tolist(), "termDocument": x.T}


def truncated() -> dict[str, object]:
    rng = np.random.default_rng(0)
    out: dict[str, object] = {}
    for name, (m, n, k) in {"small": (12, 9, 3), "tall": (90, 70, 5), "wide": (70, 110, 6)}.items():
        a = rng.random((m, n)) * (rng.random((m, n)) < 0.2)
        a[:, 0] += 1.0  # a dominant direction, so the leading singular values are well separated
        svd = TruncatedSVD(n_components=k, algorithm="arpack", random_state=0).fit(a)
        u, s, _ = np.linalg.svd(a, full_matrices=False)
        out[name] = {
            "matrix": a,
            "k": k,
            "singularValues": svd.singular_values_,
            "energy": s[:k] ** 2 / (s**2).sum(),
            "lsaRows": np.abs(u[:, :k] * s[:k]),
        }
    return out


def analogies() -> dict[str, object]:
    rng = np.random.default_rng(1)
    v = rng.normal(size=(12, 4))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cases = []
    for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 2, 9)]:
        t = u[a] - u[b] + u[c]
        cos = u @ t / np.linalg.norm(t)
        cos[[a, b, c]] = -np.inf
        order = np.argsort(-cos, kind="stable")[:3]
        cases.append({"abc": [a, b, c], "answers": order, "cosines": cos[order]})
    return {"vectors": v, "cases": cases}


def minhash() -> dict[str, object]:
    def shingles(text: str, k: int) -> set[str]:
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    pairs = [
        ("the cat watches the grass", "some cats watch the grass"),
        ("the cook bakes the bread in the kitchen", "the cook bakes the cake in the kitchen"),
        ("a red bus stops near the market", "the painter paints the wall red"),
    ]
    out = []
    for x, y in pairs:
        a, b = shingles(x, 4), shingles(y, 4)
        ma, mb = MinHash(num_perm=256, seed=1), MinHash(num_perm=256, seed=1)
        for s in a:
            ma.update(s.encode())
        for s in b:
            mb.update(s.encode())
        out.append({"a": x, "b": y, "jaccard": len(a & b) / len(a | b), "datasketch": ma.jaccard(mb)})
    return {"pairs": out}


def cases() -> dict[str, object]:
    return {"docs": DOCS, **term_document(), "truncated": truncated(), "analogies": analogies(), "minhash": minhash()}
