"""Golden values for aifn-methods/inference/sequence-models: a linear-chain CRF trained by CRFsuite (python-crfsuite,
L-BFGS with L2) on a small tagging task, with every state and transition weight, and the marginals and Viterbi tags of
held-out sequences. The attributes are the CRF++ template strings of `TEMPLATES`, expanded here independently of aifn:
unigram strings conjoined with every label (`feature.possible_states`) and all label pairs as transitions (`B`)."""

import os
import tempfile

import numpy as np

# pycrfsuite re-exports its classes from a compiled Cython module with `import *`, which Pyright cannot see.
import pycrfsuite

TEMPLATES = ["U00:%x[-1,0]", "U01:%x[0,0]", "U02:%x[1,0]", "U03:%x[0,1]", "U04:%x[0,0]/%x[0,1]", "B"]

WORDS = {
    "D": ["the", "a", "this"],
    "N": ["dog", "cat", "park", "ball", "man"],
    "V": ["runs", "sees", "likes", "chases"],
    "A": ["big", "red", "old"],
}


def shape(word: str) -> str:
    return "s" if word.endswith("s") else "x"


def cell(rows: list[list[str]], n: int, r: int, c: int) -> str:
    at = n + r
    if at < 0:
        return f"_B{at}"
    if at >= len(rows):
        return f"_B+{at - len(rows) + 1}"
    return rows[at][c]


def attributes(rows: list[list[str]]) -> list[list[str]]:
    out = []
    for n in range(len(rows)):
        out.append(
            [
                f"U00:{cell(rows, n, -1, 0)}",
                f"U01:{cell(rows, n, 0, 0)}",
                f"U02:{cell(rows, n, 1, 0)}",
                f"U03:{cell(rows, n, 0, 1)}",
                f"U04:{cell(rows, n, 0, 0)}/{cell(rows, n, 0, 1)}",
            ]
        )
    return out


def sentence(rng: np.random.Generator) -> tuple[list[list[str]], list[str]]:
    pattern = ["D", *(["A"] * int(rng.integers(0, 2))), "N", "V", "D", *(["A"] * int(rng.integers(0, 2))), "N"]
    # Some noise in the tags, so the optimum is interior.
    tags = [t if rng.random() > 0.08 else str(rng.choice(list(WORDS))) for t in pattern]
    words = [str(rng.choice(WORDS[t])) for t in pattern]
    return [[w, shape(w)] for w in words], tags


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    train = [sentence(rng) for _ in range(40)]
    test = [sentence(rng) for _ in range(5)]
    c2 = 0.1
    trainer = pycrfsuite.Trainer(algorithm="lbfgs", verbose=False)  # pyright: ignore[reportAttributeAccessIssue]
    for rows, tags in train:
        trainer.append(attributes(rows), tags)
    trainer.set_params(
        {
            "c1": 0.0,
            "c2": c2,
            "max_iterations": 5000,
            "epsilon": 1e-12,
            "delta": 1e-14,
            "period": 20,
            "feature.minfreq": 0,
            "feature.possible_states": True,
            "feature.possible_transitions": True,
        }
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.crfsuite")
        trainer.train(path)
        tagger = pycrfsuite.Tagger()  # pyright: ignore[reportAttributeAccessIssue]
        tagger.open(path)
        info = tagger.info()
        state = [[a, y, w] for (a, y), w in info.state_features.items()]
        trans = [[a, b, w] for (a, b), w in info.transitions.items()]
        held = []
        for rows, _ in test:
            tagger.set(attributes(rows))
            viterbi = tagger.tag()
            labels = tagger.labels()
            marg = [[tagger.marginal(y, n) for y in labels] for n in range(len(rows))]
            held.append({"rows": rows, "viterbi": viterbi, "labels": labels, "marginals": marg})
        tagger.close()
    return {
        "templates": "\n".join(TEMPLATES),
        "c2": c2,
        "train": [{"rows": r, "labels": t} for r, t in train],
        "state": state,
        "transitions": trans,
        "test": held,
    }
