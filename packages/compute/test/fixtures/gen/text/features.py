"""aifn-compute/text: scikit-learn's text vectorisers (CountVectorizer, TfidfVectorizer, HashingVectorizer, the
analyzers and
the MurmurHash3 they hash with), Python's Unicode normalisation and case folding, and the token pattern."""

import re
import unicodedata
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.utils import murmurhash3_32

DOCS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "a dog and a cat played",
    "the bird sang",
]

MESSY = [
    "Café ﬁancée naïve résumé — Straße ΣΊΣΥΦΟΣ",
    "The QUICK brown fox's tail: don't stop_now 42 times!",
    "Ｆｕｌｌ-width ① x² ẞ and İstanbul",  # noqa: RUF001 (full-width letters on purpose: NFKC folds them)
]


def dense(matrix: Any) -> np.ndarray:
    """A vectoriser's sparse document-term matrix as a dense array (sklearn is unstubbed; returns are unknown)."""
    return matrix.toarray()


def tokens(doc: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", doc.lower())


def vectoriser_cases() -> dict[str, object]:
    out: dict[str, object] = {}
    cv = CountVectorizer()
    counts = dense(cv.fit_transform(DOCS))
    out["count"] = {"vocabulary": cv.get_feature_names_out().tolist(), "matrix": counts}
    cvb2 = CountVectorizer(binary=True, ngram_range=(1, 2)).fit(DOCS)
    out["count_binary_bigrams"] = {
        "vocabulary": cvb2.get_feature_names_out().tolist(),
        "matrix": dense(cvb2.transform(DOCS)),
    }
    cvdf = CountVectorizer(min_df=2, max_df=0.7).fit(DOCS)
    out["count_df_limits"] = {"vocabulary": cvdf.get_feature_names_out().tolist()}
    for name, kw in {
        "default": {},
        "sublinear": {"sublinear_tf": True},
        "no_smooth": {"smooth_idf": False},
        "l1": {"norm": "l1"},
        "no_norm_no_idf": {"norm": None, "use_idf": False},
    }.items():
        tv = TfidfVectorizer(**kw).fit(DOCS)
        out[f"tfidf_{name}"] = {
            "options": kw,
            "vocabulary": tv.get_feature_names_out().tolist(),
            "matrix": dense(tv.transform(DOCS)),
            "idf": tv.idf_ if kw.get("use_idf", True) else None,
        }
    # norm is inferred as str from its default; None (no normalisation) is documented.
    hv = HashingVectorizer(n_features=8, norm=None, token_pattern=r"(?u)\b\w+\b")  # pyright: ignore[reportArgumentType]
    out["hashing_8"] = {"matrix": dense(hv.transform(DOCS))}
    hv2 = HashingVectorizer(n_features=16, alternate_sign=False)
    out["hashing_16_unsigned_l2"] = {"matrix": dense(hv2.transform(DOCS))}
    return out


def analyzer_cases() -> dict[str, object]:
    text = "the  cat\tsat on\nthe mat"
    out: dict[str, object] = {"text": text}
    for name, kw in {
        "char_2_3": {"analyzer": "char", "ngram_range": (2, 3)},
        "char_wb_2_4": {"analyzer": "char_wb", "ngram_range": (2, 4)},
        "char_wb_5": {"analyzer": "char_wb", "ngram_range": (5, 5)},
        "word_1_3": {"analyzer": "word", "ngram_range": (1, 3)},
    }.items():
        out[name] = CountVectorizer(lowercase=False, **kw).build_analyzer()(text)
    return out


def hash_cases() -> dict[str, object]:
    words = ["", "a", "ab", "abc", "abcd", "abcde", "the", "cat", "naïve", "日本語", "🙂", "hello world", "x" * 37]
    return {
        "words": words,
        "hashes": [murmurhash3_32(w, seed=0) for w in words],
        "seeded": [murmurhash3_32(w, seed=42) for w in words],
    }


def unicode_cases() -> dict[str, object]:
    out: list[dict[str, str | list[str]]] = []
    for s in MESSY:
        out.append(
            {
                "text": s,
                "nfkc": unicodedata.normalize("NFKC", s),
                "casefold": s.casefold(),
                "nfkc_casefold": unicodedata.normalize("NFC", unicodedata.normalize("NFKC", s).casefold()),
                # scikit-learn's strip_accents='unicode'
                "strip_accents": "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)),
                "sklearn_tokens": re.findall(r"(?u)\b\w\w+\b", s),
            }
        )
    return {"strings": out}


def cases() -> dict[str, object]:
    return {
        "docs": DOCS,
        "vectorisers": vectoriser_cases(),
        "analyzers": analyzer_cases(),
        "murmurhash3": hash_cases(),
        "unicode": unicode_cases(),
        "tokens": [tokens(d) for d in DOCS],
        "numpy": np.__version__,
    }
