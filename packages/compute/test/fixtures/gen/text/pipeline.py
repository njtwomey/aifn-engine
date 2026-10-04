"""aifn-compute/text/pipeline against Hugging Face `tokenizers`: pre-tokenisers on crafted text, and whole pipelines
(BPE,
byte-level BPE, WordPiece with a BERT template, Unigram with Metaspace, SentencePiece-style BPE with byte fallback,
word level), each trained here on a small corpus, then rebuilt in aifn from its vocabulary and merges and compared
encoding by encoding (ids, tokens, offsets, word ids, masks), with truncation, overflow and padding. Offsets are in
Unicode code points, as `tokenizers` reports them.

`tokenizers` (0.22 or later) is in the `fixtures` dependency group. The cases are recomputed only when it imports;
otherwise the recorded `text/pipeline.json` is kept. Refresh with `uv run python
packages/compute/test/fixtures/generate.py
text/pipeline`.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

OUT = Path(__file__).parents[2] / "text" / "pipeline.json"

PATTERN_INPUTS = [
    "Hello world! It's 2024: we'll ship 1234567 items, don't we?",
    "I'M SHOUTING 'S 'T 'Re camelCaseWords and HTTPServer2Go",
    "  leading spaces\tand tabs\n\nnewlines   \n  trailing  ",
    "naïve café — 東京タワー, привет мир! 👍🏽👨‍👩‍👧 ½ x² ①",
    "price: $3.50/kg, 99.9% off!!!   ...  ok\r\nnext line",
    "é combining and क्ष Devanagari",
    "path/to/file.txt and a+b=c; <tag> {json: [1,2]}",
]

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "A tokeniser splits text into tokens; tokens become ids.",
    "Byte-pair encoding merges the most frequent pair of symbols, again and again.",
    "WordPiece merges the pair that most raises the likelihood of the corpus.",
    "The unigram model keeps the pieces whose removal costs the most likelihood.",
    "Lower lowest newer newest wider widest: the old examples of subword units.",
    "Numbers like 1234 and 56 are split into digits by some tokenisers.",
    "Don't forget contractions: it's, we're, they'll and I'd.",
    "the the the and and of of to to in in is is that that",
    "tokenisation tokeniser tokenisers tokenised tokens token",
] * 3

ENCODE = [
    "The quick brown fox.",
    "Tokenisers don't split 1234 the same way!",
    "Unseen wörds like café, naïve and 東京 🙂",
    "  spaces   and\ttabs  ",
    "lowest newest widest",
]

PAIRS = [["The quick brown fox jumps over the lazy dog.", "A tokeniser splits text into tokens."], ["short", "pair"]]


def make[T](cls: type[T], *args: object, **options: object) -> T:
    """Construct a `tokenizers` component. Its shipped stubs declare most constructors without their parameters."""
    return cast(Callable[..., T], cls)(*args, **options)


def assign(tok: object, **parts: object) -> None:
    """Set pipeline stages on a `Tokenizer`; the stubs declare them as read-only properties, which have setters."""
    for name, part in parts.items():
        setattr(tok, name, part)


def encoding(e: Any) -> dict[str, object]:
    return {
        "ids": e.ids,
        "tokens": e.tokens,
        "offsets": [list(o) for o in e.offsets],
        "wordIds": [-1 if w is None else w for w in e.word_ids],
        "typeIds": e.type_ids,
        "specialTokensMask": e.special_tokens_mask,
        "attentionMask": e.attention_mask,
        "overflowing": [encoding(o) for o in e.overflowing],
    }


def canonical_unigram(vocab: list[list[Any]]) -> list[tuple[str, float]]:
    """Remove the run-to-run variation of a trained Unigram vocabulary, keeping the trainer's rules.

    The trainer iterates hash maps, so (1) its EM sums run in a varying order, which moves scores in the last digits;
    (2) the characters it must keep but did not learn get the scores `min + 0, min + 1e-4, min + 2e-4, ...` (`min` the
    lowest learned score) in a varying order; (3) equal scores are listed in a varying order. Here scores are rounded
    to 10 significant digits, the fallback scores are dealt to those characters in code-point order (the multiset of
    scores is unchanged), and pieces are sorted by score, then text."""
    head, pieces = vocab[0], [(str(p), float(f"{s:.10g}")) for p, s in vocab[1:]]
    low = min(s for _, s in pieces)

    def on_grid(s: float) -> bool:
        return abs((s - low) / 1e-4 - round((s - low) / 1e-4)) < 1e-4 and s - low < 1e-4 * len(pieces)

    grid = [i for i, (p, s) in enumerate(pieces) if len(p) == 1 and on_grid(s)]
    chars = sorted(pieces[i][0] for i in grid)
    dealt = dict(zip(chars, sorted((pieces[i][1] for i in grid), reverse=True), strict=True))
    pieces = [(p, dealt.get(p, s)) for p, s in pieces]
    pieces.sort(key=lambda ps: (-ps[1], ps[0]))
    return [(str(head[0]), float(head[1])), *pieces]


def hf_cases() -> dict[str, object]:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

    out: dict[str, object] = {}
    pre = {
        "whitespace": pre_tokenizers.Whitespace(),
        "whitespaceSplit": pre_tokenizers.WhitespaceSplit(),
        "bert": pre_tokenizers.BertPreTokenizer(),
        "punctuation": pre_tokenizers.Punctuation(),
        "digits": pre_tokenizers.Digits(individual_digits=True),
        "digitRuns": pre_tokenizers.Digits(individual_digits=False),
        "metaspace": pre_tokenizers.Metaspace(prepend_scheme="always"),
        "metaspaceFirst": pre_tokenizers.Metaspace(prepend_scheme="first"),
        "metaspaceNever": pre_tokenizers.Metaspace(prepend_scheme="never"),
        "byteLevel": pre_tokenizers.ByteLevel(add_prefix_space=False),
        "byteLevelPrefix": pre_tokenizers.ByteLevel(add_prefix_space=True),
    }
    out["preTokenisers"] = [
        {"text": t, **{k: [[s, list(o)] for s, o in p.pre_tokenize_str(t)] for k, p in pre.items()}}
        for t in PATTERN_INPUTS + ENCODE
    ]
    out["byteAlphabet"] = sorted(pre_tokenizers.ByteLevel.alphabet())

    pipelines: dict[str, Tokenizer] = {}

    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    assign(t, pre_tokenizer=pre_tokenizers.Whitespace())
    t.train_from_iterator(
        CORPUS, make(trainers.BpeTrainer, vocab_size=180, special_tokens=["[UNK]"], show_progress=False)
    )
    pipelines["bpe"] = t

    t = Tokenizer(models.BPE())
    assign(t, pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False))
    assign(t, decoder=decoders.ByteLevel())
    t.train_from_iterator(
        CORPUS,
        make(
            trainers.BpeTrainer,
            vocab_size=400,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    pipelines["byteLevelBpe"] = t

    # The trainer numbers the continuation pieces ("##e") in the order it meets them while iterating a hash map, which
    # changes from run to run; merges of equal count are then broken by those ids, so the trained vocabulary changes
    # too. Listing every continuation piece as a training-time special token fixes their ids (in code-point order); the
    # tokeniser is then rebuilt from the trained vocabulary with only the real special tokens.
    wp_specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    wp_normalizer = make(normalizers.Sequence, [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    wp_pre = pre_tokenizers.BertPreTokenizer()
    continuations = sorted(
        {
            f"##{c}"
            for line in CORPUS
            for w, _ in wp_pre.pre_tokenize_str(wp_normalizer.normalize_str(line))
            for c in w[1:]
        }
    )
    t = Tokenizer(make(models.WordPiece, unk_token="[UNK]"))
    assign(t, normalizer=wp_normalizer, pre_tokenizer=wp_pre)
    t.train_from_iterator(
        CORPUS,
        make(
            trainers.WordPieceTrainer, vocab_size=200, special_tokens=wp_specials + continuations, show_progress=False
        ),
    )
    t = Tokenizer(make(models.WordPiece, t.get_vocab(), unk_token="[UNK]"))
    assign(t, normalizer=wp_normalizer, pre_tokenizer=wp_pre, decoder=decoders.WordPiece())
    t.add_special_tokens(wp_specials)
    assign(
        t,
        post_processor=processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))],
        ),
    )
    pipelines["wordPiece"] = t

    t = Tokenizer(make(models.Unigram))
    assign(t, normalizer=normalizers.NFKC(), pre_tokenizer=pre_tokenizers.Metaspace(), decoder=decoders.Metaspace())
    t.train_from_iterator(
        CORPUS,
        make(trainers.UnigramTrainer, vocab_size=150, special_tokens=["<unk>"], unk_token="<unk>", show_progress=False),
    )
    t = Tokenizer(make(models.Unigram, canonical_unigram(json.loads(t.to_str())["model"]["vocab"]), unk_id=0))
    assign(t, normalizer=normalizers.NFKC(), pre_tokenizer=pre_tokenizers.Metaspace(), decoder=decoders.Metaspace())
    t.add_special_tokens(["<unk>"])
    pipelines["unigram"] = t

    # SentencePiece-style BPE with byte fallback, as LLaMA: ▁ for spaces, <0xNN> for characters outside the vocabulary.
    # The byte tokens are ordinary vocabulary entries after the three specials (ids 3 … 258), not special tokens.
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    assign(t, pre_tokenizer=pre_tokenizers.Metaspace(prepend_scheme="first"))
    t.train_from_iterator(
        CORPUS, make(trainers.BpeTrainer, vocab_size=240, special_tokens=["<unk>", "<s>", "</s>"], show_progress=False)
    )
    trained = json.loads(t.to_str())["model"]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"<0x{b:02X}>": 3 + b for b in range(256)}}
    for token, _ in sorted(trained["vocab"].items(), key=lambda kv: kv[1]):
        vocab.setdefault(token, len(vocab))
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in trained["merges"]]
    t = Tokenizer(make(models.BPE, vocab, merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    assign(t, pre_tokenizer=pre_tokenizers.Metaspace(prepend_scheme="first"))
    assign(
        t,
        decoder=decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
        ),
    )
    t.add_special_tokens(["<unk>", "<s>", "</s>"])
    assign(
        t, post_processor=processors.TemplateProcessing(single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", 1)])
    )
    pipelines["sentencePieceBpe"] = t

    t = Tokenizer(make(models.WordLevel, unk_token="[UNK]"))
    assign(t, pre_tokenizer=pre_tokenizers.Whitespace())
    t.train_from_iterator(CORPUS, make(trainers.WordLevelTrainer, special_tokens=["[UNK]"], show_progress=False))
    pipelines["wordLevel"] = t

    cases: dict[str, object] = {}
    for name, tok in pipelines.items():
        model = json.loads(tok.to_str())["model"]
        entry: dict[str, object] = {
            "model": {k: model[k] for k in ("vocab", "merges", "unk_token", "byte_fallback") if k in model},
            "encode": [
                {"text": x, **encoding(tok.encode(x)), "decoded": tok.decode(tok.encode(x).ids)} for x in ENCODE
            ],
        }
        if name == "unigram":
            entry["model"] = {"vocab": model["vocab"], "unk_id": model["unk_id"]}
        if name in ("wordPiece", "sentencePieceBpe"):
            entry["pairs"] = [{"a": a, "b": b, **encoding(tok.encode(a, b))} for a, b in PAIRS]
        cases[name] = entry

    wp = pipelines["wordPiece"]
    wp.enable_truncation(max_length=10, stride=3, strategy="longest_first")
    trunc = {
        "single": [{"text": x, **encoding(wp.encode(x))} for x in [CORPUS[2], CORPUS[4]]],
    }
    wp.enable_truncation(max_length=16, stride=2, strategy="longest_first")
    trunc["pairs"] = [{"a": a, "b": b, **encoding(wp.encode(a, b))} for a, b in PAIRS[:1]]
    # tokenizers 0.22+ tokenises each sequence only until its splits hold max_length tokens (the first sequence is
    # exempt under only_second), so overflowing windows stop there; specials typed in the text count as splits.
    long_a = CORPUS[2] + " " + CORPUS[3]
    variants = []
    for max_length, stride, strategy, direction, a, b in [
        (10, 2, "longest_first", "left", long_a, None),
        (12, 0, "longest_first", "right", "the [MASK] of [MASK] and the dog. " + CORPUS[0], None),
        (12, 3, "longest_first", "left", CORPUS[4] + " [MASK] the end", None),
        (20, 2, "only_first", "right", long_a, PAIRS[0][1]),
        (20, 2, "only_second", "right", PAIRS[0][1], long_a),
        (14, 1, "longest_first", "left", PAIRS[0][0], long_a),
    ]:
        wp.enable_truncation(max_length=max_length, stride=stride, strategy=strategy, direction=direction)
        e = wp.encode(a) if b is None else wp.encode(a, b)
        variants.append(
            {
                "maxLength": max_length,
                "stride": stride,
                "strategy": strategy,
                "direction": direction,
                "a": a,
                "b": b,
                **encoding(e),
            }
        )
    trunc["variants"] = variants
    wp.no_truncation()
    wp.enable_padding(pad_id=wp.token_to_id("[PAD]"), pad_token="[PAD]")
    trunc["batch"] = [encoding(e) for e in wp.encode_batch(ENCODE)]
    wp.enable_padding(pad_id=wp.token_to_id("[PAD]"), pad_token="[PAD]", length=16, direction="left")
    trunc["left"] = [encoding(e) for e in wp.encode_batch(ENCODE[:2])]
    cases["truncation"] = trunc
    out["pipelines"] = cases
    return out


def cases() -> dict[str, object]:
    try:
        return hf_cases()
    except ImportError as e:
        print(f"  hf_cases: {e.name} is not installed; keeping the recorded cases")
        return json.loads(OUT.read_text()) if OUT.exists() else {}
