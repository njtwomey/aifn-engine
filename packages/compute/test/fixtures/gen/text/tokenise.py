"""aifn-compute/text/tokenise against its references: NLTK's TreebankWordTokenizer, TweetTokenizer and the pre-trained
English
Punkt model, and the GPT-2, cl100k_base and o200k_base pre-tokeniser patterns run by the `regex` module (the engine
tiktoken's patterns were written for).

The references are not dependencies of the project. Each section is recomputed only when its library imports (NLTK
3.9.1 and regex 2.5 were used); otherwise the section already in `text/tokenise.json` is kept. To refresh, run
`python packages/compute/test/fixtures/generate.py text/tokenise` with an interpreter that has them.
"""

import json
import pickle
from pathlib import Path

OUT = Path(__file__).parents[2] / "text" / "tokenise.json"

TREEBANK = [
    "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.",
    "They'll save and invest more.",
    "hi, my name can't hello,",
    'She said, "I don\'t know -- maybe (or not) [sic] {x} <y>?"',
    "I cannot believe it's gonna rain; we're wanna go! 'Tis true.",
    "The U.S.A. isn't O'Neill's... is it? 3,000 people: 50% @home #1 & more.",
    "''Quoted'' and ``quoted'' text, 'single' quotes'",
    "Ünïcödé café's naïveté isn't résumé-ready, ½ done.",
    "Lemme gimme d'ye more'n gotta Whaddya",
    "Hello.\n",
    "End with colon:\n",
    "\"Start quote\" and DON'T SHOUT, YOU'LL see.",
]

CASUAL = [
    "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--",
    "@remy: This is waaaaayyyy too much for you!!!!!!",
    "Check https://example.com/path?q=1 and www.foo.org or foo.co.uk/bar, mail me@x.com",
    "Call +1 (800) 555-1234 now :D ;) 8-) </3 x-D",
    "Emoji 👍🏽 family 👨‍👩‍👧 flag 🇮🇪 heart ❤️ and #hash_tag-ok @user_name123",
    "Price: &pound;100 &amp; 5&#37; off &bogus; &#x263A;",
    "Numbers 3.14 -2,5 1/2 12:30 and ... dots . . . end",
    "<b>bold</b> it's state-of-the-art O'Reilly's",
    "@abcdefghijklmnopqrstuvwxyz is long, @ok_handle fine",
]

SENTENCES = [
    "Mr. Smith went to Washington. He arrived at 3 p.m. on Jan. 5. Then he left!",
    '"Really?" she asked. The U.S. economy grew 3.5 percent. Dr. Who? Yes.',
    "This is one. This is two... And three? Four! Five.",
    "J. R. R. Tolkien wrote books. They sold well. Prof. Brown agrees.",
    "Prices rose 5 percent in Oct. The board met in Washington. It voted no.",
    "I saw Mt. Everest. It was tall. (It really was.) We went home.",
    "The meeting ended at 5 p.m. The next one is tomorrow.",
    "Version 2. is out. Download it now.",
]

PATTERN_INPUTS = [
    "Hello world! It's 2024: we'll ship 1234567 items, don't we?",
    "I'M SHOUTING 'S 'T 'Re camelCaseWords and HTTPServer2Go",
    "  leading spaces\tand tabs\n\nnewlines   \n  trailing  ",
    "naïve café — 東京タワー, привет мир! 👍🏽👨‍👩‍👧 ½ x² ①",
    "price: $3.50/kg, 99.9% off!!!   ...  ok\r\nnext line",
    "é combining and क्ष Devanagari",
    "path/to/file.txt and a+b=c; <tag> {json: [1,2]}",
]

PATTERNS = {
    "gpt2": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    "cl100k": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",  # noqa: E501
    "o200k": "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    ),
}


def treebank_cases() -> dict[str, object]:
    from nltk.tokenize import TreebankWordTokenizer
    from nltk.tokenize.casual import TweetTokenizer

    tb = TreebankWordTokenizer()
    out: dict[str, object] = {
        "treebank": [
            {
                "text": t,
                "tokens": tb.tokenize(t),
                "parens": tb.tokenize(t, convert_parentheses=True),
                "spans": [list(s) for s in tb.span_tokenize(t)],
            }
            for t in TREEBANK
        ]
    }
    options = [
        {},
        {"preserve_case": False},
        {"reduce_len": True, "strip_handles": True},
        {"match_phone_numbers": False},
    ]
    out["casual"] = [
        {"text": t, "options": o, "tokens": TweetTokenizer(**o).tokenize(t)} for t in CASUAL for o in options
    ]
    punkt = Path.home() / "nltk_data" / "tokenizers" / "punkt" / "PY3" / "english.pickle"
    with punkt.open("rb") as f:
        sentences = pickle.load(f)
    out["sentences"] = [{"text": t, "sentences": sentences.tokenize(t)} for t in SENTENCES]
    return out


def pattern_cases() -> dict[str, object]:
    import regex

    return {
        "patterns": [{"text": t, **{name: regex.findall(p, t) for name, p in PATTERNS.items()}} for t in PATTERN_INPUTS]
    }


def cases() -> dict[str, object]:
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for section in (treebank_cases, pattern_cases):
        try:
            out.update(section())
        except ImportError as e:
            print(f"  {section.__name__}: {e.name} is not installed; keeping the recorded cases")
    return out
