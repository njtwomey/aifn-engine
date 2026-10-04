"""Vendor a frequency-ranked subset of the Moby Hyphenator II word list into aifn-methods.

Usage (the source file is not kept in the repository; download it first):

    curl -sSL -o .scratch/hyph/mhyph.txt https://www.gutenberg.org/files/3204/files/mhyph.txt
    uv run python scripts/hyphenation.py .scratch/hyph/mhyph.txt

Source: Grady Ward, "Moby Hyphenator II" (Project Gutenberg etext #3204), placed in the public domain by grant from
the author, January 2001. The file is Mac Roman text, one entry per line (CRLF), with byte 0xA5 ("•") marking each
hyphenation point. Only entries that are a single lowercase word of the letters a–z are kept: entries with spaces,
hyphens, apostrophes, capitals (proper nouns) or accented letters are dropped. A word listed twice keeps its first
hyphenation. Words are ranked by their frequency in the Brown corpus (NLTK's copy, lowercased) and the most frequent
`--words` of length 4–15 with at least two Brown occurrences are written, one per line in rank order, with "-" at each
hyphenation point.
"""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

from nltk.corpus import brown

OUT = Path(__file__).resolve().parent.parent / "packages/methods/src/data/real/hyphenation/words.ts"
MARK = 0xA5
ENTRY = re.compile(rb"^[a-z\xa5]+$")


def moby(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    raw = path.read_bytes()
    words: dict[str, str] = {}
    stats = Counter[str]()
    for line in raw.split(b"\r\n"):
        if not line:
            continue
        stats["entries"] += 1
        if not ENTRY.match(line) or line.startswith(b"\xa5") or line.endswith(b"\xa5") or b"\xa5\xa5" in line:
            stats["dropped"] += 1
            continue
        hyphenated = line.replace(bytes([MARK]), b"-").decode("ascii")
        word = hyphenated.replace("-", "")
        if word in words:
            stats["duplicates"] += 1
            if words[word] != hyphenated:
                stats["conflicts"] += 1
            continue
        words[word] = hyphenated
    stats["kept"] = len(words)
    return words, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="mhyph.txt from Project Gutenberg etext #3204")
    parser.add_argument("--words", type=int, default=8000, help="words to keep")
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    words, stats = moby(args.source)
    counts = Counter(w.lower() for w in brown.words() if w.isalpha())
    ranked = sorted(
        (w for w in words if 4 <= len(w) <= 15 and counts[w] >= 2),
        key=lambda w: (-counts[w], w),
    )[: args.words]
    lines = [words[w] for w in ranked]
    points = sum(h.count("-") for h in lines)
    body = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "/**\n"
        " * The vendored word list of `mobyHyphenation`: do not edit by hand; regenerate with\n"
        " *\n"
        " *   uv run python scripts/hyphenation.py .scratch/hyph/mhyph.txt\n"
        " *\n"
        ' * Source: Grady Ward, "Moby Hyphenator II", Project Gutenberg etext #3204\n'
        " * (https://www.gutenberg.org/ebooks/3204, file mhyph.txt, sha256 " + digest[:16] + "…).\n"
        ' * Licence: public domain ("Public Domain material by grant from the author, January, 2001").\n'
        " * Ranked by frequency in the Brown corpus (Francis and Kučera 1979; NLTK's copy).\n"
        f" * Moby entries: {stats['entries']}; single lowercase a–z words: {stats['kept']}"
        f" ({stats.get('conflicts', 0)} listed twice with different points; the first kept).\n"
        f" * Kept: the {len(lines)} most frequent of 4–15 letters seen at least twice in Brown,"
        f" {points} hyphenation points.\n"
        " * One word per line, in rank order, with '-' at each hyphenation point.\n"
        " */\n\n"
        f"export const SOURCE_SHA256 = '{digest}'\n\n"
        f"export const WORDS = `{body}`\n"
    )
    print(f"{len(lines)} words, {points} points, {OUT.stat().st_size} bytes; moby {stats}")
    print("last kept:", ranked[-1], counts[ranked[-1]])


if __name__ == "__main__":
    main()
