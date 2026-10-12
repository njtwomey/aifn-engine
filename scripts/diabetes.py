"""Vendor the diabetes data of Efron, Hastie, Johnstone and Tibshirani (2004) into aifn-methods.

Usage (downloads the pinned source, or reuses it from .scratch/vendor; `--source FILE` reads a local copy instead):

    uv run python scripts/diabetes.py

Source: https://hastie.su.domains/Papers/LARS/diabetes.data, the raw table of "Least angle regression" (Annals of
Statistics 32(2), 2004): 442 patients, ten baseline measurements (age, sex, body mass index, mean blood pressure and
six blood serum measurements) and a quantitative measure of disease progression one year later. Tab-separated, one
header row, 18 496 bytes. scikit-learn ships the same values (`load_diabetes(scaled=False)`); its default copy is
pre-scaled, so the raw file is the source. The browser cannot fetch it at run time (the server sends no CORS
headers), so it is vendored.

Writes `packages/methods/src/data/real/diabetes-source.ts`: the file's lines verbatim, so that the module's test can
hash them back to the pinned sha256.
"""

import argparse
from pathlib import Path

from vendor import ROOT, Pin, fetch

PIN = Pin(
    url="https://hastie.su.domains/Papers/LARS/diabetes.data",
    size=18_496,
    sha256="4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd",
)
OUT = ROOT / "packages/methods/src/data/real/diabetes-source.ts"
HEADER = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]
ROWS = 442


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor the diabetes data of Efron et al. (2004).")
    parser.add_argument("--source", type=Path, help="a local copy of diabetes.data (else it is downloaded)")
    args = parser.parse_args()
    text = fetch(PIN, args.source).decode("ascii")
    if not text.endswith("\n") or "\r" in text:
        raise SystemExit("diabetes.data: expected LF line endings and a final newline")
    lines = text[:-1].split("\n")
    if lines[0].split("\t") != HEADER or len(lines) != ROWS + 1:
        raise SystemExit(f"diabetes.data: expected the header {HEADER} and {ROWS} rows")
    for i, line in enumerate(lines[1:], start=1):
        cells = line.split("\t")
        if len(cells) != len(HEADER) or any(not c or float(c) != float(c) for c in cells):
            raise SystemExit(f"diabetes.data: line {i + 1} is not {len(HEADER)} numbers")
    rows = "\n".join(f"  '{line}'," for line in lines).replace("\t", "\\t")
    OUT.write_text(
        f"""/**
 * The vendored source of `diabetes` (`aifn-methods/data/real`): do not edit by hand; regenerate with
 *
 *   uv run python scripts/diabetes.py
 *
 * Source: {PIN.url} ({PIN.size} bytes, sha256 {PIN.sha256[:16]}…),
 * the raw data of Efron, Hastie, Johnstone and Tibshirani (2004), "Least angle regression", Annals of Statistics
 * 32(2). Its lines verbatim, tab-separated: a header row and {ROWS} patients. Joined by newlines, with a final
 * newline, they are the file.
 */

/** The sha256 of the source file. */
export const DIABETES_SHA256 = '{PIN.sha256}'

/** The source's URL. */
export const DIABETES_URL = '{PIN.url}'

/** The lines of the source file, the header first; each is tab-separated. */
export const DIABETES_LINES: readonly string[] = [
{rows}
]
"""
    )
    print(f"wrote {OUT.relative_to(ROOT)}: {ROWS} rows, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
