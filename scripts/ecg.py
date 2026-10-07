"""Vendor SciPy's electrocardiogram, with PhysioNet's beat annotations, into aifn-methods.

Usage (the source files are not kept in the repository; download them first):

    mkdir -p .scratch/ecg
    curl -sSL -o .scratch/ecg/ecg.dat https://raw.githubusercontent.com/scipy/dataset-ecg/main/ecg.dat
    curl -sSL -o .scratch/ecg/208.atr https://physionet.org/files/mitdb/1.0.0/208.atr
    curl -sSL -o .scratch/ecg/208.dat https://physionet.org/files/mitdb/1.0.0/208.dat   # optional: checks the offset
    uv run python scripts/ecg.py .scratch/ecg && npx prettier --write packages/methods/src/data/real/ecg/record.ts

Source: `scipy.datasets.electrocardiogram()`, five minutes of lead MLII of record 208 of the MIT-BIH Arrhythmia
Database (Moody and Mark, 2001) on PhysioNet (Goldberger et al., 2000), sampled at 360 Hz as raw 11-bit ADC values
(zero 1024, gain 200 per mV). SciPy's documentation places the excerpt at 19:35; it starts at sample 422820 of the
record (19:34.5), which this script checks against the record itself when `208.dat` is given. The reference beat
annotations of record 208 (`208.atr`, the cardiologists' labels) that fall in the excerpt are shifted onto it; the
non-beat annotations (rhythm changes, noise, artefacts) are dropped. The database is under the Open Data Commons
Attribution License v1.0.

Writes `packages/methods/src/data/real/ecg/record.ts`: the samples as base64 little-endian int16 (ADC value minus
1024), and the beat positions and symbols.
"""

import argparse
import base64
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "packages/methods/src/data/real/ecg/record.ts"
# The sha256 of ecg.dat in SciPy's dataset registry (scipy/datasets/_registry.py).
ECG_SHA256 = "f20ad3365fb9b7f845d0e5c48b6fe67081377ee466c3a220b7f69f35c8958baf"
START = 422820
ADC_ZERO = 1024
# MIT annotation codes of beats, with their symbols (WFDB's ecgcodes.h); every other code is not a beat.
BEATS = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E", 11: "j", 12: "/",
         13: "Q", 25: "B", 34: "e", 35: "n", 38: "f"}  # fmt: skip


def annotations(path: Path) -> list[tuple[int, int]]:
    """(sample, code) of every annotation in an MIT-format annotation file."""
    words = [int(w) for w in np.frombuffer(path.read_bytes(), "<u2")]
    out: list[tuple[int, int]] = []
    t = i = 0
    while i < len(words):
        code, value = words[i] >> 10, words[i] & 0x3FF
        if code == 0 and value == 0:
            break
        if code == 59:  # SKIP: a 32-bit interval follows, high word first
            t += (words[i + 1] << 16) | words[i + 2]
            i += 3
        elif code == 63:  # AUX: `value` bytes of text follow, padded to a whole word
            i += 1 + (value + 1) // 2
        elif code in (60, 61, 62):  # NUM, SUB, CHN: modifiers of the last annotation
            i += 1
        else:
            t += value
            out.append((t, code))
            i += 1
    return out


def mlii(path: Path) -> np.ndarray:
    """Lead MLII (the first of two signals) of a format-212 record file."""
    b = np.frombuffer(path.read_bytes(), np.uint8).astype(int)
    b = b[: len(b) // 3 * 3].reshape(-1, 3)
    s = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    return np.where(s > 2047, s - 4096, s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, help="the directory holding ecg.dat, 208.atr and optionally 208.dat")
    args = parser.parse_args()
    ecg_file = args.sources / "ecg.dat"
    digest = hashlib.sha256(ecg_file.read_bytes()).hexdigest()
    if digest != ECG_SHA256:
        raise SystemExit(f"ecg.dat has sha256 {digest}, not SciPy's {ECG_SHA256}")
    with np.load(ecg_file) as f:
        adc = f["ecg"].astype(int)
    n = len(adc)
    record = args.sources / "208.dat"
    if record.exists():
        if not np.array_equal(mlii(record)[START : START + n], adc):
            raise SystemExit(f"the excerpt is not lead MLII of record 208 from sample {START}")
        print(f"checked: the excerpt is samples {START}–{START + n - 1} of record 208, lead MLII")
    beats = [(s - START, BEATS[c]) for s, c in annotations(args.sources / "208.atr") if c in BEATS]
    beats = [(s, sym) for s, sym in beats if 0 <= s < n]
    counts = Counter(sym for _, sym in beats)
    centred = (adc - ADC_ZERO).astype("<i2")
    if not np.array_equal(centred.astype(int) + ADC_ZERO, adc):
        raise SystemExit("an ADC value does not fit in int16")
    signal = base64.b64encode(centred.tobytes()).decode("ascii")
    positions = ", ".join(str(s) for s, _ in beats)
    symbols = "".join(sym for _, sym in beats)
    tally = ", ".join(f"{sym} {k}" for sym, k in counts.most_common())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"""/**
 * The vendored record of `aifn-methods/data/real/ecg`: do not edit by hand; regenerate with
 *
 *   uv run python scripts/ecg.py .scratch/ecg
 *
 * Source: SciPy's `scipy.datasets.electrocardiogram()` (https://github.com/scipy/dataset-ecg, file ecg.dat, sha256
 * {ECG_SHA256[:16]}…): lead MLII of record 208 of the MIT-BIH Arrhythmia Database, samples {START} to
 * {START + n - 1} (19:34.5 to 24:34.5), and the reference beat annotations of record 208 (208.atr) in that span.
 * Licence: Open Data Commons Attribution License v1.0 (MIT-BIH Arrhythmia Database, PhysioNet).
 * Cite: Moody and Mark (2001), "The impact of the MIT-BIH Arrhythmia Database", IEEE Engineering in Medicine and
 * Biology 20(3); Goldberger et al. (2000), "PhysioBank, PhysioToolkit, and PhysioNet", Circulation 101(23).
 * {n} samples at 360 Hz; {len(beats)} annotated beats ({tally}).
 */

export const SOURCE_SHA256 = '{ECG_SHA256}'

/** The sampling rate, in Hz. */
export const SAMPLE_RATE = 360

/** The ADC gain: ADC units per mV. */
export const ADC_GAIN = 200

/** The first sample of the excerpt within record 208. */
export const RECORD_START = {START}

/** The samples as base64 little-endian int16, each the ADC value minus its zero ({ADC_ZERO}). */
export const SAMPLES =
  '{signal}'

/** The sample (within the excerpt) of each annotated beat, in order. */
export const BEAT_SAMPLES: readonly number[] = [{positions}]

/** The MIT-BIH symbol of each annotated beat, one character per beat. */
export const BEAT_SYMBOLS = '{symbols}'
"""
    )
    print(f"wrote {OUT.relative_to(Path.cwd())}: {n} samples, {len(beats)} beats ({tally}), {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
