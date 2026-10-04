"""Golden values for aifn-compute/signal/wavelets from PyWavelets: the filter banks of haar and db2–db10, single- and
multi-level periodic DWTs, and the cascade scaling and wavelet functions.

aifn's periodic DWT is pywt's 'periodization' mode up to two shifts: for dbN, a[k] = Σ_m h[m] x[(2k + m) mod n] equals
pywt's coefficients of x rolled by (N + 1) mod 2 samples, rolled by −⌊N/2⌋ coefficients. The cases record pywt's
coefficients mapped that way, level by level (each level transforms the previous level's approximation)."""

import numpy as np
import pywt

# pywt re-exports `Wavelet` from a compiled extension, which Pyright cannot see.
Wavelet = pywt.Wavelet  # pyright: ignore[reportAttributeAccessIssue]

NAMES = ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10"]


def order(name: str) -> int:
    return 1 if name == "haar" else int(name[2:])


def dwt_aifn(x: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One level in aifn's alignment, from pywt.dwt."""
    n = order(name)
    approx, detail = pywt.dwt(np.roll(x, (n + 1) % 2), name, mode="periodization")
    return np.roll(np.asarray(approx), -(n // 2)), np.roll(np.asarray(detail), -(n // 2))


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261002)
    filters = {}
    for name in NAMES:
        w = Wavelet(name)
        filters[name] = {"decLo": w.dec_lo, "decHi": w.dec_hi, "recLo": w.rec_lo, "recHi": w.rec_hi}
    single = []
    for name, n in [("haar", 16), ("db2", 16), ("db3", 32), ("db4", 32), ("db6", 64), ("db10", 64)]:
        x = rng.normal(size=n)
        a, d = dwt_aifn(x, name)
        single.append({"wavelet": name, "x": x, "approx": a, "detail": d})
    multi = []
    for name, n, levels in [("haar", 64, 4), ("db2", 64, 3), ("db4", 128, 3), ("db8", 256, 2)]:
        x = np.cumsum(rng.normal(size=n)) + np.sin(np.arange(n) / 5)
        a = x
        details = []
        for _ in range(levels):
            a, d = dwt_aifn(a, name)
            details.append(d)
        multi.append({"wavelet": name, "x": x, "levels": levels, "approx": a, "details": details})
    funs = []
    for name, level in [("db2", 6), ("db3", 5), ("db4", 4)]:
        phi, psi, t = Wavelet(name).wavefun(level=level)
        funs.append({"wavelet": name, "iterations": level, "t": t, "phi": phi, "psi": psi})
    return {"filters": filters, "dwt": single, "wavedec": multi, "wavefun": funs}
