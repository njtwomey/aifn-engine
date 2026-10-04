"""Golden values for aifn-compute/signal/multirate from scipy.signal: upfirdn over a grid of factors, resample_poly
(default
Kaiser window, an explicit window, given taps, and the statistic pad types), decimate (IIR and FIR, zero-phase and
causal), and the DFT analysis filter bank computed directly (each channel's modulated filter, then every M-th
sample)."""

import numpy as np
from scipy import signal


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    n = 97
    t = np.arange(n)
    x = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.21 * t) + 0.2 * rng.normal(size=n) + 1.5
    h = signal.firwin(15, 0.3)
    up = [
        {"up": u, "down": d, "y": signal.upfirdn(h, x, u, d)}
        for u, d in [(1, 1), (3, 1), (1, 4), (3, 2), (2, 5), (7, 3)]
    ]
    resample = [
        {"up": u, "down": d, "y": signal.resample_poly(x, u, d)}
        for u, d in [(2, 1), (1, 3), (3, 2), (2, 3), (6, 4), (5, 7)]
    ]
    taps = signal.firwin(31, 0.25, window="hann")
    variants = {
        "hamming": {"up": 3, "down": 2, "window": "hamming", "y": signal.resample_poly(x, 3, 2, window="hamming")},
        "taps": {"up": 1, "down": 4, "taps": taps, "y": signal.resample_poly(x, 1, 4, window=taps)},
        "mean": {"up": 2, "down": 3, "padtype": "mean", "y": signal.resample_poly(x, 2, 3, padtype="mean")},
        "median": {"up": 5, "down": 2, "padtype": "median", "y": signal.resample_poly(x, 5, 2, padtype="median")},
        "minimum": {"up": 3, "down": 4, "padtype": "minimum", "y": signal.resample_poly(x, 3, 4, padtype="minimum")},
        "maximum": {"up": 4, "down": 3, "padtype": "maximum", "y": signal.resample_poly(x, 4, 3, padtype="maximum")},
    }
    long = np.sin(2 * np.pi * 0.01 * np.arange(400)) + 0.3 * rng.normal(size=400)
    decimate = [
        {"q": q, "ftype": ft, "zeroPhase": zp, "y": signal.decimate(long, q, ftype=ft, zero_phase=zp)}
        for q in (2, 3, 5)
        for ft in ("iir", "fir")
        for zp in (True, False)
    ]
    decimate.append({"q": 4, "ftype": "iir", "n": 4, "zeroPhase": True, "y": signal.decimate(long, 4, n=4)})

    M = 4
    proto = np.asarray(signal.firwin(24, 1 / M))
    xs = rng.normal(size=61)
    frames = -(-(len(xs) + len(proto) - 1) // M)
    bank = np.zeros((M, frames), dtype=complex)
    for k in range(M):
        hk = proto * np.exp(2j * np.pi * k * np.arange(len(proto)) / M)
        bank[k] = np.convolve(xs, hk)[::M]
    poly = np.zeros((M, -(-len(proto) // M)))
    for k in range(M):
        e = proto[k::M]
        poly[k, : len(e)] = e
    return {
        "x": x,
        "h": h,
        "upfirdn": up,
        "resamplePoly": {"cases": resample, "variants": variants},
        "long": long,
        "decimateSignal": decimate,
        "dftFilterBank": {"x": xs, "prototype": proto, "channels": M, "re": np.real(bank), "im": np.imag(bank)},
        "polyphase": {"h": proto, "branches": M, "components": poly},
    }
