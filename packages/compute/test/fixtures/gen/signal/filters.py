"""Golden values for the elliptic, Bessel and Parks–McClellan designs and the smoothing filters of
aifn-compute/signal/filters
from scipy.signal (`ellip`, `bessel`, `remez`, `savgol_filter`, `savgol_coeffs`, `medfilt`, `wiener`), and the Jacobi
elliptic functions from scipy.special (`ellipj`, `ellipkm1`) that the elliptic prototype uses. The earlier
Butterworth, Chebyshev and FIR window cases live in `signal.json`."""

from typing import cast

import numpy as np
from scipy import signal, special

type ZPK = tuple[np.ndarray, np.ndarray, float]
type BA = tuple[np.ndarray, np.ndarray]


def cx(z: np.ndarray) -> dict[str, object]:
    z = np.sort_complex(np.asarray(z, dtype=complex))
    return {"re": np.real(z), "im": np.imag(z)}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    x = np.sin(np.arange(40) * 0.7) + 0.1 * np.arange(40) ** 1.2 + 0.2 * rng.normal(size=40)
    ellip = []
    for n, rp, rs, wn, btype in [
        (4, 1.0, 40.0, 0.3, "lowpass"),
        (5, 0.5, 60.0, 0.4, "lowpass"),
        (3, 2.0, 30.0, 0.6, "highpass"),
        (4, 0.5, 50.0, [0.2, 0.4], "bandpass"),
        (2, 1.0, 35.0, [0.3, 0.5], "bandstop"),
    ]:
        z, p, k = cast(ZPK, signal.ellip(n, rp, rs, wn, btype=btype, output="zpk"))
        ellip.append({"n": n, "rp": rp, "rs": rs, "wn": wn, "btype": btype, "z": cx(z), "p": cx(p), "k": k})
    bessel = []
    for n, wn, btype in [(1, 0.3, "lowpass"), (4, 0.2, "lowpass"), (7, 0.45, "highpass"), (3, [0.2, 0.5], "bandpass")]:
        b, a = cast(BA, signal.bessel(n, wn, btype=btype))
        bessel.append({"n": n, "wn": wn, "btype": btype, "b": b, "a": a})
    remez = []
    for numtaps, bands, desired, weight in [
        (31, [0, 0.2, 0.25, 0.5], [1, 0], None),
        (24, [0, 0.15, 0.25, 0.5], [1, 0], [1, 10]),
        (41, [0, 0.1, 0.15, 0.3, 0.35, 0.5], [0, 1, 0], None),
        (33, [0, 0.2, 0.3, 0.5], [0, 1], None),
    ]:
        h = signal.remez(numtaps, bands, desired, weight=weight)
        remez.append({"numtaps": numtaps, "bands": bands, "desired": desired, "weight": weight, "h": h})
    savgol = [
        {"window": w, "order": o, "deriv": d, "mode": m, "y": signal.savgol_filter(x, w, o, deriv=d, mode=m)}
        for w, o, d, m in [
            (7, 2, 0, "interp"),
            (11, 4, 0, "interp"),
            (7, 3, 1, "mirror"),
            (9, 2, 2, "nearest"),
            (5, 1, 0, "wrap"),
            (7, 3, 1, "interp"),
        ]
    ]
    coeffs = [
        {"window": w, "order": o, "deriv": d, "h": signal.savgol_coeffs(w, o, deriv=d, use="dot")}
        for w, o, d in [(5, 2, 0), (9, 3, 1), (11, 4, 2)]
    ]
    medfilt = [{"k": k, "y": signal.medfilt(x, k)} for k in [3, 5, 9]]
    wiener = [
        {"size": s, "noise": nz, "y": signal.wiener(x, s, noise=nz)} for s, nz in [(3, None), (5, None), (7, 0.3)]
    ]
    jac = []
    for u, m in [(0.7, 0.4), (1.9, 0.93), (0.3, 1e-11), (2.5, 1 - 1e-11), (4.0, 0.999)]:
        sn, cn, dn, ph = special.ellipj(u, m)
        jac.append({"u": u, "m": m, "sn": sn, "cn": cn, "dn": dn, "ph": ph})
    km1 = [{"p": p, "y": special.ellipkm1(p)} for p in [1e-15, 1e-8, 1e-3, 0.4]]
    return {
        "x": x,
        "ellip": ellip,
        "bessel": bessel,
        "remez": remez,
        "savgol": savgol,
        "savgolCoeffs": coeffs,
        "medfilt": medfilt,
        "wiener": wiener,
        "ellipj": jac,
        "ellipkm1": km1,
    }
