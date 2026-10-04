"""Golden values for aifn-compute/control from scipy: Riccati and Lyapunov equations, discretisation, tf/ss conversion,
Bode;
and from python-control: root loci and Nyquist encirclement counts."""

from collections.abc import Callable
from typing import Protocol, cast

import control as ct  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import scipy.linalg as sla
import scipy.signal as sig
from scipy.optimize import brentq


class PlacedPoles(Protocol):
    """The fields read from place_poles' Bunch (scipy is unstubbed)."""

    gain_matrix: np.ndarray
    nb_iter: int
    X: np.ndarray


def root(f: Callable[[float], float], lo: float, hi: float) -> float:
    """brentq's root; its declared return also covers the (root, RootResults) pair of full_output=True."""
    return cast(float, brentq(f, lo, hi))


def margins_reference(num: list[float], den: list[float]) -> dict[str, float]:
    """Gain and phase margins by root finding on |L(jw)| = 1 and Im L(jw) = 0 with Re L(jw) < 0."""

    def response(w: float) -> complex:
        s = 1j * w
        return complex(np.polyval(num, s) / np.polyval(den, s))

    ws = np.logspace(-3, 3, 20001)
    mags = np.array([abs(response(w)) for w in ws])
    ims = np.array([response(w).imag for w in ws])
    gc = [
        root(lambda w: abs(response(w)) - 1, ws[i], ws[i + 1])
        for i in range(len(ws) - 1)
        if (mags[i] - 1) * (mags[i + 1] - 1) < 0
    ]
    pc = [
        root(lambda w: response(w).imag, ws[i], ws[i + 1])
        for i in range(len(ws) - 1)
        if ims[i] * ims[i + 1] < 0 and response(ws[i]).real < 0
    ]
    w_gc = gc[0]
    w_pc = pc[0]
    return {
        "gainCrossover": w_gc,
        "phaseMargin": 180 + float(np.degrees(np.angle(response(w_gc)))),
        "phaseCrossover": w_pc,
        "gainMargin": 1 / abs(response(w_pc)),
    }


def place_cases() -> dict[str, object]:
    """scipy place_poles: KNV0 on real poles (step-for-step reference) and YT for robustness comparisons."""
    rng = np.random.default_rng(11)
    out = {}
    # scipy's docstring example (Kautsky et al.'s test system).
    a1 = np.array(
        [
            [1.380, -0.2077, 6.715, -5.676],
            [-0.5814, -4.290, 0, 0.6750],
            [1.067, 4.273, -6.654, 5.893],
            [0.0480, 4.273, 1.343, -2.104],
        ]
    )
    b1 = np.array([[0, 5.679], [1.136, 1.136], [0, 0], [-3.146, 0]])
    a2 = rng.normal(size=(6, 6))
    b2 = rng.normal(size=(6, 3))
    systems = {
        "kautsky": (a1, b1, [-0.2, -0.5, -5.0566, -8.6659], [-0.5 + 1j, -0.5 - 1j, -5.0566, -8.6659]),
        "random6x3": (a2, b2, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [-1 + 2j, -1 - 2j, -2 + 1j, -2 - 1j, -3.0, -4.0]),
    }
    for key, (a, b, real_poles, complex_poles) in systems.items():
        knv = cast(PlacedPoles, sig.place_poles(a, b, real_poles, method="KNV0"))
        yt_real = cast(PlacedPoles, sig.place_poles(a, b, real_poles, method="YT"))
        yt = cast(PlacedPoles, sig.place_poles(a, b, complex_poles, method="YT"))
        out[key] = {
            "A": a,
            "B": b,
            "real": real_poles,
            "complexRe": np.real(complex_poles),
            "complexIm": np.imag(complex_poles),
            "K": knv.gain_matrix,
            "iterations": knv.nb_iter,
            "condKnv": np.linalg.cond(knv.X),
            "condYtReal": np.linalg.cond(yt_real.X),
            "condYt": np.linalg.cond(yt.X),
        }
    return out


def locus_case(num: list[float], den: list[float], gains: list[float]) -> dict[str, object]:
    """Closed-loop poles at each gain, sorted per gain (branch order is not compared)."""
    data = ct.root_locus_map(ct.tf(num, den), gains)  # pyright: ignore[reportUnknownMemberType]
    loci = np.asarray(data.loci)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    rows = [np.sort_complex(row) for row in loci]
    return {
        "num": num,
        "den": den,
        "gains": gains,
        "re": [r.real for r in rows],
        "im": [r.imag for r in rows],
    }


def nyquist_case(num: list[float], den: list[float], dt: float | None = None) -> dict[str, object]:
    """python-control's encirclement count N and the open-loop unstable poles P."""
    sys = ct.tf(num, den, dt) if dt else ct.tf(num, den)  # pyright: ignore[reportUnknownMemberType]
    resp = ct.nyquist_response(sys)  # pyright: ignore[reportUnknownMemberType]
    p = np.asarray(sys.poles())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportOptionalMemberAccess]
    unstable = int(np.sum(np.abs(p) > 1 + 1e-9)) if dt else int(np.sum(p.real > 1e-9))  # pyright: ignore[reportAttributeAccessIssue]
    return {"num": num, "den": den, "dt": dt, "count": int(resp.count), "P": unstable}  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def criteria_cases() -> dict[str, object]:
    """Root loci and Nyquist encirclement counts from python-control."""
    return {
        "rootLocus": [
            locus_case([1.0], [1.0, 3.0, 2.0, 0.0], [0.0, 0.1, 0.5, 1.0, 6.0, 20.0, 100.0]),
            locus_case([1.0, 2.0], [1.0, 2.0, 5.0, 0.0], [0.0, 0.3, 1.0, 4.0, 10.0]),
            locus_case([1.0, 1.0], [1.0, -1.0, 0.0], [0.0, 0.5, 1.0, 2.0, 8.0]),
        ],
        "nyquist": [
            nyquist_case([1.0], [1.0, 3.0, 2.0, 0.0]),
            nyquist_case([20.0], [1.0, 3.0, 2.0, 0.0]),
            nyquist_case([2.0, 2.0], [1.0, -1.0]),
            nyquist_case([3.0], [1.0, 2.0, 2.0, 1.0]),
            nyquist_case([10.0], [1.0, 2.0, 2.0, 1.0]),
            nyquist_case([2.0, 1.0], [1.0, 1.0, 0.0, 0.0]),
            nyquist_case([0.5], [1.0, -1.2, 0.35], 0.1),
            nyquist_case([2.0], [1.0, -1.2, 0.35], 0.1),
        ],
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3))
    b = rng.normal(size=(3, 2))
    m = rng.normal(size=(3, 3))
    q = m @ m.T + np.eye(3)
    r = np.array([[2.0, 0.3], [0.3, 1.0]])

    # Linearised cart-pole (states x, x', theta, theta'; open-loop unstable).
    cp_a = np.array([[0, 1, 0, 0], [0, 0, -0.98, 0], [0, 0, 0, 1], [0, 0, 21.56, 0]], dtype=float)
    cp_b = np.array([[0], [1], [0], [-2]], dtype=float)
    cp_q = np.diag([1.0, 0.1, 10.0, 0.1])
    cp_r = np.array([[0.5]])

    ad = a / (1.2 * max(abs(np.linalg.eigvals(a))))  # a stable discrete A

    stable = a - (max(np.linalg.eigvals(a).real) + 0.5) * np.eye(3)
    c = rng.normal(size=(2, 3))
    d = rng.normal(size=(2, 2))
    # A state-space system discretises to (A, B, C, D, dt).
    zoh = cast(tuple[np.ndarray, ...], sig.cont2discrete((a, b, c, d), 0.1, method="zoh"))
    tustin = cast(tuple[np.ndarray, ...], sig.cont2discrete((a, b, c, d), 0.1, method="bilinear"))

    num = [1.0, 3.0]
    den = [1.0, 2.0, 3.0, 4.0]
    A_tf, B_tf, C_tf, D_tf = sig.tf2ss(num, den)
    ss_num, ss_den = sig.ss2tf(cp_a, cp_b, np.array([[1.0, 0, 0, 0]]), np.array([[0.0]]))

    w = np.logspace(-2, 2, 50)
    # Complex responses and conversions for LtiSystem on complex128.
    fr_num, fr_den = [1.0, 0.5, 2.0], [1.0, 1.2, 3.0, 1.5]
    fr_w = np.logspace(-1, 1, 7)
    _, fr_h = sig.freqresp(sig.lti(fr_num, fr_den), fr_w)
    z, p, k = sig.tf2zpk(fr_num, fr_den)
    # Distinct zeros: a repeated root is ill-conditioned. ellip's declared return is a union of output forms.
    d_b, d_a = cast(tuple[np.ndarray, np.ndarray], sig.ellip(4, 1, 40, 0.3))
    d_w, d_h = sig.freqz(d_b, d_a, worN=9)  # pyright: ignore[reportArgumentType]  # a is inferred as int from its default
    d_z, d_p, d_k = sig.tf2zpk(d_b, d_a)
    d_sos = sig.zpk2sos(d_z, d_p, d_k, pairing="nearest")
    _, mag, phase = sig.bode(sig.lti([2.0, 1.0], [1.0, 0.5, 4.0, 0.0]), w)

    return {
        "care": {"A": a, "B": b, "Q": q, "R": r, "P": sla.solve_continuous_are(a, b, q, r)},
        "cartpole": {"A": cp_a, "B": cp_b, "Q": cp_q, "R": cp_r, "P": sla.solve_continuous_are(cp_a, cp_b, cp_q, cp_r)},
        "dare": {"A": a, "B": b, "Q": q, "R": r, "P": sla.solve_discrete_are(a, b, q, r)},
        "dare_stable": {"A": ad, "B": b, "Q": q, "R": r, "P": sla.solve_discrete_are(ad, b, q, r)},
        "lyapunov": {
            "A": stable,
            "Q": q,
            "X": sla.solve_continuous_lyapunov(stable, -q),
            "Ad": ad,
            "Xd": sla.solve_discrete_lyapunov(ad, q),
        },
        "discretise": {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "dt": 0.1,
            "zoh": {"A": zoh[0], "B": zoh[1], "C": zoh[2], "D": zoh[3]},
            "tustin": {"A": tustin[0], "B": tustin[1], "C": tustin[2], "D": tustin[3]},
        },
        "tf2ss": {"num": num, "den": den, "A": A_tf, "B": B_tf, "C": C_tf, "D": D_tf},
        "ss2tf": {"num": ss_num[0], "den": ss_den},
        "bode": {"num": [2.0, 1.0], "den": [1.0, 0.5, 4.0, 0.0], "w": w, "mag": mag, "phase": phase},
        "freqresp": {"num": fr_num, "den": fr_den, "w": fr_w, "re": fr_h.real, "im": fr_h.imag},
        "tf2zpk": {
            "num": fr_num,
            "den": fr_den,
            "zeros": {"re": np.sort_complex(z).real, "im": np.sort_complex(z).imag},
            "poles": {"re": np.sort_complex(p).real, "im": np.sort_complex(p).imag},
            "gain": k,
        },
        "discrete": {"b": d_b, "a": d_a, "w": d_w, "re": d_h.real, "im": d_h.imag, "sos": d_sos},
        "margins": {"num": [2.0], "den": [1.0, 3.0, 2.0, 0.0], **margins_reference([2.0], [1.0, 3.0, 2.0, 0.0])},
        "place": place_cases(),
        **criteria_cases(),
    }
