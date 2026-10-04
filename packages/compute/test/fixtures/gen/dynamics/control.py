"""Golden values for aifn-compute/dynamics/control's LQG design from python-control: the LQR gain and the Kalman
gain."""

import control as ct  # pyright: ignore[reportMissingTypeStubs]
import numpy as np


def cases() -> dict[str, object]:
    # A lightly damped oscillator with a measured position (continuous) and its zero-order-hold sampling (discrete).
    a = np.array([[0.0, 1.0], [-2.0, -0.3]])
    b = np.array([[0.0], [1.0]])
    c = np.array([[1.0, 0.0]])
    q = np.diag([4.0, 1.0])
    r = np.array([[0.5]])
    w = np.diag([0.1, 0.4])
    v = np.array([[0.05]])
    k, _, _ = ct.lqr(a, b, q, r)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    gain, s, _ = ct.lqe(a, np.eye(2), c, w, v)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sysd = ct.c2d(ct.ss(a, b, c, 0), 0.1)  # pyright: ignore[reportUnknownMemberType]
    ad = np.asarray(sysd.A)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    bd = np.asarray(sysd.B)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    kd, _, _ = ct.dlqr(ad, bd, q, r)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # dlqe returns the predictor gain L (x̂[n+1] = Ax̂[n] + Bu[n] + L(y[n] − Cx̂[n])) and the prior covariance P.
    md, pd, _ = ct.dlqe(ad, np.eye(2), c, w, v)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return {
        "lqg": {
            "A": a,
            "B": b,
            "C": c,
            "Q": q,
            "R": r,
            "W": w,
            "V": v,
            "K": np.asarray(k),
            "L": np.asarray(gain),
            "S": np.asarray(s),
            "Ad": ad,
            "Bd": bd,
            "Kd": np.asarray(kd),
            "Ld": np.asarray(md),
            "Sd": np.asarray(pd),
        }
    }
