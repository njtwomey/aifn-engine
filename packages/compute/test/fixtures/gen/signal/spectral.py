"""Golden values for aifn-compute/signal/spectral's estimators and aifn-compute/signal/statistical's parametric spectra:
scipy.signal's
welch (Welch and Bartlett settings), csd, coherence and dpss; astropy's LombScargle (classic, floating-mean and
generalised; Baluev false-alarm probability and level); statsmodels' Burg and Yule-Walker AR fits; scipy.stats' chi2
quantiles; the ARMA spectrum by scipy.signal.freqz; and numpy ports where no library has the estimator: the
Blackman-Tukey sum, Thomson's adaptive multitaper weights (Percival and Walden, 1993, eqs. 368a and 370a, as nitime's
adaptive_weights) and the modified covariance AR fit."""

from typing import cast

import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal, stats
from scipy.signal import windows
from statsmodels.regression.linear_model import burg, yule_walker


def series(n: int, rng: np.random.Generator) -> np.ndarray:
    """An AR(2) (poles 0.9 e^{±2πi 0.1}) plus a tone at 0.31 cycles per sample."""
    a = [1.0, -2 * 0.9 * np.cos(2 * np.pi * 0.1), 0.81]
    e = rng.normal(size=n + 500)
    x = signal.lfilter([1.0], a, e)[500:]
    return x + 0.8 * np.sin(2 * np.pi * 0.31 * np.arange(n) + 0.4)


def blackman_tukey(x: np.ndarray, M: int, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    xc = x - x.mean()
    g = np.array([xc[: n - k] @ xc[k:] / n for k in range(M + 1)])
    w = windows.bartlett(2 * M + 1)[M:]
    f = np.fft.rfftfreq(nfft, 1 / fs)
    k = np.arange(1, M + 1)
    S = np.array([w[0] * g[0] + 2 * np.sum(w[1:] * g[1:] * np.cos(2 * np.pi * fk * k / fs)) for fk in f]) / fs
    S[1:] *= 2
    if nfft % 2 == 0:
        S[-1] /= 2
    return f, S


def adaptive_multitaper(x: np.ndarray, NW: float, K: int, fs: float) -> dict[str, object]:
    n = len(x)
    xc = x - x.mean()
    tapers, ratios = cast(tuple[np.ndarray, np.ndarray], windows.dpss(n, NW, K, return_ratios=True))
    Sk = np.abs(np.fft.rfft(tapers * xc, axis=1)) ** 2 / fs
    sig2 = xc @ xc / n
    B = sig2 / fs
    S = (Sk[0] + Sk[1]) / 2
    for _ in range(150):
        d = np.sqrt(ratios)[:, None] * S / (ratios[:, None] * S + (1 - ratios)[:, None] * B)
        new = (d**2 * Sk).sum(0) / (d**2).sum(0)
        done = np.all(np.abs(new - S) <= 1e-10 * new)
        S = new
        if done:
            break
    d = np.sqrt(ratios)[:, None] * S / (ratios[:, None] * S + (1 - ratios)[:, None] * B)
    nu = 2 * (d**2).sum(0) ** 2 / (d**4).sum(0)
    plain = Sk.mean(0)
    for arr in (S, plain):
        arr[1:] *= 2
        if n % 2 == 0:
            arr[-1] /= 2
    return {"adaptive": S, "plain": plain, "dof": nu, "ratios": ratios}


def modified_covariance(x: np.ndarray, p: int) -> dict[str, object]:
    c = x - x.mean()
    n = len(c)
    rows, rhs = [], []
    for t in range(p, n):
        rows.append([c[t - i] for i in range(1, p + 1)])
        rhs.append(c[t])
    for t in range(n - p):
        rows.append([c[t + i] for i in range(1, p + 1)])
        rhs.append(c[t])
    A, b = np.array(rows), np.array(rhs)
    phi = np.linalg.lstsq(A, b, rcond=None)[0]
    return {"ar": phi, "sigma2": float(np.sum((A @ phi - b) ** 2) / len(b))}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    n = 600
    x = series(n, rng)
    # y = delayed low-pass of x plus noise.
    y = signal.lfilter([0, 0, 0, 0.25, 0.5, 0.25], [1.0], x) + 0.7 * rng.normal(size=n)
    fs = 2.0

    f, Pw = signal.welch(x, fs=fs, nperseg=128, noverlap=64, window="hann")
    _, Pb = signal.welch(x, fs=fs, nperseg=100, noverlap=0, window="boxcar")
    _, Pp = signal.periodogram(x, fs=fs)
    fc, Pxy = signal.csd(x, y, fs=fs, nperseg=128)
    _, Cxy = signal.coherence(x, y, fs=fs, nperseg=128)
    fbt, Sbt = blackman_tukey(x, 30, fs, n)
    mt = adaptive_multitaper(x, 4.0, 7, fs)

    # Uneven sampling: random times, a sinusoid at 0.37 plus noise with per-sample sds.
    t = np.sort(rng.uniform(0, 100, 150))
    dy = 0.5 * 4 ** (rng.uniform(size=150) - 0.5)
    yu = 1.3 + np.sin(2 * np.pi * 0.37 * t + 1.0) + dy * rng.normal(size=150)
    freq = np.linspace(0.01, 1.5, 300)
    classic = LombScargle(t, yu, fit_mean=False, center_data=True).power(freq, normalization="standard")
    floating = LombScargle(t, yu).power(freq, normalization="standard")
    generalised = LombScargle(t, yu, dy).power(freq, normalization="standard")
    psd = LombScargle(t, yu).power(freq, normalization="psd")
    ls = LombScargle(t, yu)
    levels = [0.05, 0.2, 0.35, 0.5]
    fap = ls.false_alarm_probability(np.array(levels), method="baluev", maximum_frequency=1.5)
    level = ls.false_alarm_level([0.01, 0.1], method="baluev", maximum_frequency=1.5)
    window = np.abs(np.exp(-2j * np.pi * np.outer(freq, t)).sum(1)) ** 2 / len(t) ** 2

    # AR fits and spectra.
    rho_b, sigma2_b = burg(x, order=4, demean=True)
    rho_yw, sigma_yw = cast(
        tuple[np.ndarray, float], yule_walker(x, order=4, method="mle", demean=True, result_object=False)
    )
    lsq = modified_covariance(x, 4)
    ar = [1.2, -0.6]
    ma = [0.4, 0.2]
    # scipy is unstubbed, so Pyright infers freqz's `a` as int from its default `a=1`.
    w, h = signal.freqz([1.0, *ma], [1.0, *(-np.array(ar))], worN=65, fs=fs, include_nyquist=True)  # pyright: ignore[reportArgumentType]
    arma_psd = 2 * 1.5 * np.abs(h) ** 2 / fs
    arma_psd[0] /= 2
    arma_psd[-1] /= 2

    return {
        "x": x,
        "y": y,
        "fs": fs,
        "welch": {"f": f, "psd": Pw},
        "bartlett": Pb,
        "periodogram": Pp,
        "csd": {"f": fc, "re": np.real(Pxy), "im": np.imag(Pxy)},
        "coherence": Cxy,
        "blackmanTukey": {"f": fbt, "psd": Sbt},
        "multitaper": mt,
        "chi2": {
            "nu": [2.0, 7.5, 30.0],
            "lo": [stats.chi2.ppf(0.025, v) for v in (2.0, 7.5, 30.0)],
            "hi": [stats.chi2.ppf(0.975, v) for v in (2.0, 7.5, 30.0)],
        },
        "uneven": {
            "t": t,
            "y": yu,
            "dy": dy,
            "f": freq,
            "classic": classic,
            "floating": floating,
            "generalised": generalised,
            "psd": psd,
            "levels": levels,
            "fap": fap,
            "fapTargets": [0.01, 0.1],
            "fapLevel": level,
            "window": window,
        },
        "ar": {
            "burg": {"ar": rho_b, "sigma2": float(sigma2_b)},
            "yuleWalker": {"ar": rho_yw, "sigma2": float(sigma_yw**2)},
            "leastSquares": lsq,
            "arma": {"ar": ar, "ma": ma, "sigma2": 1.5, "f": w, "psd": arma_psd},
        },
    }
