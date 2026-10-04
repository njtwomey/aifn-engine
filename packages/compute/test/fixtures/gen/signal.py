"""Golden values for aifn-compute/dsp, from numpy.fft, scipy.signal, scipy.fft and scipy.ndimage."""

from typing import cast

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal
from scipy.signal.windows import dpss

# scipy is unstubbed, and its filter designs declare a union of output forms (and None); these name the form asked for.
type Zpk = tuple[np.ndarray, np.ndarray, float]
type Ba = tuple[np.ndarray, np.ndarray]


def cx(z: np.ndarray) -> dict[str, object]:
    return {"re": np.real(z), "im": np.imag(z)}


def vmd_reference(f: np.ndarray, alpha: float, tau: float, K: int, DC: bool, tol: float):
    """The authors' VMD code as ported by vmdpy (Carvalho et al., 2020), init = 1 (uniform), in numpy."""
    if len(f) % 2:
        f = f[:-1]
    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))
    T = len(fMirr)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - (1 / T)
    Niter = 500
    Alpha = alpha * np.ones(K)
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: T // 2] = 0
    omega_plus = np.zeros([Niter, K])
    for i in range(K):
        omega_plus[0, i] = (0.5 / K) * i
    if DC:
        omega_plus[0, 0] = 0
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)
    uDiff = tol + np.spacing(1)
    n = 0
    sum_uk = 0
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)
    while uDiff > tol and n < Niter - 1:
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
            1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
        )
        if not DC:
            p = np.abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            omega_plus[n + 1, k] = np.dot(freqs[T // 2 : T], p) / np.sum(p)
        for k in np.arange(1, K):
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
            )
            p = np.abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            omega_plus[n + 1, k] = np.dot(freqs[T // 2 : T], p) / np.sum(p)
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)
        n = n + 1
        uDiff = np.spacing(1)
        for i in range(K):
            d = u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
            uDiff = uDiff + (1 / T) * np.dot(d, np.conj(d))
        uDiff = np.abs(uDiff)
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]
    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2 : T, :] = u_hat_plus[Niter - 1, T // 2 : T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2 : T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    u = u[:, T // 4 : 3 * T // 4]
    return u, omega, n


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    out: dict[str, object] = {}

    ffts = []
    for n in [1, 2, 7, 8, 13, 100, 128]:
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        ffts.append(
            {"x": cx(z), "fft": cx(np.fft.fft(z)), "ifft": cx(np.fft.ifft(z)), "fft_pad": cx(np.fft.fft(z, n + 5))}
        )
    out["fft"] = ffts
    real = rng.normal(size=9)
    out["rfft"] = {
        "x": real,
        "rfft": cx(np.fft.rfft(real)),
        "rfft16": cx(np.fft.rfft(real, 16)),
        "irfft": np.fft.irfft(np.fft.rfft(real), 9),
        "fftfreq": np.fft.fftfreq(9, 0.1),
        "rfftfreq": np.fft.rfftfreq(10, 0.1),
    }
    img = rng.normal(size=(5, 6))
    out["fft2"] = {"x": img, "fft2": cx(np.fft.fft2(img))}

    windows = {}
    specs: list[tuple[str, object]] = [
        ("hann", "hann"),
        ("hamming", "hamming"),
        ("blackman", "blackman"),
        ("blackmanharris", "blackmanharris"),
        ("nuttall", "nuttall"),
        ("flattop", "flattop"),
        ("bartlett", "bartlett"),
        ("triangular", "triang"),
        ("boxcar", "boxcar"),
        ("cosine", "cosine"),
        ("kaiser", ("kaiser", 8.0)),
        ("gaussian", ("gaussian", 2.5)),
        ("tukey", ("tukey", 0.5)),
    ]
    for name, spec in specs:
        for n in (10, 11):
            windows[f"{name}-{n}-sym"] = signal.get_window(spec, n, fftbins=False)
            windows[f"{name}-{n}-periodic"] = signal.get_window(spec, n, fftbins=True)
    out["windows"] = windows

    t = np.arange(1000) / 100.0
    x = np.sin(2 * np.pi * 7 * t) + 0.5 * rng.normal(size=t.size)
    f, p = signal.welch(x, fs=100.0, nperseg=128)
    fm, pm = signal.welch(x, fs=100.0, nperseg=100, noverlap=25, average="median", detrend="linear")
    fp, pp = signal.periodogram(x[:300], fs=100.0, window="hann", nfft=512)
    fs_, ts_, sxx = signal.spectrogram(x, fs=100.0, nperseg=64)
    fz, tz, zxx = signal.stft(x[:500], fs=100.0, nperseg=64)
    out["spectral"] = {
        "x": x,
        "welch": {"f": f, "psd": p},
        "welchMedian": {"f": fm, "psd": pm},
        "periodogram": {"f": fp, "psd": pp},
        "spectrogram": {"f": fs_, "t": ts_, "power": sxx},
        "stft": {"f": fz, "t": tz, "Z": cx(zxx)},
    }
    # istft: round trips under COLA and non-COLA (NOLA) windows, a two-sided case, and a modified (inconsistent) STFT.
    xi = x[:500]
    istft_cases = {}
    for key, window, nperseg, noverlap in [
        ("hann64", "hann", 64, 32),
        ("hamming50", "hamming", 50, 30),
        ("boxcar40", "boxcar", 40, 0),
        ("blackman33", "blackman", 33, 25),
    ]:
        _, tt, zz = signal.stft(xi, fs=100.0, window=window, nperseg=nperseg, noverlap=noverlap)
        _, xr = signal.istft(zz, fs=100.0, window=window, nperseg=nperseg, noverlap=noverlap)
        istft_cases[key] = {"window": window, "nperseg": nperseg, "noverlap": noverlap, "x": xr}
    _, _, z2 = signal.stft(xi, fs=100.0, nperseg=64, return_onesided=False)
    _, x2 = signal.istft(z2, fs=100.0, nperseg=64, input_onesided=False)
    mask = (np.abs(zxx) > np.median(np.abs(zxx))).astype(float)
    _, xm = signal.istft(zxx * mask, fs=100.0, nperseg=64)
    cola = {
        f"{w}-{n}-{o}": {
            "window": w,
            "nperseg": n,
            "noverlap": o,
            "cola": bool(signal.check_COLA(w, n, o)),
            "nola": bool(signal.check_NOLA(w, n, o)),
        }
        for w, n, o in [
            ("hann", 64, 32),
            ("hann", 64, 48),
            ("hamming", 50, 30),
            ("boxcar", 40, 0),
            ("boxcar", 40, 15),
            ("blackman", 33, 25),
            ("bartlett", 64, 32),
            ("hann", 4, 0),
        ]
    }
    out["istft"] = {
        "x": xi,
        "cases": istft_cases,
        "twoSided": {"Z": cx(z2), "x": cx(x2)},
        "masked": {"Z": cx(zxx * mask), "x": xm},
        "cola": cola,
    }
    # VMD: three tones (2, 24, 288 cycles per 1000 samples) plus a little noise, vmdpy's reference code.
    tv = np.arange(1000) / 1000.0
    xv = np.cos(2 * np.pi * 2 * tv) + 0.25 * np.cos(2 * np.pi * 24 * tv) + 1 / 16 * np.cos(2 * np.pi * 288 * tv)
    xv = xv + 0.01 * rng.normal(size=tv.size)
    vmd_out: dict[str, object] = {}
    for key, (alpha, tau, K, DC) in {"plain": (2000.0, 0.0, 3, False), "dc_tau": (1000.0, 0.1, 3, True)}.items():
        u, omega, iters = vmd_reference(xv, alpha, tau, K, DC, 1e-7)
        vmd_out[key] = {"alpha": alpha, "tau": tau, "K": K, "dc": DC, "u": u, "omega": omega[-1], "iterations": iters}
    out["vmd"] = {"x": xv, "cases": vmd_out}
    tapers, ratios = dpss(64, 3.0, 5, return_ratios=True)
    out["dpss"] = {"tapers": tapers, "ratios": ratios}

    out["firwin"] = {
        "lowpass": signal.firwin(31, 0.3),
        "highpass": signal.firwin(31, 0.3, pass_zero=False),
        # window is inferred as str from its default; a (name, parameter) tuple is documented.
        "bandpass": signal.firwin(41, [0.2, 0.5], pass_zero=False, window=("kaiser", 6.0)),  # pyright: ignore[reportArgumentType]
        "bandstop": signal.firwin(41, [0.2, 0.5]),
        "fs": signal.firwin(21, 10.0, fs=100.0, window="hann"),
        "kaiserord": list(signal.kaiserord(60.0, 0.05)),
    }
    iir = {}
    designs: dict[str, Zpk] = {
        "butter-low": cast(Zpk, signal.butter(4, 0.2, output="zpk")),
        "butter-high": cast(Zpk, signal.butter(3, 0.4, btype="highpass", output="zpk")),
        "butter-band": cast(Zpk, signal.butter(3, [0.2, 0.5], btype="bandpass", output="zpk")),
        "butter-stop": cast(Zpk, signal.butter(2, [0.2, 0.5], btype="bandstop", output="zpk")),
        "cheby1-low": cast(Zpk, signal.cheby1(4, 1.0, 0.3, output="zpk")),
        "cheby2-low": cast(Zpk, signal.cheby2(4, 40.0, 0.3, output="zpk")),
        "cheby2-odd": cast(Zpk, signal.cheby2(5, 30.0, 0.3, output="zpk")),
        "butter-fs": cast(Zpk, signal.butter(4, 10.0, fs=100.0, output="zpk")),
    }
    for key, (z, pl, k) in designs.items():
        b, a = signal.zpk2tf(z, pl, k)
        iir[key] = {"b": b, "a": a, "k": k}
    out["iir"] = iir

    b, a = cast(Ba, signal.butter(4, 0.2))
    zi = signal.lfilter_zi(b, a)
    y, zf = signal.lfilter(b, a, x[:200], zi=zi * x[0])
    w, h = signal.freqz(b, a, worN=64)  # pyright: ignore[reportArgumentType]  # a is inferred as int from its default
    wg, gd = signal.group_delay((b, a), w=64)
    out["filtering"] = {
        "b": b,
        "a": a,
        "x": x[:200],
        "lfilter": signal.lfilter(b, a, x[:200]),
        "zi": zi,
        "lfilterZi": {"y": y, "zf": zf},
        "filtfilt": signal.filtfilt(b, a, x[:200]),
        "filtfiltEven": signal.filtfilt(b, a, x[:200], padtype="even", padlen=20),
        "fir": signal.lfilter(signal.firwin(15, 0.3), [1.0], x[:50]),
        "freqz": {"w": w, "h": cx(h)},
        "groupDelay": {"w": wg, "delay": gd},
    }
    u = rng.normal(size=9)
    v = rng.normal(size=4)
    conv: dict[str, object] = {"u": u, "v": v}
    for mode in ("full", "same", "valid"):
        conv[f"convolve-{mode}"] = signal.convolve(u, v, mode=mode)
        conv[f"correlate-{mode}"] = signal.correlate(u, v, mode=mode)
        conv[f"lags-{mode}"] = signal.correlation_lags(9, 4, mode=mode)
        conv[f"lags-swap-{mode}"] = signal.correlation_lags(4, 9, mode=mode)
    long_u = rng.normal(size=300)
    long_v = rng.normal(size=80)
    conv["long"] = {"u": long_u, "v": long_v, "full": signal.convolve(long_u, long_v)}
    out["convolution"] = conv

    out["hilbert"] = {
        "even": {"x": x[:16], "z": cx(signal.hilbert(x[:16]))},
        "odd": {"x": x[:15], "z": cx(signal.hilbert(x[:15]))},
    }
    phase = np.cumsum(rng.uniform(0.5, 2.5, size=30))
    out["unwrap"] = {"wrapped": np.angle(np.exp(1j * phase)), "unwrapped": np.unwrap(np.angle(np.exp(1j * phase)))}
    m = rng.normal(size=(3, 8))
    out["dct"] = {"x": m, "dct": sfft.dct(m, norm="ortho", axis=-1)}

    image = rng.uniform(size=(12, 10))
    kernel = rng.normal(size=(3, 3))
    nd = {"image": image, "kernel": kernel}
    for mode in ("reflect", "mirror", "nearest", "constant", "wrap"):
        nd[f"correlate-{mode}"] = ndimage.correlate(image, kernel, mode=mode)
    nd["convolve"] = ndimage.convolve(image, kernel, mode="reflect")
    nd["gaussian"] = ndimage.gaussian_filter(image, 1.5)
    nd["sobelX"] = ndimage.sobel(image, axis=1)
    nd["sobelY"] = ndimage.sobel(image, axis=0)
    out["image"] = nd

    tt = np.linspace(0, 2, 50)
    out["chirp"] = {
        "t": tt,
        "linear": signal.chirp(tt, 1.0, 2.0, 6.0),
        "quadratic": signal.chirp(tt, 1.0, 2.0, 6.0, method="quadratic", vertex_zero=True),
        "logarithmic": signal.chirp(tt, 1.0, 2.0, 6.0, method="logarithmic"),
    }
    return out
