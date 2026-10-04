"""Golden values for aifn-compute/signal/time-frequency's constant-Q transform by its direct definition (Brown, 1991):
for each
bin k at f_k = f_min 2^{k/b}, with Q = 1/(2^{1/b} − 1) and N_k = round(Q fs/f_k), the windowed sum
X[k, m] = (1/N_k) Σ_{n<N_k} w_k[n] x[m·hop − ⌊N_k/2⌋ + n] e^{−i2πQn/N_k}, with a symmetric Hamming (or Hann) window
from scipy.signal.get_window and x zero outside its samples. librosa is not a dependency, so the definition is the
reference."""

import numpy as np
from scipy import signal


def direct_cqt(x, fs, fmin, b, bins, hop, window):
    Q = 1 / (2 ** (1 / b) - 1)
    freqs = fmin * 2 ** (np.arange(bins) / b)
    n = len(x)
    frames = max(1, -(-n // hop))
    out = np.zeros((bins, frames), dtype=complex)
    for k, f in enumerate(freqs):
        Nk = int(np.round(Q * fs / f))
        w = signal.get_window(window, Nk, fftbins=False)
        kernel = w * np.exp(-2j * np.pi * Q * np.arange(Nk) / Nk) / Nk
        for m in range(frames):
            start = m * hop - Nk // 2
            idx = start + np.arange(Nk)
            seg = np.where((idx >= 0) & (idx < n), x[np.clip(idx, 0, n - 1)], 0.0)
            out[k, m] = seg @ kernel
    return freqs, out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    fs, n = 8000.0, 3000
    t = np.arange(n) / fs
    fmin = 110.0
    # Two tones on bin centres (A3 = 220 Hz is bin 12, E5 ≈ 659.26 Hz is bin 31) and a little noise.
    x = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * fmin * 2 ** (31 / 12) * t) + 0.05 * rng.normal(size=n)
    out = {}
    for name, (bpo, bins, hop, window) in {
        "semitones": (12, 48, 160, "hamming"),
        "thirdOctaves": (3, 14, 333, "hann"),
    }.items():
        freqs, X = direct_cqt(x, fs, fmin, bpo, bins, hop, window)
        out[name] = {
            "binsPerOctave": bpo,
            "bins": bins,
            "hop": hop,
            "window": window,
            "f": freqs,
            "re": np.real(X),
            "im": np.imag(X),
        }
    return {"x": x, "fs": fs, "fmin": fmin, "cqt": out}
