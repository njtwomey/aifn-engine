/**
 * The constant-Q transform (Brown, 1991, "Calculation of a constant Q spectral transform", JASA 89(1)), computed from
 * one FFT per frame with spectral kernels (Brown and Puckette, 1992, "An efficient algorithm for the calculation of a
 * constant Q transform", JASA 92(5)).
 */

import { fft, nextPowerOfTwo } from 'aifn-compute/foundation/fourier'
import { fromData, imagPart, realPart, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { getWindow, type WindowSpec } from 'aifn-compute/signal/windows'
import { complexValues, readSamples, timeFrequency, type SignalInput, type TimeFrequency } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options for `cqt`. */
export type CqtOptions = {
  /** The lowest centre frequency $f_{\min}$, in Hz (in cycles per sample when the input has no sample rate). */
  fmin: Scalar
  /** Bins per octave $b$ (default 12: semitones). */
  binsPerOctave?: Size
  /** Number of bins $K$ (default: every bin whose centre lies below Nyquist). */
  bins?: Size
  /** Samples between frame centres (default: a quarter of the shortest window, at least 1). */
  hop?: Size
  /** The window $w_k$ of each bin, stretched to its length $N_k$ (default Hamming, as Brown). */
  window?: WindowSpec
  /**
   * Drop spectral-kernel entries below this fraction of each kernel's largest magnitude (Brown and Puckette's
   * sparsification, e.g. 0.0054). Default 0: the kernels are dense and the result equals the direct sum.
   */
  sparsity?: Scalar
  /** The sample rate, overriding the input's. */
  fs?: Scalar
}

/**
 * The constant-Q transform of a single-channel signal: for bins $k = 0, \dots, K - 1$ at
 * $f_k = f_{\min} 2^{k/b}$, with $Q = 1/(2^{1/b} - 1)$ and window length $N_k = \operatorname{round}(Q f_s / f_k)$,
 * $X[k, m] = \frac{1}{N_k} \sum_{n < N_k} w_k[n]\, x[c_m - \lfloor N_k/2 \rfloor + n]\, e^{-i 2\pi Q n / N_k}$, each
 * window centred on the frame centre $c_m = m \cdot \text{hop}$ ($x$ is 0 outside its samples). Every window spans
 * $Q$ cycles of its own frequency, so the bins have constant relative bandwidth. Computed per frame as
 * $\frac{1}{N} \sum_j X_m[j]\, K_k^*[j]$ (Parseval), with $X_m$ the $N$-point FFT of the frame ($N$ the power of two
 * $\ge N_0$) and $K_k$ the FFT of bin $k$'s kernel. Throws `DomainError` when $f_{\min}$ is not in $(0, f_s/2)$,
 * $b$ or $K$ is not a positive integer, or the top bin lies at or above Nyquist.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The lowest frequency (required), bins per octave, number of bins, hop, window, sparsity and sample
 *   rate; see `CqtOptions`.
 * @returns A complex `TimeFrequency` ($[K, \text{frames}]$, `method: 'cqt'`, `frequencyScale: 'log'`) with
 *   $\lceil n / \text{hop} \rceil$ frames (at least 1) at times $t_0 + c_m / f_s$, the bin frequencies $f_k$, and the
 *   window's name, longest length $N_0$ and hop.
 *
 * @example A 440 Hz tone
 * // Semitone bins from 220 Hz: bin 12 is 440 Hz, and holds (Hamming mean 0.54) / 2 of the unit amplitude.
 * const fs = 8000
 * const x = Array.from({ length: 2048 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / fs))
 * const C = cqt(x, { fmin: 220, bins: 25, fs })
 * print('shape =', C.values.shape, ' f[12] =', C.f.data[12], 'Hz')
 * const frames = C.t.shape[0]
 * const mag = complexAbs(C.values).data
 * const middle = Array.from({ length: 25 }, (_, k) => mag[k * frames + (frames >> 1)])
 * print('loudest bin =', middle.indexOf(Math.max(...middle)), ' |X| there =', Math.max(...middle))
 */
export function cqt(x: SignalInput, options: CqtOptions): TimeFrequency {
  const where = 'cqt'
  const s = readSamples(x, where, options.fs)
  const fs = s.fs
  const b = options.binsPerOctave ?? 12
  const fmin = options.fmin
  if (!(fmin > 0 && fmin < fs / 2)) throw new DomainError(where, `${where}: fmin must lie in (0, fs/2)`)
  if (!(Number.isInteger(b) && b >= 1))
    throw new DomainError(where, `${where}: binsPerOctave must be a positive integer`)
  const Q = 1 / (2 ** (1 / b) - 1)
  const K = options.bins ?? Math.ceil(b * Math.log2(fs / 2 / fmin))
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError(where, `${where}: bins must be a positive integer`)
  const freqs = Array.from({ length: K }, (_, k) => fmin * 2 ** (k / b))
  if (freqs[K - 1] >= fs / 2)
    throw new DomainError(where, `${where}: the top bin ${freqs[K - 1]} lies at or above Nyquist`)
  const lengths = freqs.map((f) => Math.round((Q * fs) / f))
  const N = nextPowerOfTwo(lengths[0])
  const hop = options.hop ?? Math.max(1, Math.floor(lengths[K - 1] / 4))
  const window = options.window ?? 'hamming'

  // Temporal kernels t_k[s_k + n] = w_k[n] e^{i2πQn/N_k}/N_k with s_k = N/2 − ⌊N_k/2⌋, so frame sample N/2 is c_m.
  const kre = new Float64Array(K * N)
  const kim = new Float64Array(K * N)
  for (let k = 0; k < K; k++) {
    const Nk = lengths[k]
    const w = toFlat(getWindow(window, Nk))
    const start = N / 2 - Math.floor(Nk / 2)
    for (let n = 0; n < Nk; n++) {
      const phase = (2 * Math.PI * Q * n) / Nk
      kre[k * N + start + n] = (w[n] * Math.cos(phase)) / Nk
      kim[k * N + start + n] = (w[n] * Math.sin(phase)) / Nk
    }
  }
  const spectral = fft(complexValues(kre, kim, [K, N]), { axis: -1 })
  const sre = toFlat(realPart(spectral))
  const sim = toFlat(imagPart(spectral))
  // Each kernel's support in frequency (all entries when sparsity is 0).
  const threshold = options.sparsity ?? 0
  const support = Array.from({ length: K }, (_, k) => {
    let peak = 0
    for (let j = 0; j < N; j++) peak = Math.max(peak, Math.hypot(sre[k * N + j], sim[k * N + j]))
    const keep: number[] = []
    for (let j = 0; j < N; j++) if (Math.hypot(sre[k * N + j], sim[k * N + j]) > threshold * peak) keep.push(j)
    return keep
  })

  const n = s.values.length
  const frames = Math.max(1, Math.ceil(n / hop))
  const segments = new Float64Array(frames * N)
  for (let m = 0; m < frames; m++) {
    const first = m * hop - N / 2
    for (let j = 0; j < N; j++) {
      const i = first + j
      if (i >= 0 && i < n) segments[m * N + j] = s.values[i]
    }
  }
  const X = fft(fromData(segments, [frames, N]), { axis: -1 })
  const xre = toFlat(realPart(X))
  const xim = toFlat(imagPart(X))
  // X[k, m] = (1/N) Σⱼ X_m[j] conj(K_k[j]).
  const ore = new Float64Array(K * frames)
  const oim = new Float64Array(K * frames)
  for (let k = 0; k < K; k++)
    for (let m = 0; m < frames; m++) {
      let re = 0
      let im = 0
      for (const j of support[k]) {
        const ar = xre[m * N + j]
        const ai = xim[m * N + j]
        const br = sre[k * N + j]
        const bi = sim[k * N + j]
        re += ar * br + ai * bi
        im += ai * br - ar * bi
      }
      ore[k * frames + m] = re / N
      oim[k * frames + m] = im / N
    }
  return timeFrequency({
    t: tensor(Array.from({ length: frames }, (_, m) => s.t0 + (m * hop) / fs)),
    f: tensor(freqs),
    values: complexValues(ore, oim, [K, frames]),
    quantity: 'complex',
    method: 'cqt',
    frequencyScale: 'log',
    window: { name: typeof window === 'string' ? window : window.name, length: lengths[0], hop },
  })
}
