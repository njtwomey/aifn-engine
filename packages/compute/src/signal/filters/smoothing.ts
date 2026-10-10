/**
 * Smoothing and detection filters, as scipy.signal: the Savitzky–Golay filter (local least-squares polynomials;
 * Savitzky and Golay, 1964, Anal. Chem. 36(8)), the running median (Tukey, 1977), the local adaptive Wiener filter
 * (`scipy.signal.wiener`; Lim, 1990, "Two-Dimensional Signal and Image Processing", §9.2), a frequency-domain Wiener
 * shrinkage for additive white noise of known variance (Wiener, 1949), and the matched filter (Turin, 1960, IRE Trans.
 * Inf. Theory 6(3)): correlation with the template, which maximises the output SNR in white noise.
 *
 * Each takes a single-channel `Signal` or bare samples (at rate 1) and returns a `Signal` with the input's rate, start
 * time and unit. They run on concrete samples and are not differentiable.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Signal, Size } from 'aifn-compute/foundation/contracts'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import { solve } from 'aifn-compute/numerics/linalg'
import { readSamples, signal, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A `Signal` on the input's time axis holding `values`.
 *
 * @param input The input's rate, start time and unit, as `readSamples` returns them.
 * @param values The output samples, kept (not copied).
 * @returns The `Signal`.
 */
function like(input: { fs: Scalar; t0: Scalar; unit?: string }, values: Float64Array): Signal {
  return signal(fromData(values, [values.length]), {
    fs: input.fs,
    t0: input.t0,
    ...(input.unit !== undefined ? { unit: input.unit } : {}),
  })
}

/**
 * Throws `DomainError` unless `k` is a positive odd integer (a window with a centre sample).
 *
 * @param k The window length.
 * @param where The caller's name, for error messages.
 */
const oddSize = (k: Size, where: string) => {
  if (!(Number.isInteger(k) && k >= 1 && k % 2 === 1))
    throw new DomainError(where, `${where}: the window length must be odd`)
}

// ── Savitzky–Golay ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Savitzky–Golay convolution weights, as `scipy.signal.savgol_coeffs(window, polyorder, deriv, delta,
 * use='dot')`: weight $i$ ($i = 0, \dots, w - 1$ for a window of $w$, centred at $(w - 1)/2$) gives the $d$-th
 * derivative at the centre of the least-squares polynomial of degree `polyorder` through the window, $d!/\Delta^d$
 * times row $d$ of $(\Vmat^\top\Vmat)^{-1}\Vmat^\top$ with $\Vmat$ the Vandermonde matrix of the offsets. Throws
 * `DomainError` for an even window or a `polyorder` not below it.
 *
 * @param window The window length $w$, a positive odd integer.
 * @param polyorder The degree of the fitted polynomial, from 0 to $w - 1$.
 * @param options Which derivative, and the sample spacing.
 * @param options.deriv The order $d$ of the derivative (0, the default, smooths); above `polyorder` the weights are 0.
 * @param options.delta The sample spacing $\Delta$, which scales a derivative by $\Delta^{-d}$ (default 1).
 * @returns The $w$ weights, applied as a dot product with the window's samples in time order.
 *
 * @example A 5-point quadratic smoother, against scipy.signal.savgol_coeffs
 * // scipy: [-3, 12, 17, 12, -3] / 35.
 * print('smooth =', savgolCoeffs(5, 2))
 * print('first derivative =', savgolCoeffs(5, 2, { deriv: 1 }))
 */
export function savgolCoeffs(
  window: Size,
  polyorder: Size,
  { deriv = 0, delta = 1 }: { deriv?: Size; delta?: Scalar } = {},
): Tensor {
  oddSize(window, 'savgolCoeffs')
  if (!(polyorder >= 0 && polyorder < window))
    throw new DomainError('savgolCoeffs', 'savgolCoeffs: polyorder must be below the window')
  const half = (window - 1) / 2
  return fromData(polyWeights(window, polyorder, deriv, half, delta), [window])
}

/**
 * Weights that evaluate the $d$-th derivative at offset `at` (from the window start) of the window's least-squares
 * polynomial: $w_i = \sum_k g_k u_i^k / \Delta^d$ with $u_i = i - (w - 1)/2$ and $(\Vmat^\top\Vmat)\gvec = \evec$,
 * where $\evec$ holds the $d$-th derivative of each monomial $u^k$ at the offset.
 *
 * @param window The window length $w$.
 * @param order The polynomial degree.
 * @param deriv The derivative order $d$; above `order` the weights are all 0.
 * @param at Where to evaluate, in samples from the window's first (the centre is $(w - 1)/2$).
 * @param delta The sample spacing $\Delta$.
 * @returns The $w$ weights, for a dot product with the window's samples.
 */
function polyWeights(window: Size, order: Size, deriv: Size, at: number, delta: Scalar): Float64Array {
  if (deriv > order) return new Float64Array(window)
  const p = order + 1
  const half = (window - 1) / 2
  // Normal equations in the centred offsets u = i − half (well conditioned for the small orders used).
  const VtV = new Float64Array(p * p)
  for (let i = 0; i < window; i++) {
    const u = i - half
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) VtV[a * p + b] += u ** (a + b)
  }
  // The d-th derivative at u₀ = at − half of Σ cₖ uᵏ: Σ_{k ≥ d} cₖ k!/(k − d)! u₀^{k−d}.
  const u0 = at - half
  const e = new Float64Array(p)
  for (let k = deriv; k < p; k++) {
    let f = 1
    for (let j = 0; j < deriv; j++) f *= k - j
    e[k] = f * u0 ** (k - deriv)
  }
  // w = V (VᵀV)⁻¹ e, so Σ wᵢ xᵢ = eᵀ c with c the least-squares coefficients.
  const g = (solve(fromData(VtV, [p, p]), fromData(e, [p])) as Tensor).data as Float64Array
  const w = new Float64Array(window)
  for (let i = 0; i < window; i++) {
    const u = i - half
    let s = 0
    for (let k = 0; k < p; k++) s += g[k] * u ** k
    w[i] = s / delta ** deriv
  }
  return w
}

/** Options for `savgolFilter`. */
export type SavgolOptions = {
  /** The order of the derivative to estimate (default 0: smooth). */
  deriv?: Size
  /** Sample spacing for derivatives (default $1/f_s$ of a `Signal`, else 1). */
  delta?: Scalar
  /**
   * The ends: `interp` (default, as scipy) fits the polynomial to the first and last windows and evaluates it there;
   * `mirror`, `nearest`, `constant` (zeros) and `wrap` extend the signal.
   */
  mode?: 'interp' | 'mirror' | 'nearest' | 'constant' | 'wrap'
}

/**
 * The Savitzky–Golay filter, as `scipy.signal.savgol_filter`: each sample replaced by the value (or derivative) at
 * its centre of the degree-`polyorder` least-squares polynomial through the `window` samples around it. It preserves
 * polynomials up to that degree, so peaks keep their height better than under a moving average of the same width.
 * Throws `DomainError` for an even window or, in `interp` mode, a window longer than the signal. A `polyorder` of at
 * least `window` is not checked here (unlike `savgolCoeffs`): it leaves the least-squares system singular.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples.
 * @param window The window length, a positive odd integer.
 * @param polyorder The degree of the fitted polynomial, below `window`.
 * @param options The derivative and its sample spacing, and how the ends are handled (`mode`).
 * @returns The smoothed signal (or its derivative), on the input's time axis.
 *
 * @example A parabola passes a quadratic filter unchanged, and its derivative is exact
 * const x = [0, 1, 4, 9, 16, 25, 36]
 * print('smoothed =', savgolFilter(x, 5, 2).data)
 * print('derivative =', savgolFilter(x, 5, 2, { deriv: 1 }).data)
 *
 * @example A narrow peak keeps more of its height than under a moving average
 * const x = [0, 0, 0, 1, 4, 1, 0, 0, 0]
 * print('savgol =', savgolFilter(x, 5, 2).data)
 * // The centred 5-point mean: a causal one, two samples later.
 * const mean5 = lfilter({ b: [0.2, 0.2, 0.2, 0.2, 0.2], a: [1] }, [...x, 0, 0]).y
 * print('moving average =', slice(mean5, [2, 11]))
 */
export function savgolFilter(x: SignalInput, window: Size, polyorder: Size, options: SavgolOptions = {}): Signal {
  const input = readSamples(x, 'savgolFilter')
  const v = input.values
  const n = v.length
  const { deriv = 0, mode = 'interp' } = options
  const delta = options.delta ?? 1 / input.fs
  oddSize(window, 'savgolFilter')
  if (mode === 'interp' && window > n)
    throw new DomainError('savgolFilter', 'savgolFilter: interp mode needs window ≤ the length')
  const half = (window - 1) / 2
  const w = polyWeights(window, polyorder, deriv, half, delta)
  const at = (j: number): number => {
    if (j >= 0 && j < n) return v[j]
    if (mode === 'constant') return 0
    if (mode === 'nearest') return v[j < 0 ? 0 : n - 1]
    if (mode === 'wrap') return v[((j % n) + n) % n]
    // mirror: reflect about the end samples without repeating them (numpy's `reflect`).
    const period = 2 * (n - 1)
    let k = ((j % period) + period) % period
    if (k >= n) k = period - k
    return v[k]
  }
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let k = 0; k < window; k++) s += w[k] * at(i + k - half)
    y[i] = s
  }
  if (mode === 'interp') {
    // The first and last half-windows: the polynomial of the edge window evaluated at each position.
    for (let i = 0; i < half; i++) {
      const wl = polyWeights(window, polyorder, deriv, i, delta)
      const wr = polyWeights(window, polyorder, deriv, window - 1 - i, delta)
      let sl = 0
      let sr = 0
      for (let k = 0; k < window; k++) {
        sl += wl[k] * v[k]
        sr += wr[k] * v[n - window + k]
      }
      y[i] = sl
      y[n - 1 - i] = sr
    }
  }
  return like(input, y)
}

// ── Median ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The running median over an odd `kernelSize`, as `scipy.signal.medfilt` (zero padding at the ends) or, with
 * `padding: 'nearest' | 'mirror'`, `scipy.ndimage.median_filter`. Removes impulses and keeps steps, which a linear
 * smoother cannot do at once. Throws `DomainError` for an even kernel.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples.
 * @param kernelSize The window length, a positive odd integer.
 * @param options How the signal is extended beyond its ends.
 * @param options.padding `zeros` (default, as `medfilt`), `nearest` (the end sample repeated) or `mirror` (reflected
 *   about the end with the end sample repeated, $x_1, x_0 \mid x_0, x_1$: ndimage's `reflect` mode rather than its
 *   `mirror`).
 * @returns The median-filtered signal, on the input's time axis.
 *
 * @example An impulse is removed and a step kept, against scipy.signal.medfilt
 * // scipy: [1, 1, 1, 1, 1, 5, 5, 5].
 * print('y =', medfilt([1, 1, 9, 1, 1, 5, 5, 5], 3).data)
 */
export function medfilt(
  x: SignalInput,
  kernelSize: Size = 3,
  { padding = 'zeros' }: { padding?: 'zeros' | 'nearest' | 'mirror' } = {},
): Signal {
  oddSize(kernelSize, 'medfilt')
  const input = readSamples(x, 'medfilt')
  const v = input.values
  const n = v.length
  const half = (kernelSize - 1) / 2
  const at = (j: number) => {
    if (j >= 0 && j < n) return v[j]
    if (padding === 'zeros') return 0
    if (padding === 'nearest') return v[j < 0 ? 0 : n - 1]
    const period = 2 * n
    let k = ((j % period) + period) % period
    if (k >= n) k = period - 1 - k
    return v[k]
  }
  const y = new Float64Array(n)
  const buf = new Float64Array(kernelSize)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < kernelSize; k++) buf[k] = at(i + k - half)
    buf.sort()
    y[i] = buf[half]
  }
  return like(input, y)
}

// ── Wiener ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The local adaptive Wiener filter, as `scipy.signal.wiener(x, mysize, noise)`: with the local mean $\mu$ and variance
 * $\sigma^2$ over a window of `size` samples (zero-padded), $y = \mu + (1 - \nu/\sigma^2)(x - \mu)$ where
 * $\sigma^2 \ge \nu$, else $\mu$. The noise power $\nu$ defaults to the mean of the local variances. Throws
 * `DomainError` for an even window.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples.
 * @param size The window length, a positive odd integer.
 * @param options The noise power.
 * @param options.noise The noise power $\nu$ (a variance); left out, the mean of the local variances.
 * @returns The filtered signal, on the input's time axis.
 *
 * @example An outlier is pulled towards its neighbours, against scipy.signal.wiener
 * // scipy: [1, 1.333333, 1.915751, 3.809524, 1.915751, 1.333333, 1].
 * print('y =', wiener([1, 2, 1, 5, 1, 2, 1], 3).data)
 */
export function wiener(x: SignalInput, size: Size = 3, { noise }: { noise?: Scalar } = {}): Signal {
  oddSize(size, 'wiener')
  const input = readSamples(x, 'wiener')
  const v = input.values
  const n = v.length
  const half = (size - 1) / 2
  const mean = new Float64Array(n)
  const variance = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    let s2 = 0
    for (let k = -half; k <= half; k++) {
      const u = i + k >= 0 && i + k < n ? v[i + k] : 0
      s += u
      s2 += u * u
    }
    mean[i] = s / size
    variance[i] = s2 / size - mean[i] ** 2
  }
  const nu = noise ?? variance.reduce((a, b) => a + b, 0) / n
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) y[i] = variance[i] < nu ? mean[i] : mean[i] + (1 - nu / variance[i]) * (v[i] - mean[i])
  return like(input, y)
}

/**
 * The result of `wienerDenoise`: the estimate `signal`, and the `gain` applied to each DFT bin from DC to Nyquist
 * ($\lfloor n/2 \rfloor + 1$ of them for $n$ samples) at frequencies `f` (in Hz, in cycles per sample at rate 1).
 */
export type WienerDenoised = { signal: Signal; gain: Tensor; f: Tensor }

/**
 * Frequency-domain Wiener shrinkage of $x = s + w$ with $w$ white of known variance $\sigma^2$: each DFT bin is
 * multiplied by $G = \max(0, 1 - n\sigma^2/\hat{P})$ for $n$ samples, the Wiener gain $S/(S + n\sigma^2)$ with the
 * signal power $S$ estimated as $\hat{P} - n\sigma^2$ ($n\sigma^2$ is the expected $\lvert X \rvert^2$ of the noise),
 * $\hat{P}$ the periodogram $\lvert X \rvert^2$ averaged over neighbouring bins (circularly). The whole record is one
 * block (non-causal); bins where the signal stands above the noise pass, the rest are suppressed.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples.
 * @param noiseVariance The variance $\sigma^2$ of the white noise, per sample.
 * @param options How the periodogram is smoothed.
 * @param options.smoothing The number of bins averaged, centred on each (rounded up to odd; default 5).
 * @returns The estimate as a `Signal` on the input's time axis, with the gain of each bin from DC to Nyquist and their
 *   frequencies.
 *
 * @example A tone in white noise
 * const s = tensor(Array.from({ length: 64 }, (_, k) => Math.sin((2 * Math.PI * 4 * k) / 64)))
 * const x = add(s, normals(stream(1), 64, 0, 0.3))
 * const { signal: y, gain } = wienerDenoise(x, 0.09)
 * print('rms error before =', Math.sqrt(mean(square(sub(x, s)))))
 * print('rms error after =', Math.sqrt(mean(square(sub(y.data, s)))))
 * print('gain, bins 0 to 8 =', slice(gain, [0, 9]))
 */
export function wienerDenoise(
  x: SignalInput,
  noiseVariance: Scalar,
  { smoothing = 5 }: { smoothing?: Size } = {},
): WienerDenoised {
  const input = readSamples(x, 'wienerDenoise')
  const n = input.values.length
  const X = fft(fromData(Float64Array.from(input.values))).data as Float64Array
  const power = new Float64Array(n)
  for (let k = 0; k < n; k++) power[k] = X[2 * k] ** 2 + X[2 * k + 1] ** 2
  const half = Math.floor(smoothing / 2)
  const gain = new Float64Array(n)
  const noise = n * noiseVariance
  for (let k = 0; k < n; k++) {
    let s = 0
    for (let j = -half; j <= half; j++) s += power[(((k + j) % n) + n) % n]
    const p = s / (2 * half + 1)
    gain[k] = p > 0 ? Math.max(0, 1 - noise / p) : 0
  }
  const Y = new Float64Array(2 * n)
  for (let k = 0; k < n; k++) {
    Y[2 * k] = X[2 * k] * gain[k]
    Y[2 * k + 1] = X[2 * k + 1] * gain[k]
  }
  const y = ifft(fromData(Y, [n], 'complex128')).data as Float64Array
  const re = Float64Array.from({ length: n }, (_, k) => y[2 * k])
  const half1 = Math.floor(n / 2) + 1
  return {
    signal: like(input, re),
    gain: fromData(gain.slice(0, half1), [half1]),
    f: fromData(
      Float64Array.from({ length: half1 }, (_, k) => (k * input.fs) / n),
      [half1],
    ),
  }
}

// ── Matched filter ────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of `matchedFilter`. */
export type Matched = {
  /**
   * $y[n] = \sum_k t[k]\, x[n + k] / \lVert t \rVert$: the template's correlation with $x$ at each start $n$ (same
   * length as $x$; the template is cut short at the end of $x$).
   */
  output: Signal
  /** The start sample of the best match (the first, of the largest value, not magnitude) and its output value. */
  peak: { index: Size; value: Scalar }
}

/**
 * The matched filter for a known template $t$ in additive white noise: the FIR filter $h[n] = t[L - 1 - n]$ (the
 * time-reversed template), so the output is the correlation $y[n] = \sum_k t[k]\, x[n + k]$, here aligned so that
 * $y[n]$ scores a template starting at sample $n$, and scaled by $1/\lVert t \rVert$ so a unit-variance noise gives a
 * unit-variance output. With `normalized`, each value is also divided by the local norm
 * $\lVert x[n], \dots, x[n + L - 1] \rVert$, giving a correlation coefficient in $[-1, 1]$ that ignores the amplitude.
 *
 * @param x The signal to search: a single-channel `Signal`, or bare samples.
 * @param template The template $t$ of $L$ samples (a `Signal`'s rate is ignored).
 * @param options Whether to normalise by the local norm.
 * @param options.normalized `true`: the correlation coefficient (0 where the window of $x$ is all zero). Default
 *   `false`: the correlation scaled by $1/\lVert t \rVert$ only.
 * @returns The `output` at every start, on the input's time axis, and its `peak`.
 *
 * @example Find a template in a signal
 * // The template [1, 2, 1] starts at sample 2, at half amplitude.
 * const x = [0, 0, 0.5, 1, 0.5, 0, 0, 0]
 * const raw = matchedFilter(x, [1, 2, 1])
 * print('output =', raw.output.data)
 * print('peak =', raw.peak)
 * print('normalised peak =', matchedFilter(x, [1, 2, 1], { normalized: true }).peak)
 */
export function matchedFilter(
  x: SignalInput,
  template: SignalInput,
  { normalized = false }: { normalized?: boolean } = {},
): Matched {
  const input = readSamples(x, 'matchedFilter')
  const t = readSamples(template, 'matchedFilter').values
  const v = input.values
  const n = v.length
  const L = t.length
  const norm = Math.sqrt(t.reduce((s, u) => s + u * u, 0)) || 1
  const y = new Float64Array(n)
  let best = 0
  for (let i = 0; i < n; i++) {
    let s = 0
    let e = 0
    for (let k = 0; k < L && i + k < n; k++) {
      s += t[k] * v[i + k]
      e += v[i + k] ** 2
    }
    y[i] = normalized ? (e > 0 ? s / (norm * Math.sqrt(e)) : 0) : s / norm
    if (y[i] > y[best]) best = i
  }
  return { output: like(input, y), peak: { index: best, value: y[best] } }
}
