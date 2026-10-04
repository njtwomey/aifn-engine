/**
 * Smoothing and detection filters, as scipy.signal: the Savitzky–Golay filter (local least-squares polynomials;
 * Savitzky and Golay, 1964, Anal. Chem. 36(8)), the running median (Tukey, 1977), the local adaptive Wiener filter
 * (`scipy.signal.wiener`; Lim, 1990, "Two-Dimensional Signal and Image Processing", §9.2), a frequency-domain Wiener
 * shrinkage for additive white noise of known variance (Wiener, 1949), and the matched filter (Turin, 1960, IRE Trans.
 * Inf. Theory 6(3)): correlation with the template, which maximises the output SNR in white noise.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Signal, Size } from 'aifn-compute/foundation/contracts'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import { solve } from 'aifn-compute/numerics/linalg'
import { readSamples, signal, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A `Signal` on the input's time axis holding `values`. */
function like(input: { fs: Scalar; t0: Scalar; unit?: string }, values: Float64Array): Signal {
  return signal(fromData(values, [values.length]), {
    fs: input.fs,
    t0: input.t0,
    ...(input.unit !== undefined ? { unit: input.unit } : {}),
  })
}

const oddSize = (k: Size, where: string) => {
  if (!(Number.isInteger(k) && k >= 1 && k % 2 === 1))
    throw new DomainError(where, `${where}: the window length must be odd`)
}

// ── Savitzky–Golay ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Savitzky–Golay convolution weights, as `scipy.signal.savgol_coeffs(window, polyorder, deriv, delta,
 * use='dot')`: weight i (i = 0 … window − 1, centred at (window − 1)/2) gives the `deriv`-th derivative at the centre
 * of the least-squares polynomial of degree `polyorder` through the window, d!/Δᵈ times row d of (VᵀV)⁻¹Vᵀ with
 * V the Vandermonde matrix of the offsets.
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

/** Weights that evaluate the d-th derivative at offset `at` (from the window start) of the window's LS polynomial. */
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
  deriv?: Size
  /** Sample spacing for derivatives (default 1/fs of a `Signal`, else 1). */
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
 * smoother cannot do at once.
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
 * The local adaptive Wiener filter, as `scipy.signal.wiener(x, mysize, noise)`: with the local mean μ and variance
 * σ² over a window of `size` samples (zero-padded), y = μ + (1 − ν/σ²)(x − μ) where σ² > ν, else μ. The noise power
 * ν defaults to the mean of the local variances.
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

/** The result of `wienerDenoise`: the estimate and the gain applied to each DFT bin. */
export type WienerDenoised = { signal: Signal; gain: Tensor; f: Tensor }

/**
 * Frequency-domain Wiener shrinkage of x = s + w with w white of known variance σ²: each DFT bin is multiplied by
 * G = max(0, 1 − Nσ² / P̂), the Wiener gain S/(S + N) with the signal power S estimated as P̂ − Nσ², P̂ the periodogram
 * |X|² averaged over `smoothing` neighbouring bins. The whole record is one block (non-causal); bins where the signal
 * stands above the noise pass, the rest are suppressed.
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
  /** y[n] = Σₖ t[k] x[n + k] / ‖t‖: the template's correlation with x at each start n (same length as x). */
  output: Signal
  /** The start sample of the best match and its value. */
  peak: { index: Size; value: Scalar }
}

/**
 * The matched filter for a known template t in additive white noise: the FIR filter h[n] = t[L − 1 − n] (the
 * time-reversed template), so the output is the correlation y[n] = Σₖ t[k] x[n + k], here aligned so that y[n] scores
 * a template starting at sample n, and scaled by 1/‖t‖ so a unit-variance noise gives a unit-variance output. With
 * `normalized`, each value is also divided by the local energy ‖x[n … n + L − 1]‖, giving a correlation coefficient in
 * [−1, 1] that ignores the amplitude.
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
