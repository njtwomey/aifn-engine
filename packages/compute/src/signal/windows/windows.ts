/**
 * Window functions, as `scipy.signal.windows` (Harris, 1978, "On the use of windows for harmonic analysis with the
 * discrete Fourier transform", Proc. IEEE 66(1)). Symmetric by default (for filter design); `periodic: true` gives the
 * DFT-even form used for spectral analysis (a symmetric window of length $n + 1$ with its last sample dropped).
 *
 * The cosine-sum windows (rectangular, Hann, Hamming, Blackman, Blackman–Harris, Nuttall, flat top) are
 * $w[k] = \sum_j (-1)^j a_j \cos\big(2\pi j k / (m - 1)\big)$ for $k = 0, \dots, m - 1$ (window length $m$), with
 * scipy's coefficients $a_j$; the others are built from their own formulas. A window of length 1 is $[1]$.
 */

import { besselI0 } from 'aifn-compute/numerics/special'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The windows that need no parameter, as `getWindow` names them: `'boxcar'` is `'rectangular'`, `'bartlett'` has zero
 * end points and `'triangular'` (scipy's `triang`) has non-zero ones.
 */
export type WindowName =
  | 'rectangular'
  | 'boxcar'
  | 'hann'
  | 'hamming'
  | 'blackman'
  | 'blackmanharris'
  | 'nuttall'
  | 'flattop'
  | 'bartlett'
  | 'triangular'

/**
 * A window by name, or a parameterised window: `kaiser` with shape $\beta$ (`beta`; larger lowers the side lobes and
 * widens the main lobe), `gaussian` with standard deviation `std` in samples, `tukey` with `alpha`, the fraction of the
 * window inside the cosine tapers (0 is rectangular, 1 is Hann), and the cosine (sine) window, which takes none.
 */
export type WindowSpec =
  | WindowName
  | { name: 'kaiser'; beta: number }
  | { name: 'gaussian'; std: number }
  | { name: 'tukey'; alpha: number }
  | { name: 'cosine' }

/**
 * A window given by spec, or explicitly as values (a tensor or array, whose length must match the segment it is applied
 * to).
 */
export type WindowInput = WindowSpec | Tensor | ArrayLike<number>

/** The coefficients $a_0, a_1, \dots$ of each cosine-sum window, as scipy's (Hamming's are 0.54 and 0.46). */
const COSINE_SUMS: Partial<Record<WindowName, readonly number[]>> = {
  rectangular: [1],
  boxcar: [1],
  hann: [0.5, 0.5],
  hamming: [0.54, 0.46],
  blackman: [0.42, 0.5, 0.08],
  blackmanharris: [0.35875, 0.48829, 0.14128, 0.01168],
  nuttall: [0.3635819, 0.4891775, 0.1365995, 0.0106411],
  flattop: [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
}

/**
 * The symmetric window of length $m$ (its values mirror about the centre $(m - 1)/2$). Throws `DomainError` for a spec
 * it does not know.
 *
 * @param spec The window: a name, or a parameterised spec whose parameter is read here (not checked for range).
 * @param m The number of samples, at least 1 (a length of 1 gives $[1]$ for every window).
 * @returns A fresh array of $m$ values.
 */
function symmetric(spec: WindowSpec, m: number): Float64Array {
  const w = new Float64Array(m)
  if (m === 1) return w.fill(1)
  const name = typeof spec === 'string' ? spec : spec.name
  const a = typeof spec === 'string' ? COSINE_SUMS[spec] : undefined
  if (a) {
    // General cosine window w[n] = Σ_k (−1)^k a_k cos(2πkn/(M − 1)).
    for (let n = 0; n < m; n++) {
      let v = 0
      for (let k = 0; k < a.length; k++) v += (k % 2 ? -1 : 1) * a[k] * Math.cos((2 * Math.PI * k * n) / (m - 1))
      w[n] = v
    }
    return w
  }
  switch (name) {
    case 'bartlett':
      for (let n = 0; n < m; n++) w[n] = (2 / (m - 1)) * ((m - 1) / 2 - Math.abs(n - (m - 1) / 2))
      return w
    case 'triangular': {
      // scipy.signal.windows.triang: non-zero end points.
      const half = Math.floor((m + 1) / 2)
      for (let k = 1; k <= half; k++) {
        const v = m % 2 === 0 ? (2 * k - 1) / m : (2 * k) / (m + 1)
        w[k - 1] = v
        w[m - k] = v
      }
      return w
    }
    case 'kaiser': {
      const beta = (spec as { beta: number }).beta
      const scale = besselI0(beta)
      for (let n = 0; n < m; n++) {
        const r = (2 * n) / (m - 1) - 1
        w[n] = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / scale
      }
      return w
    }
    case 'gaussian': {
      const std = (spec as { std: number }).std
      for (let n = 0; n < m; n++) w[n] = Math.exp(-0.5 * ((n - (m - 1) / 2) / std) ** 2)
      return w
    }
    case 'tukey': {
      const alpha = (spec as { alpha: number }).alpha
      if (alpha <= 0) return w.fill(1)
      if (alpha >= 1) return symmetric('hann', m)
      // scipy's construction: cosine tapers over the first and last α(M − 1)/2 samples.
      const width = Math.floor((alpha * (m - 1)) / 2)
      for (let n = 0; n < m; n++) {
        if (n <= width) w[n] = 0.5 * (1 + Math.cos(Math.PI * (-1 + (2 * n) / alpha / (m - 1))))
        else if (n >= m - width - 1) w[n] = 0.5 * (1 + Math.cos(Math.PI * (-2 / alpha + 1 + (2 * n) / alpha / (m - 1))))
        else w[n] = 1
      }
      return w
    }
    case 'cosine':
      for (let n = 0; n < m; n++) w[n] = Math.sin((Math.PI * (n + 0.5)) / m)
      return w
  }
  throw new DomainError('getWindow', `getWindow: unknown window ${JSON.stringify(spec)}`)
}

/**
 * A window of length $n$, as `scipy.signal.get_window(spec, n, fftbins=periodic)`. Names: rectangular (boxcar), hann,
 * hamming, blackman, blackmanharris, nuttall, flattop, bartlett, triangular; parameterised: kaiser ($\beta$), gaussian
 * (std, in samples), tukey ($\alpha$, the tapered fraction), and cosine (`{ name: 'cosine' }`). Throws `DomainError`
 * for a length that is not a non-negative integer or an unknown window.
 *
 * @param spec The window, by name or as a parameterised spec.
 * @param n The number of samples; 0 gives an empty window.
 * @param options Which form of the window to build.
 * @param options.periodic `false` (default): the symmetric window, for filter design. `true`: the periodic (DFT-even)
 *   window for spectral analysis, the symmetric window of length $n + 1$ without its last sample.
 * @returns The $n$ window values, as a float64 tensor.
 *
 * @example A Hann window, symmetric and periodic
 * print('symmetric =', getWindow('hann', 5))
 * print('periodic =', getWindow('hann', 4, { periodic: true }))
 *
 * @example Parameterised windows
 * print('kaiser, beta 5 =', getWindow({ name: 'kaiser', beta: 5 }, 5))
 * print('tukey, alpha 0.5 =', getWindow({ name: 'tukey', alpha: 0.5 }, 8))
 */
export function getWindow(spec: WindowSpec, n: Size, { periodic = false }: { periodic?: boolean } = {}): Tensor {
  if (!Number.isInteger(n) || n < 0)
    throw new DomainError('getWindow', `getWindow: length must be a non-negative integer, got ${n}`)
  if (n === 0) return fromData(new Float64Array(0))
  if (!periodic) return fromData(symmetric(spec, n))
  return fromData(symmetric(spec, n + 1).slice(0, n))
}

/**
 * The values of a window input of length $n$ as a fresh float64 array: a spec is built with the given symmetry
 * (`periodic` for spectral analysis), explicit values are copied and checked for length. Throws `ShapeError` when
 * explicit values do not have $n$ entries.
 *
 * @param input The window: a spec (name or parameterised), or its values as a tensor or array.
 * @param n The length the window must have: the segment length it will multiply.
 * @param periodic For a spec, whether to build the periodic (DFT-even) window rather than the symmetric one; ignored
 *   for explicit values.
 * @returns A fresh array of $n$ values, safe to modify.
 *
 * @example A spec is built, explicit values are copied
 * print('hamming =', windowValues('hamming', 4, true))
 * print('given =', windowValues([1, 2, 2, 1], 4, false))
 */
export function windowValues(input: WindowInput, n: Size, periodic: boolean): Float64Array {
  if (typeof input === 'string' || (typeof input === 'object' && 'name' in input && !isTensor(input)))
    return getWindow(input as WindowSpec, n, { periodic }).data as Float64Array
  const v = Float64Array.from(isTensor(input) ? dense.data(input) : (input as ArrayLike<number>))
  if (v.length !== n)
    throw new ShapeError('windowValues', `window of length ${v.length} does not match segment length ${n}`)
  return v
}
