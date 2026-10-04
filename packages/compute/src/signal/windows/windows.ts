/**
 * Window functions, as `scipy.signal.windows` (Harris, 1978, "On the use of windows for harmonic analysis with the
 * discrete Fourier transform", Proc. IEEE 66(1)). Symmetric by default (for filter design); `periodic: true` gives the
 * DFT-even form used for spectral analysis (a symmetric window of length n + 1 with its last sample dropped).
 */

import { besselI0 } from 'aifn-compute/numerics/special'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The windows that need no parameter. */
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

/** A window by name, or a parameterised window. */
export type WindowSpec =
  | WindowName
  | { name: 'kaiser'; beta: number }
  | { name: 'gaussian'; std: number }
  | { name: 'tukey'; alpha: number }
  | { name: 'cosine' }

/** A window given by spec, or explicitly as values (its length must match). */
export type WindowInput = WindowSpec | Tensor | ArrayLike<number>

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
 * A window of length n, as `scipy.signal.get_window(spec, n, fftbins=periodic)`. Names: rectangular (boxcar), hann,
 * hamming, blackman, blackmanharris, nuttall, flattop, bartlett, triangular, cosine; parameterised: kaiser (β),
 * gaussian (std, in samples), tukey (α, the tapered fraction).
 */
export function getWindow(spec: WindowSpec, n: Size, { periodic = false }: { periodic?: boolean } = {}): Tensor {
  if (!Number.isInteger(n) || n < 0)
    throw new DomainError('getWindow', `getWindow: length must be a non-negative integer, got ${n}`)
  if (n === 0) return fromData(new Float64Array(0))
  if (!periodic) return fromData(symmetric(spec, n))
  return fromData(symmetric(spec, n + 1).slice(0, n))
}

/**
 * The values of a window input of length n as a fresh float64 array: a spec is built with the given symmetry
 * (`periodic` for spectral analysis), explicit values are checked for length.
 */
export function windowValues(input: WindowInput, n: Size, periodic: boolean): Float64Array {
  if (typeof input === 'string' || (typeof input === 'object' && 'name' in input && !isTensor(input)))
    return getWindow(input as WindowSpec, n, { periodic }).data as Float64Array
  const v = Float64Array.from(isTensor(input) ? dense.data(input) : (input as ArrayLike<number>))
  if (v.length !== n)
    throw new ShapeError('windowValues', `window of length ${v.length} does not match segment length ${n}`)
  return v
}
