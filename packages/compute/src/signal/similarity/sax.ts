/**
 * Piecewise aggregate approximation and SAX (Lin, Keogh, Lonardi and Chiu 2003, "A symbolic representation of time
 * series, with implications for streaming algorithms", DMKD workshop; Keogh et al. 2001 for PAA). PAA replaces a
 * series of length n by the means of w equal frames (a frame boundary inside a sample splits that sample's weight).
 * SAX z-normalises, takes the PAA, and maps each mean to one of a symbols by the breakpoints β₁ < … < β_{a−1} that cut
 * N(0, 1) into a equiprobable regions, so each symbol is equally likely for a Gaussian series. MINDIST between two
 * words, √(n/w) √(Σ dist(sᵢ, tᵢ)²) with dist the gap between the symbols' regions, lower-bounds the Euclidean distance
 * between the z-normalised series.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { zNormalise } from './profile'

/** The PAA of a series: the means of w equal frames (w ≤ n); frames that split a sample share it by weight. */
export function paa(x: VectorLike, segments: Size): Tensor {
  const v = dense.toF64(x, 'paa')
  const n = v.length
  if (!(Number.isInteger(segments) && segments >= 1 && segments <= n))
    throw new DomainError('paa', `paa: the number of segments must lie in 1 … ${n}`)
  const out = new Float64Array(segments)
  // Sample i covers [i·w, (i + 1)·w) in units of n·w; frame k covers [k·n, (k + 1)·n).
  for (let k = 0; k < segments; k++) {
    const lo = k * n
    const hi = (k + 1) * n
    let s = 0
    for (let i = Math.floor(lo / segments); i < Math.min(n, Math.ceil(hi / segments)); i++) {
      const a = Math.max(lo, i * segments)
      const b = Math.min(hi, (i + 1) * segments)
      if (b > a) s += v[i] * (b - a)
    }
    out[k] = s / n
  }
  return fromData(out)
}

/** The a − 1 breakpoints of an a-symbol SAX alphabet: the standard normal quantiles at 1/a, 2/a, …, (a − 1)/a. */
export function saxBreakpoints(alphabet: Size): Tensor {
  if (!(Number.isInteger(alphabet) && alphabet >= 2))
    throw new DomainError('saxBreakpoints', 'saxBreakpoints: the alphabet needs at least 2 symbols')
  return fromData(Float64Array.from({ length: alphabet - 1 }, (_, k) => normalQuantile((k + 1) / alphabet) as number))
}

/** Options of {@link sax}. */
export interface SaxOptions {
  segments: Size
  alphabet: Size
  /** z-normalise first (default true). */
  normalise?: boolean
}

/** A SAX word. */
export interface SaxWord {
  /** Symbol indices 0 … a − 1 (int32), one per segment. */
  readonly symbols: Tensor
  /** The symbols as letters a, b, c, …. */
  readonly word: string
  /** The PAA the symbols quantise. */
  readonly paa: Tensor
}

/** The SAX word of a series (module notes). */
export function sax(x: VectorLike, options: SaxOptions): SaxWord {
  const { segments, alphabet, normalise = true } = options
  const z = normalise ? zNormalise(x) : fromData(dense.toF64(x, 'sax'))
  const p = paa(z, segments)
  const beta = dense.data(saxBreakpoints(alphabet))
  const symbols = Int32Array.from(dense.data(p), (v) => {
    let s = 0
    while (s < beta.length && v >= beta[s]) s++
    return s
  })
  return {
    symbols: fromData(symbols),
    word: Array.from(symbols, (s) => String.fromCharCode(97 + s)).join(''),
    paa: p,
  }
}

/**
 * MINDIST between two SAX words of w symbols from series of length n: √(n/w) √(Σ dist(sᵢ, tᵢ)²), where dist is 0 for
 * equal or adjacent symbols and β_{max(s,t)−1} − β_{min(s,t)} otherwise.
 */
export function saxMinDist(a: VectorLike, b: VectorLike, n: Size, alphabet: Size): number {
  const s = dense.toF64(a, 'saxMinDist')
  const t = dense.toF64(b, 'saxMinDist')
  if (s.length !== t.length) throw new ShapeError('saxMinDist', 'saxMinDist: the words must have equal lengths')
  const beta = dense.data(saxBreakpoints(alphabet))
  let q = 0
  for (let i = 0; i < s.length; i++) {
    const lo = Math.min(s[i], t[i])
    const hi = Math.max(s[i], t[i])
    if (hi - lo > 1) q += (beta[hi - 1] - beta[lo]) ** 2
  }
  return Math.sqrt(n / s.length) * Math.sqrt(q)
}
