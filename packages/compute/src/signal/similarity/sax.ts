/**
 * Piecewise aggregate approximation (PAA) and symbolic aggregate approximation (SAX) of time series (Lin, Keogh,
 * Lonardi and Chiu 2003, "A symbolic representation of time series, with implications for streaming algorithms", DMKD
 * workshop; Keogh et al. 2001 for PAA).
 *
 * PAA replaces a series of length $n$ by the means of $w$ equal frames (a frame boundary inside a sample splits that
 * sample's weight). SAX z-normalises, takes the PAA, and maps each mean to one of $a$ symbols by the breakpoints
 * $\beta_1 < \dots < \beta_{a-1}$ that cut $\Gauss(0, 1)$ into $a$ equiprobable regions, so each symbol is equally
 * likely for a Gaussian series; symbol $s \in \{0, \dots, a - 1\}$ (the letter `a`, `b`, ...) covers
 * $[\beta_s, \beta_{s+1})$, with $\beta_0 = -\infty$ and $\beta_a = \infty$. MINDIST between two words,
 * $\sqrt{n/w}\,\sqrt{\sum_i \operatorname{dist}(s_i, t_i)^2}$ with $\operatorname{dist}$ the gap between the symbols'
 * regions, lower-bounds the Euclidean distance between the z-normalised series.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { zNormalise } from './profile'

/**
 * The PAA of a series: the means of $w$ equal frames ($w \le n$); frames that split a sample share it by weight, so
 * $w$ need not divide $n$. Throws `DomainError` unless $w$ is an integer in $1, \dots, n$.
 *
 * @param x The series, of length $n$.
 * @param segments The number of frames $w$.
 * @returns The $w$ frame means, in order.
 *
 * @example Frames that divide the series, and frames that split samples
 * print('3 frames =', paa([1, 2, 3, 4, 5, 6], 3))
 * print('4 frames =', paa([1, 2, 3, 4, 5, 6], 4))
 */
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

/**
 * The $a - 1$ breakpoints of an $a$-symbol SAX alphabet: the standard normal quantiles at
 * $1/a, 2/a, \dots, (a - 1)/a$. Throws `DomainError` unless $a$ is an integer of at least 2.
 *
 * @param alphabet The number of symbols $a$.
 * @returns The breakpoints $\beta_1 < \dots < \beta_{a-1}$, ascending.
 *
 * @example Three and four symbols
 * print('a = 3:', saxBreakpoints(3))
 * print('a = 4:', saxBreakpoints(4))
 */
export function saxBreakpoints(alphabet: Size): Tensor {
  if (!(Number.isInteger(alphabet) && alphabet >= 2))
    throw new DomainError('saxBreakpoints', 'saxBreakpoints: the alphabet needs at least 2 symbols')
  return fromData(Float64Array.from({ length: alphabet - 1 }, (_, k) => normalQuantile((k + 1) / alphabet) as number))
}

/** Options of {@link sax}. */
export interface SaxOptions {
  /** The number of PAA frames $w$, one symbol each. */
  segments: Size
  /** The number of symbols $a$ (at least 2). */
  alphabet: Size
  /** z-normalise first (default true). */
  normalise?: boolean
}

/** A SAX word. */
export interface SaxWord {
  /** Symbol indices $0, \dots, a - 1$ (int32), one per segment. */
  readonly symbols: Tensor
  /** The symbols as letters: 0 is `a`, 1 is `b`, and so on. */
  readonly word: string
  /** The PAA the symbols quantise. */
  readonly paa: Tensor
}

/**
 * The SAX word of a series (see the file notes): z-normalise (unless `normalise` is false), take the PAA, and give
 * each frame mean the symbol of the breakpoint region it falls in. Throws `DomainError` for a number of segments
 * outside $1, \dots, n$ or an alphabet of fewer than 2 symbols.
 *
 * @param x The series, of length $n$.
 * @param options The number of `segments` $w$ and the `alphabet` size $a$, and `normalise` (default true), whether to
 *   z-normalise first.
 * @returns The word's symbol indices, its letters, and the PAA they quantise.
 *
 * @example A rising ramp climbs through the alphabet
 * const s = sax([0, 1, 2, 3, 4, 5, 6, 7], { segments: 4, alphabet: 4 })
 * print('word =', s.word)
 * print('symbols =', s.symbols)
 * print('paa =', s.paa)
 */
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
 * MINDIST between two SAX words of $w$ symbols from series of length $n$:
 * $\sqrt{n/w}\,\sqrt{\sum_i \operatorname{dist}(s_i, t_i)^2}$, where $\operatorname{dist}(s, t)$ is 0 for equal or
 * adjacent symbols and otherwise the gap $\beta_{\max(s, t)} - \beta_{\min(s, t) + 1}$ between their regions (symbols
 * counted from 0, breakpoints from 1). A lower bound on the Euclidean distance between the z-normalised series. Throws
 * `ShapeError` when the words differ in length.
 *
 * @param a The first word's symbol indices (the `symbols` of `sax`).
 * @param b The second word's symbol indices, as many as `a`.
 * @param n The length of the series the words were made from.
 * @param alphabet The alphabet size $a$ the words use.
 * @returns The MINDIST lower bound.
 *
 * @example A rising and a falling ramp: the bound beside the true distance
 * const x = [0, 1, 2, 3, 4, 5, 6, 7]
 * const y = [7, 6, 5, 4, 3, 2, 1, 0]
 * const options = { segments: 4, alphabet: 4 }
 * print('words =', sax(x, options).word, sax(y, options).word)
 * print('MINDIST =', saxMinDist(sax(x, options).symbols, sax(y, options).symbols, 8, 4))
 * const d = sub(zNormalise(x), zNormalise(y))
 * print('Euclidean distance =', sqrt(sum(mul(d, d))))
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
