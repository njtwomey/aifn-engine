/**
 * Distance profiles: the distance from one query subsequence to every subsequence of a series, and the sliding
 * statistics they are built from.
 *
 * With z-normalisation (each subsequence shifted to mean 0 and scaled to standard deviation 1) the squared distance
 * between the subsequences of length $m$ at $i$ and $j$ is
 * $d_{ij}^2 = 2m\left(1 - \frac{QT_{ij} - m\mu_i\mu_j}{m\sigma_i\sigma_j}\right)$, where $QT_{ij}$ is their dot
 * product and $\mu$, $\sigma$ their means and (population) standard deviations. A whole profile then needs only the
 * sliding dot products, which one FFT convolution gives: MASS, Mueen's Algorithm for Similarity Search (Mueen et al.
 * 2017, "The fastest similarity search algorithm for time series subsequences under Euclidean distance"; Yeh et al.
 * 2016, "Matrix Profile I", ICDM). Constant subsequences (standard deviation below $10^{-12}$) follow stumpy: two
 * constant subsequences are at distance 0, a constant and a varying one at $\sqrt{m}$.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fftConvolve } from 'aifn-compute/foundation/convolution'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { readSamples, type SignalInput } from '../signal'

/** Below this standard deviation a subsequence is treated as constant. */
const FLAT = 1e-12

/**
 * The mean and (population) standard deviation of every length-$m$ window of a series of length $n$. Throws
 * `DomainError` unless $m$ is an integer in $2, \dots, n$.
 *
 * @param x The series: a single-channel signal or its samples.
 * @param m The window length.
 * @returns `mean` and `std`, each of length $n - m + 1$; entry $i$ is of the window starting at sample $i$.
 *
 * @example Windows of length 3 slide along a ramp, then a jump
 * const { mean, std } = slidingMeanStd([1, 2, 3, 4, 5, 9], 3)
 * print('mean =', mean)
 * print('std =', std)
 */
export function slidingMeanStd(x: SignalInput, m: Size): { mean: Tensor; std: Tensor } {
  const v = readSamples(x, 'slidingMeanStd').values
  checkWindow(m, v.length, 'slidingMeanStd')
  const { mean, std } = meanStd(v, m)
  return { mean: fromData(mean), std: fromData(std) }
}

/**
 * Throws `DomainError` unless the window length $m$ is an integer in $2, \dots, n$.
 *
 * @param m The window (subsequence) length to check.
 * @param n The length of the series it slides along.
 * @param where The caller's name for error messages.
 */
export function checkWindow(m: Size, n: Size, where: string): void {
  if (!(Number.isInteger(m) && m >= 2 && m <= n))
    throw new DomainError(where, `${where}: the window length ${m} must lie in 2 … ${n}`)
}

/**
 * Window means and population standard deviations, each window summed afresh (two passes, $O(nm)$) so no rounding
 * accumulates. Unchecked: the caller ensures $1 \le m \le n$.
 *
 * @param v The series, of length $n$; not modified.
 * @param m The window length.
 * @returns `mean` and `std`, each of length $n - m + 1$, for the windows starting at samples $0, \dots, n - m$.
 */
export function meanStd(v: Float64Array, m: number): { mean: Float64Array; std: Float64Array } {
  const k = v.length - m + 1
  const mean = new Float64Array(k)
  const std = new Float64Array(k)
  for (let i = 0; i < k; i++) {
    let s = 0
    for (let j = 0; j < m; j++) s += v[i + j]
    const mu = s / m
    let q = 0
    for (let j = 0; j < m; j++) q += (v[i + j] - mu) ** 2
    mean[i] = mu
    std[i] = Math.sqrt(q / m)
  }
  return { mean, std }
}

/**
 * The dot products $QT_i = \sum_{k=0}^{m-1} q_k t_{i+k}$ of a query with every window of a series, by one FFT
 * convolution of the series with the reversed query. Throws `DomainError` unless the query's length $m$ is in
 * $2, \dots, n$.
 *
 * @param q The query, of length $m$.
 * @param t The series, of length $n$: a single-channel signal or its samples.
 * @returns The $n - m + 1$ dot products, entry $i$ for the window starting at sample $i$ (exact up to FFT rounding).
 *
 * @example A window of ones gives the sums of adjacent pairs
 * print('dot products =', slidingDotProduct([1, 1], [1, 2, 3, 4]))
 */
export function slidingDotProduct(q: VectorLike, t: SignalInput): Tensor {
  const qv = dense.toF64(q, 'slidingDotProduct')
  const tv = readSamples(t, 'slidingDotProduct').values
  checkWindow(qv.length, tv.length, 'slidingDotProduct')
  const rev = Float64Array.from(qv).reverse()
  return fromData(Float64Array.from(toFlat(fftConvolve(fromData(tv), fromData(rev), { mode: 'valid' }) as Tensor)))
}

/**
 * The z-normalised distance between two windows from their dot product, means and standard deviations (the file's
 * formula, with the correlation clipped at 1 against rounding). A window with standard deviation below $10^{-12}$ is
 * constant: two constant windows are at 0, a constant and a varying one at $\sqrt{m}$.
 *
 * @param qt The dot product $QT$ of the two windows.
 * @param m The window length.
 * @param mu1 The mean of the first window.
 * @param s1 The population standard deviation of the first window.
 * @param mu2 The mean of the second window.
 * @param s2 The population standard deviation of the second window.
 * @returns The distance between the z-normalised windows, in $[0, 2\sqrt{m}]$.
 *
 * @example The same shape at twice the scale, and the shape reversed
 * // [1, 2, 3] has mean 2 and standard deviation sqrt(2/3); [2, 4, 6] and [3, 2, 1] are compared with it.
 * const s = Math.sqrt(2 / 3)
 * print('[1, 2, 3] vs [2, 4, 6] =', zDistance(1 * 2 + 2 * 4 + 3 * 6, 3, 2, s, 4, 2 * s))
 * print('[1, 2, 3] vs [3, 2, 1] =', zDistance(1 * 3 + 2 * 2 + 3 * 1, 3, 2, s, 2, s))
 */
export function zDistance(qt: number, m: number, mu1: number, s1: number, mu2: number, s2: number): number {
  const flat1 = s1 < FLAT
  const flat2 = s2 < FLAT
  if (flat1 && flat2) return 0
  if (flat1 || flat2) return Math.sqrt(m)
  const rho = (qt - m * mu1 * mu2) / (m * s1 * s2)
  return Math.sqrt(Math.max(0, 2 * m * (1 - Math.min(1, rho))))
}

/** Options of {@link distanceProfile}. */
export interface DistanceProfileOptions {
  /** z-normalise every subsequence (default true); false gives plain Euclidean distances. */
  normalise?: boolean
}

/**
 * The distance profile of a query against a series, by MASS (see the file notes): the distance from the query to
 * every window of the series of the query's length. Throws `DomainError` unless the query's length $m$ is in
 * $2, \dots, n$.
 *
 * @param query The query subsequence, of length $m$.
 * @param series The series to search, of length $n$: a single-channel signal or its samples.
 * @param options `normalise`: z-normalised distances (the default), or plain Euclidean ones with `false`.
 * @returns The $n - m + 1$ distances, entry $i$ for the window starting at sample $i$.
 *
 * @example A bump found at its own scale and at three times it
 * const series = [0, 0, 1, 3, 1, 0, 0, 2, 6, 2, 0]
 * print('z-normalised =', distanceProfile([1, 3, 1], series))
 * print('Euclidean =', distanceProfile([1, 3, 1], series, { normalise: false }))
 */
export function distanceProfile(query: VectorLike, series: SignalInput, options: DistanceProfileOptions = {}): Tensor {
  const q = dense.toF64(query, 'distanceProfile')
  const t = readSamples(series, 'distanceProfile').values
  const m = q.length
  checkWindow(m, t.length, 'distanceProfile')
  const qt = toFlat(slidingDotProduct(q, t))
  const k = t.length - m + 1
  const out = new Float64Array(k)
  if (options.normalise === false) {
    let qq = 0
    for (const v of q) qq += v * v
    let tt = 0
    for (let j = 0; j < m; j++) tt += t[j] * t[j]
    for (let i = 0; i < k; i++) {
      if (i > 0) tt += t[i + m - 1] ** 2 - t[i - 1] ** 2
      out[i] = Math.sqrt(Math.max(0, qq + tt - 2 * qt[i]))
    }
    return fromData(out)
  }
  const { mean: mq, std: sq } = meanStd(q, m)
  const { mean, std } = meanStd(t, m)
  for (let i = 0; i < k; i++) out[i] = zDistance(qt[i], m, mq[0], sq[0], mean[i], std[i])
  return fromData(out)
}

/**
 * A copy of a series shifted to mean 0 and scaled to (population) standard deviation 1. A constant series (standard
 * deviation below $10^{-12}$) becomes zeros.
 *
 * @param x The series; not modified.
 * @returns The z-normalised series, the same length as `x`.
 *
 * @example A ramp, and a constant
 * print('ramp =', zNormalise([1, 2, 3]))
 * print('constant =', zNormalise([5, 5, 5]))
 */
export function zNormalise(x: VectorLike): Tensor {
  const v = dense.toF64(x, 'zNormalise')
  const { mean, std } = meanStd(v, v.length)
  return fromData(Float64Array.from(v, (u) => (std[0] < FLAT ? 0 : (u - mean[0]) / std[0])))
}
