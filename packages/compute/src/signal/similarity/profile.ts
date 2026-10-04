/**
 * Distance profiles: the distance from one query subsequence to every subsequence of a series. With z-normalisation
 * (each subsequence shifted to mean 0 and scaled to standard deviation 1) the squared distance between subsequences
 * at i and j of length m is d² = 2m(1 − (QTᵢⱼ − m μᵢ μⱼ)/(m σᵢ σⱼ)), where QT is their dot product, so a whole profile
 * needs only the sliding dot products, which one FFT convolution gives: MASS, Mueen's Algorithm for Similarity Search
 * (Mueen et al. 2017, "The fastest similarity search algorithm for time series subsequences under Euclidean distance";
 * Yeh et al. 2016, "Matrix Profile I", ICDM). Constant subsequences follow stumpy: two constant subsequences are at
 * distance 0, a constant and a varying one at √m.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fftConvolve } from 'aifn-compute/foundation/convolution'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { readSamples, type SignalInput } from '../signal'

/** Below this standard deviation a subsequence is treated as constant. */
const FLAT = 1e-12

/** The mean and standard deviation of every length-m window of x (n − m + 1 each). */
export function slidingMeanStd(x: SignalInput, m: Size): { mean: Tensor; std: Tensor } {
  const v = readSamples(x, 'slidingMeanStd').values
  checkWindow(m, v.length, 'slidingMeanStd')
  const { mean, std } = meanStd(v, m)
  return { mean: fromData(mean), std: fromData(std) }
}

/** Throws unless the window length m lies in 2 … n. */
export function checkWindow(m: Size, n: Size, where: string): void {
  if (!(Number.isInteger(m) && m >= 2 && m <= n))
    throw new DomainError(where, `${where}: the window length ${m} must lie in 2 … ${n}`)
}

/** Window means and standard deviations, each window summed afresh (two passes, O(nm)) so no rounding accumulates. */
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

/** The dot products of q (length m) with every length-m window of t (n − m + 1), by FFT convolution. */
export function slidingDotProduct(q: VectorLike, t: SignalInput): Tensor {
  const qv = dense.toF64(q, 'slidingDotProduct')
  const tv = readSamples(t, 'slidingDotProduct').values
  checkWindow(qv.length, tv.length, 'slidingDotProduct')
  const rev = Float64Array.from(qv).reverse()
  return fromData(Float64Array.from(toFlat(fftConvolve(fromData(tv), fromData(rev), { mode: 'valid' }) as Tensor)))
}

/** The z-normalised distance from QT, the two windows' means and standard deviations and the length m. */
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
 * The distance profile of a query (length m) against a series (length n): n − m + 1 distances, by MASS (module notes).
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

/** A copy of x shifted to mean 0 and scaled to standard deviation 1 (a constant series becomes zeros). */
export function zNormalise(x: VectorLike): Tensor {
  const v = dense.toF64(x, 'zNormalise')
  const { mean, std } = meanStd(v, v.length)
  return fromData(Float64Array.from(v, (u) => (std[0] < FLAT ? 0 : (u - mean[0]) / std[0])))
}
