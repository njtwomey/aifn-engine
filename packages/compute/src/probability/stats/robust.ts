/**
 * Robust location and scatter: the minimum covariance determinant estimator (Rousseeuw, 1984) by the FastMCD
 * algorithm of Rousseeuw and Van Driessen (1999), with the consistency correction and reweighting step that
 * scikit-learn's `MinCovDet` applies, and the squared Mahalanobis distances it gives.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { child, permutation, stream as makeStream, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { cholesky, solveTriangular } from 'aifn-compute/numerics/linalg'
import { regularisedGammaP, regularisedGammaPInverse } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The χ²_p quantile at probability q: 2 P⁻¹(p/2, q). */
const chiSquareQuantile = (p: number, q: number) => 2 * (regularisedGammaPInverse(p / 2, q) as number)

/**
 * The factor that makes the covariance of the share α of normal points nearest the centre consistent for the full
 * covariance (Croux and Haesbroeck, 1999, eq. 4.2; Pison, Van Aelst and Willems, 2002): c_α = α / F_{χ²_{p+2}}(χ²_p(α)).
 */
const consistencyFactor = (p: number, alpha: number) =>
  alpha / (regularisedGammaP((p + 2) / 2, chiSquareQuantile(p, alpha) / 2) as number)

/** Rows of a data matrix as one row-major Float64Array with its shape. */
function rowsOf(x: MatrixLike, where: string): { a: Float64Array; n: Size; p: Size } {
  const m = dense.toMatrixF64(x, where)
  return { a: Float64Array.from(m.data), n: m.m, p: m.n }
}

/** The mean and maximum-likelihood covariance (divided by the count) of the rows `idx`. */
function moments(a: Float64Array, p: Size, idx: ArrayLike<number>) {
  const k = idx.length
  const mean = new Float64Array(p)
  for (let r = 0; r < k; r++) for (let j = 0; j < p; j++) mean[j] += a[idx[r] * p + j] / k
  const cov = new Float64Array(p * p)
  for (let r = 0; r < k; r++) {
    const o = idx[r] * p
    for (let i = 0; i < p; i++) {
      const di = a[o + i] - mean[i]
      for (let j = 0; j <= i; j++) cov[i * p + j] += (di * (a[o + j] - mean[j])) / k
    }
  }
  for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) cov[j * p + i] = cov[i * p + j]
  return { mean, cov }
}

/**
 * Squared Mahalanobis distances (x − μ)ᵀ Σ⁻¹ (x − μ) of every row, and log det Σ, through the Cholesky factor of Σ.
 * A singular Σ gives log det −∞ and infinite distances off its support.
 */
function distances(a: Float64Array, n: Size, p: Size, mean: Float64Array, cov: Float64Array) {
  const ch = cholesky(fromData(cov, [p, p]), { jitter: false })
  if (ch.failed) return { d2: new Float64Array(n).fill(Infinity), logDet: -Infinity }
  const L = ch.L
  const Lf = toFlat(L)
  let logDet = 0
  for (let i = 0; i < p; i++) logDet += 2 * Math.log(Lf[i * p + i])
  const centred = new Float64Array(n * p)
  for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) centred[r * p + j] = a[r * p + j] - mean[j]
  // Solve L Z = (X − μ)ᵀ for all rows at once; d²ᵢ = ‖zᵢ‖².
  const Z = toFlat(solveTriangular(L, fromData(transpose(centred, n, p), [p, n]), { lower: true }) as Tensor)
  const d2 = new Float64Array(n)
  for (let j = 0; j < p; j++) for (let r = 0; r < n; r++) d2[r] += Z[j * n + r] ** 2
  return { d2, logDet }
}

function transpose(a: Float64Array, n: Size, p: Size): Float64Array {
  const t = new Float64Array(n * p)
  for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) t[j * n + r] = a[r * p + j]
  return t
}

/** The indices of the h smallest values. */
function smallest(d2: Float64Array, h: Size): Int32Array {
  const order = Array.from(d2.keys()).sort((i, j) => d2[i] - d2[j] || i - j)
  return Int32Array.from(order.slice(0, h)).sort()
}

/** Options of `minimumCovarianceDeterminant`. */
export type McdOptions = {
  /** The share h/n of points in the support (default ⌈(n + p + 1)/2⌉/n, the maximal breakdown point). */
  supportFraction?: number
  /** Random initial h-subsets (default 30). */
  starts?: Size
  /** Starts kept after two C-steps and iterated to convergence (default 10). */
  keep?: Size
  /** Concentration steps at most per start (default 30). */
  maxSteps?: Size
  /** The stream of the random starts (default `stream('mcd')`). */
  stream?: Stream
}

/** The minimum covariance determinant estimate. */
export type Mcd = {
  /** The reweighted location and covariance (the robust estimates). */
  location: Tensor
  covariance: Tensor
  /** The raw MCD: the mean of the h-subset with the smallest determinant, and its covariance times c_{h/n}. */
  rawLocation: Tensor
  rawCovariance: Tensor
  /** The h-subset of the raw estimate (int32 indices, ascending). */
  support: Tensor
  /** 1 for the points kept by the reweighting step (squared distance below the χ²_p 97.5% quantile), else 0. */
  inliers: Tensor
  /** Squared Mahalanobis distances of every point under the reweighted estimate. */
  distances: Tensor
  /** log det of the raw (uncorrected) covariance of the support. */
  logDeterminant: number
  h: Size
}

/**
 * The minimum covariance determinant estimator (Rousseeuw, 1984): the mean and covariance of the h of n points whose
 * covariance has the smallest determinant, found by FastMCD (Rousseeuw and Van Driessen, 1999). Each random h-subset is
 * improved by concentration steps: compute its mean and covariance, then keep the h points with the smallest Mahalanobis
 * distances under them; the determinant never increases, so the steps converge. The best start is then scaled to be
 * consistent at the normal (by c_α = α/F_{χ²_{p+2}}(χ²_p(α)) with α = h/n; Croux and Haesbroeck, 1999) and reweighted:
 * the mean and covariance of the points with corrected d² below the χ²_p 97.5% quantile, scaled by c_0.975, as in
 * scikit-learn's `MinCovDet`. Up to n − h outliers cannot move it far.
 */
export function minimumCovarianceDeterminant(x: MatrixLike, options: McdOptions = {}): Mcd {
  const { a, n, p } = rowsOf(x, 'minimumCovarianceDeterminant')
  const { starts = 30, keep = 10, maxSteps = 30, stream: s = makeStream('mcd') } = options
  const h = options.supportFraction ? Math.ceil(options.supportFraction * n) : Math.ceil(0.5 * (n + p + 1))
  if (h <= p || h > n)
    throw new DomainError(
      'minimumCovarianceDeterminant',
      `minimumCovarianceDeterminant: support size ${h} must lie in (p, n]`,
    )
  type Candidate = { support: Int32Array; logDet: number }
  const cstep = (support: Int32Array): Candidate & { next: Int32Array } => {
    const { mean, cov } = moments(a, p, support)
    const { d2, logDet } = distances(a, n, p, mean, cov)
    return { support, logDet, next: smallest(d2, h) }
  }
  const run = (start: Int32Array, steps: number): Candidate => {
    let current = cstep(start)
    for (let k = 0; k < steps; k++) {
      const following = cstep(current.next)
      const same = following.support.every((v, i) => v === current.support[i])
      if (same || following.logDet >= current.logDet) {
        if (following.logDet < current.logDet) current = following
        break
      }
      current = following
    }
    return { support: current.support, logDet: current.logDet }
  }
  const candidates: Candidate[] = []
  for (let t = 0; t < starts; t++) {
    const perm = toFlat(permutation(child(s, t), n))
    candidates.push(run(Int32Array.from(perm.slice(0, h)).sort(), 2))
  }
  candidates.sort((c1, c2) => c1.logDet - c2.logDet)
  let best: Candidate | null = null
  for (const c of candidates.slice(0, keep)) {
    const r = run(c.support, maxSteps)
    if (!best || r.logDet < best.logDet) best = r
  }
  const raw = moments(a, p, best!.support)
  const rawD = distances(a, n, p, raw.mean, raw.cov).d2
  const correction = consistencyFactor(p, h / n)
  const rawCov = raw.cov.map((v) => v * correction)
  const corrected = rawD.map((d) => d / correction)
  const cut = chiSquareQuantile(p, 0.975)
  const kept: number[] = []
  corrected.forEach((d, i) => d < cut && kept.push(i))
  const re = moments(a, p, kept)
  const reweighting = consistencyFactor(p, 0.975)
  re.cov.forEach((v, i) => (re.cov[i] = v * reweighting))
  const finalD = distances(a, n, p, re.mean, re.cov).d2
  const inliers = new Int32Array(n)
  for (const i of kept) inliers[i] = 1
  return {
    location: fromData(re.mean, [p]),
    covariance: fromData(re.cov, [p, p]),
    rawLocation: fromData(raw.mean, [p]),
    rawCovariance: fromData(rawCov, [p, p]),
    support: fromData(best!.support, [h]),
    inliers: fromData(inliers, [n]),
    distances: fromData(finalD, [n]),
    logDeterminant: best!.logDet,
    h,
  }
}

/** Squared Mahalanobis distances (xᵢ − μ)ᵀ Σ⁻¹ (xᵢ − μ) of the rows of x under a location and covariance. */
export function squaredMahalanobis(
  x: MatrixLike,
  location: Tensor | ArrayLike<number>,
  covariance: MatrixLike,
): Tensor {
  const { a, n, p } = rowsOf(x, 'squaredMahalanobis')
  const mean = Float64Array.from(isTensor(location) ? toFlat(location) : location)
  const cov = rowsOf(covariance, 'squaredMahalanobis').a
  if (mean.length !== p || cov.length !== p * p)
    throw new ShapeError('squaredMahalanobis', 'squaredMahalanobis: location or covariance has the wrong size')
  return fromData(distances(a, n, p, mean, cov).d2, [n])
}
