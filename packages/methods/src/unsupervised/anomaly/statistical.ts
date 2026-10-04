/**
 * Anomaly scores from a fitted Gaussian or linear model of the data: the squared Mahalanobis distance under the
 * classical or the robust (minimum covariance determinant) estimate of location and scatter, and the reconstruction
 * error of principal component analysis; and thresholds from a share of the data or from a peaks-over-threshold tail.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import { peaksOverThreshold, tailQuantile } from 'aifn-compute/probability/extremes'
import { minimumCovarianceDeterminant, quantile, squaredMahalanobis } from 'aifn-compute/probability/stats'

/** A Gaussian model of the data: its location and covariance, robust or classical. */
export type MahalanobisModel = { location: Tensor; covariance: Tensor; robust: boolean }

/**
 * The location and covariance of the rows of x: the sample mean and maximum-likelihood covariance, or with `robust`
 * the reweighted minimum covariance determinant (`aifn-compute/probability/stats`), which the anomalies cannot drag towards
 * themselves.
 */
export function mahalanobisModel(x: MatrixLike, { robust = true }: { robust?: boolean } = {}): MahalanobisModel {
  if (robust) {
    const m = minimumCovarianceDeterminant(x)
    return { location: m.location, covariance: m.covariance, robust }
  }
  const m = dense.toMatrixF64(x, 'mahalanobisModel')
  const { m: n, n: d, data } = m
  const mean = new Float64Array(d)
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) mean[c] += data[r * d + c] / n
  const cov = new Float64Array(d * d)
  for (let r = 0; r < n; r++)
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++) cov[a * d + b] += ((data[r * d + a] - mean[a]) * (data[r * d + b] - mean[b])) / n
  return { location: fromData(mean, [d]), covariance: fromData(cov, [d, d]), robust }
}

/** The squared Mahalanobis distance of each row of x under a model (χ²_d-distributed for Gaussian inliers). */
export function mahalanobisScore(model: MahalanobisModel, x: MatrixLike): Float64Array {
  return Float64Array.from(toFlat(squaredMahalanobis(x, model.location, model.covariance)))
}

/** A principal-component model: the mean and the leading q principal axes (columns of a d × q matrix). */
export type PcaModel = { mean: Float64Array; axes: Float64Array; d: Size; q: Size }

/** Fit the mean and the first q principal axes of the rows of x by the singular value decomposition. */
export function pcaModel(x: MatrixLike, { components = 1 }: { components?: Size } = {}): PcaModel {
  const m = dense.toMatrixF64(x, 'pcaModel')
  const { m: n, n: d } = m
  // The thin SVD has min(n, d) axes; asking for more would read past them.
  const q = Math.min(components, d, n)
  const mean = new Float64Array(d)
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) mean[c] += m.data[r * d + c] / n
  const centred = Float64Array.from(m.data, (v, i) => v - mean[i % d])
  const V = toFlat(svd(fromData(centred, [n, d])).V)
  const k = Math.min(n, d)
  const axes = new Float64Array(d * q)
  for (let c = 0; c < d; c++) for (let a = 0; a < q; a++) axes[c * q + a] = V[c * k + a]
  return { mean, axes, d, q }
}

/**
 * The squared reconstruction error ‖x − x̂‖² of each row, where x̂ is its projection on the mean plus the span of the
 * principal axes: large for points off the subspace where the data lie.
 */
export function pcaReconstructionScore(model: PcaModel, x: MatrixLike): Float64Array {
  const m = dense.toMatrixF64(x, 'pcaReconstructionScore')
  const { d, q, axes, mean } = model
  const out = new Float64Array(m.m)
  const z = new Float64Array(q)
  for (let r = 0; r < m.m; r++) {
    z.fill(0)
    for (let c = 0; c < d; c++) for (let a = 0; a < q; a++) z[a] += (m.data[r * d + c] - mean[c]) * axes[c * q + a]
    let e = 0
    for (let c = 0; c < d; c++) {
      let xhat = mean[c]
      for (let a = 0; a < q; a++) xhat += z[a] * axes[c * q + a]
      e += (m.data[r * d + c] - xhat) ** 2
    }
    out[r] = e
  }
  return out
}

/** Options of `anomalyThreshold`. */
export type ThresholdOptions = {
  /**
   * `quantile`: the (1 − risk) empirical quantile of the training scores; `pot`: the tail quantile of a generalised
   * Pareto fit above the `level` quantile (Siffer et al., 2017), which reaches beyond the largest score seen. Default
   * `quantile`.
   */
  method?: 'quantile' | 'pot'
  /** The share of points expected above the threshold (default 0.05). */
  risk?: number
  /** POT: the quantile of the scores used as the initial threshold (default 0.8). */
  level?: number
}

/** A score threshold above which points are called anomalies. */
export function anomalyThreshold(scores: ArrayLike<number>, options: ThresholdOptions = {}): number {
  const { method = 'quantile', risk = 0.05, level = 0.8 } = options
  if (method === 'quantile') return quantile(Float64Array.from(scores), 1 - risk)
  const fit = peaksOverThreshold(Float64Array.from(scores), { quantile: level })
  return tailQuantile(fit, 1 - Math.min(risk, fit.rate))
}
