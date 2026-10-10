/**
 * Anomaly scores from a fitted Gaussian or linear model of the data: the squared Mahalanobis distance
 * $(\xvec - \muvec)^\top \Sigmamat^{-1} (\xvec - \muvec)$ under the classical or the robust (minimum covariance
 * determinant) estimate of location and scatter, and the reconstruction error of principal component analysis; and
 * thresholds from a share of the data or from a peaks-over-threshold tail. Each model is fitted once and then scores
 * any points.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import { peaksOverThreshold, tailQuantile } from 'aifn-compute/probability/extremes'
import { minimumCovarianceDeterminant, quantile, squaredMahalanobis } from 'aifn-compute/probability/stats'

/**
 * A Gaussian model of the data: its `location` $\muvec$ ($d$ values) and `covariance` $\Sigmamat$ ($d \times d$),
 * and whether they are the `robust` (MCD) estimate or the classical one.
 */
export type MahalanobisModel = { location: Tensor; covariance: Tensor; robust: boolean }

/**
 * The location and covariance of the rows of `x`: the sample mean and maximum-likelihood covariance (divided by $n$),
 * or with `robust` the reweighted minimum covariance determinant (`aifn-compute/probability/stats`, with its default
 * options and stream), which the anomalies cannot drag towards themselves, as scikit-learn's `MinCovDet`. The robust
 * estimate throws `DomainError` unless $n > d$.
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options The estimate to use.
 * @param options.robust The MCD estimate (default) rather than the classical mean and covariance.
 * @returns The model, to score points with `mahalanobisScore`.
 *
 * @example A gross outlier drags the classical estimate but not the robust one
 * const x = concat([normals(stream(0), [30, 2]), tensor([[20, 20]])])
 * print('classical location', mahalanobisModel(x, { robust: false }).location)
 * print('robust location', mahalanobisModel(x).location)
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

/**
 * The squared Mahalanobis distance of each row of `x` under a model ($\chi^2_d$-distributed for Gaussian inliers).
 * Throws `ShapeError` when the columns do not match the model; a covariance that is not positive definite gives every
 * distance infinite.
 *
 * @param model The model, as `mahalanobisModel` returns it.
 * @param x The points to score, $m \times d$: nested arrays or a rank-2 tensor.
 * @returns The squared distance of each row ($m$ values).
 *
 * @example The outlier is far under the robust model
 * const x = concat([normals(stream(0), [30, 2]), tensor([[4, 4]])])
 * const s = mahalanobisScore(mahalanobisModel(x), x)
 * print('outlier', s[30])
 * print('largest of the rest', Math.max(...s.slice(0, 30)))
 */
export function mahalanobisScore(model: MahalanobisModel, x: MatrixLike): Float64Array {
  return Float64Array.from(toFlat(squaredMahalanobis(x, model.location, model.covariance)))
}

/**
 * A principal-component model: the `mean` ($d$ values) and the leading $q$ principal `axes` (the orthonormal columns of
 * a $d \times q$ matrix, row-major), with the sizes `d` and `q`.
 */
export type PcaModel = { mean: Float64Array; axes: Float64Array; d: Size; q: Size }

/**
 * Fit the mean and the first $q$ principal axes of the rows of `x` by the singular value decomposition of the centred
 * data.
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options The size of the subspace.
 * @param options.components The number of axes $q$ (default 1), lowered to $\min(d, n)$.
 * @returns The model, to score points with `pcaReconstructionScore`.
 *
 * @example The axis of points along a line
 * const model = pcaModel([[0, 0], [1, 1], [2, 2], [3, 3]])
 * print('mean', model.mean)
 * print('axis', model.axes)
 */
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
 * The squared reconstruction error $\lVert \xvec - \hat{\xvec} \rVert^2$ of each row, where $\hat{\xvec}$ is its
 * projection on the mean plus the span of the principal axes: large for points off the subspace where the data lie.
 *
 * @param model The model, as `pcaModel` returns it.
 * @param x The points to score, $m \times d$: nested arrays or a rank-2 tensor (its columns are not checked against
 *   the model's $d$).
 * @returns The squared error of each row ($m$ values).
 *
 * @example On the line and off it
 * const model = pcaModel([[0, 0], [1, 1], [2, 2], [3, 3]])
 * print('errors', pcaReconstructionScore(model, [[4, 4.5], [0, 2]]))
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
   * `quantile`: the $(1 - \text{risk})$ empirical quantile of the training scores (linear interpolation); `pot`: the
   * tail quantile of a generalised Pareto fit above the `level` quantile (Siffer et al., 2017), which reaches beyond
   * the largest score seen. Default `quantile`.
   */
  method?: 'quantile' | 'pot'
  /** The share of points expected above the threshold (default 0.05). */
  risk?: number
  /** POT: the quantile of the scores used as the initial threshold (default 0.8). */
  level?: number
}

/**
 * A score threshold above which points are called anomalies. With `pot`, a risk larger than the share of scores above
 * the initial threshold gives that threshold back; the fit throws `DomainError` for fewer than three scores or fewer
 * than two above the initial threshold.
 *
 * @param scores The training scores, higher meaning more anomalous.
 * @param options The method, the risk and, for `pot`, the level of the initial threshold.
 * @returns The threshold.
 *
 * @example The 95% quantile and a peaks-over-threshold tail
 * const s = toArray(abs(normals(stream(0), 200)))
 * print('quantile', anomalyThreshold(s))
 * print('pot at risk 0.01', anomalyThreshold(s, { method: 'pot', risk: 0.01 }))
 */
export function anomalyThreshold(scores: ArrayLike<number>, options: ThresholdOptions = {}): number {
  const { method = 'quantile', risk = 0.05, level = 0.8 } = options
  if (method === 'quantile') return quantile(Float64Array.from(scores), 1 - risk)
  const fit = peaksOverThreshold(Float64Array.from(scores), { quantile: level })
  return tailQuantile(fit, 1 - Math.min(risk, fit.rate))
}
