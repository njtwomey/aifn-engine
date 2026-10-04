/**
 * LIME for tabular data (Ribeiro, Singh and Guestrin, 2016): a sparse-free local surrogate. Samples z = x + σ ⊙ ε
 * around the instance (σ the per-feature scale, ε standard normal), weights each by the paper's exponential kernel
 * exp(−‖(z − x)/σ‖² / w²) of its standardised distance (width w default 0.75√d, as the `lime` package, whose kernel is
 * the square root of this one: pass w√2 to match its weights), and fits a
 * weighted ridge regression of the model's outputs on the standardised offsets (z − x)/σ. The coefficients are the
 * local attributions per standard deviation of each feature; `score` is the surrogate's weighted R².
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import type { ScalarModel } from './shapley'

/** Options of `lime`. */
export type LimeOptions = {
  /** Perturbed samples (default 1000). */
  samples?: Size
  /** Per-feature scale σ [d] (default the standard deviation of `data`'s columns, or 1). */
  scale?: VectorLike
  /** Rows to take the scale from when `scale` is not given. */
  data?: MatrixLike
  /** Kernel width w (default 0.75√d). */
  width?: number
  /** Ridge penalty on the coefficients (default 1, as the `lime` package). */
  ridge?: number
}

/** A local linear surrogate. */
export type LimeExplanation = {
  /** Coefficient per feature [d], per standard deviation σᵢ. */
  coefficients: Float64Array
  intercept: number
  /** The surrogate's prediction at x (the intercept) against the model's. */
  localPrediction: number
  output: number
  /** Weighted R² of the surrogate on the samples. */
  score: number
  /** The perturbed samples [samples, d] and their kernel weights, for display. */
  samples: Tensor
  weights: Float64Array
}

/** A LIME explanation of `model` at x [d], perturbing from `stream`. */
export function lime(model: ScalarModel, x: VectorLike, stream: Stream, options: LimeOptions = {}): LimeExplanation {
  const xv = dense.toF64(x, 'lime')
  const d = xv.length
  const { samples: m = 1000, ridge = 1 } = options
  const width = options.width ?? 0.75 * Math.sqrt(d)
  let sigma: Float64Array
  if (options.scale) sigma = Float64Array.from(dense.toF64(options.scale, 'lime'))
  else if (options.data) {
    const D = dense.toMatrixF64(options.data, 'lime')
    sigma = new Float64Array(d)
    for (let j = 0; j < d; j++) {
      let mean = 0
      for (let i = 0; i < D.m; i++) mean += D.data[i * d + j] / D.m
      let s2 = 0
      for (let i = 0; i < D.m; i++) s2 += (D.data[i * d + j] - mean) ** 2 / D.m
      sigma[j] = Math.sqrt(s2) || 1
    }
  } else sigma = new Float64Array(d).fill(1)
  const eps = toFlat(normal(stream, 0, 1, { shape: [m * d] }))
  const Z = new Float64Array(m * d)
  const weights = new Float64Array(m)
  // The first sample is x itself, as in the lime package.
  for (let k = 0; k < m; k++) {
    let dist = 0
    for (let j = 0; j < d; j++) {
      const e = k === 0 ? 0 : eps[k * d + j]
      Z[k * d + j] = xv[j] + sigma[j] * e
      dist += e * e
    }
    weights[k] = Math.exp(-dist / (width * width))
  }
  const out = model(fromData(Z, [m, d]))
  const y = 'shape' in out ? dense.data(out as Tensor) : Float64Array.from(out)
  // Weighted ridge on [1, (z − x)/σ], the intercept unpenalised.
  const p = d + 1
  const A = new Float64Array(p * p)
  const b = new Float64Array(p)
  const row = new Float64Array(p)
  for (let k = 0; k < m; k++) {
    row[0] = 1
    for (let j = 0; j < d; j++) row[j + 1] = (Z[k * d + j] - xv[j]) / sigma[j]
    const w = weights[k]
    for (let a = 0; a < p; a++) {
      b[a] += w * row[a] * y[k]
      for (let c = 0; c < p; c++) A[a * p + c] += w * row[a] * row[c]
    }
  }
  for (let a = 1; a < p; a++) A[a * p + a] += ridge
  const beta = dense.data(solve(fromData(A, [p, p]), fromData(b, [p])) as Tensor)
  let sw = 0
  let mean = 0
  for (let k = 0; k < m; k++) {
    sw += weights[k]
    mean += weights[k] * y[k]
  }
  mean /= sw
  let ssr = 0
  let sst = 0
  for (let k = 0; k < m; k++) {
    let fit = beta[0]
    for (let j = 0; j < d; j++) fit += beta[j + 1] * ((Z[k * d + j] - xv[j]) / sigma[j])
    ssr += weights[k] * (y[k] - fit) ** 2
    sst += weights[k] * (y[k] - mean) ** 2
  }
  return {
    coefficients: Float64Array.from(beta.slice(1)),
    intercept: beta[0],
    localPrediction: beta[0],
    output: y[0],
    score: sst > 0 ? 1 - ssr / sst : 1,
    samples: fromData(Z, [m, d]),
    weights,
  }
}
