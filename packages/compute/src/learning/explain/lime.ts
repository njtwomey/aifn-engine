/**
 * LIME for tabular data (Ribeiro, Singh and Guestrin, 2016): a local linear surrogate, without LIME's feature
 * selection. Samples $\zvec = \xvec + \sigmavec \odot \epsilonvec$ around the instance ($\sigmavec$ the per-feature
 * scale, $\epsilonvec$ standard normal; the first sample is $\xvec$ itself), weights each by the paper's exponential
 * kernel $\exp(-\lVert (\zvec - \xvec)/\sigmavec \rVert^2 / w^2)$ of its standardised distance (width $w$ default
 * $0.75\sqrt{d}$, as the `lime` package, whose kernel is the square root of this one: pass $w\sqrt{2}$ to match its
 * weights), and fits a weighted ridge regression of the model's outputs on the standardised offsets
 * $(\zvec - \xvec)/\sigmavec$. The coefficients are the local attributions per standard deviation of each feature;
 * `score` is the surrogate's weighted $R^2$.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import type { ScalarModel } from './shapley'

/** Options of `lime`. */
export type LimeOptions = {
  /** The number of perturbed samples, the instance itself included (default 1000). */
  samples?: Size
  /**
   * Per-feature scale $\sigmavec$ ($d$ values; default the population standard deviation of `data`'s columns, a
   * constant column taking 1, or 1 for every feature when there is no `data`).
   */
  scale?: VectorLike
  /** Rows to take the scale from when `scale` is not given. */
  data?: MatrixLike
  /** Kernel width $w$, in standard deviations (default $0.75\sqrt{d}$). */
  width?: number
  /** Ridge penalty on the coefficients, not the intercept (default 1, as the `lime` package). */
  ridge?: number
}

/** A local linear surrogate. */
export type LimeExplanation = {
  /** Coefficient per feature ($d$ values): the change in output per standard deviation $\sigma_i$ of feature $i$. */
  coefficients: Float64Array
  /** The surrogate's intercept: its prediction at $\xvec$, where every offset is 0. */
  intercept: number
  /** The surrogate's prediction at $\xvec$ (the intercept), to compare with `output`. */
  localPrediction: number
  /** The model's output at $\xvec$. */
  output: number
  /** Weighted $R^2$ of the surrogate on the samples (1 when the outputs do not vary). */
  score: number
  /** The perturbed samples ($\text{samples} \times d$, the first being $\xvec$), for display. */
  samples: Tensor
  /** The kernel weight of each sample, for display. */
  weights: Float64Array
}

/**
 * A LIME explanation of `model` at $\xvec$ (see the file comment): one batched call of the model on all the samples,
 * then a weighted ridge fit.
 *
 * @param model The model, called once on the $\text{samples} \times d$ batch of perturbed rows.
 * @param x The instance $\xvec$ to explain ($d$ values).
 * @param stream The random stream the perturbations are drawn from.
 * @param options The sample count, scale, kernel width and ridge penalty.
 * @returns The local surrogate, with the samples and their weights.
 *
 * @example A linear model's surrogate recovers its weights
 * const model = (X) => matmul(X, tensor([2, -1, 0.5]))
 * const e = lime(model, [1, 2, 3], stream(0), { samples: 200 })
 * print('coefficients =', e.coefficients)
 * print('local prediction =', e.localPrediction, ' output =', e.output, ' R2 =', e.score)
 *
 * @example Coefficients are per standard deviation of the data
 * const model = (X) => matmul(X, tensor([2, 2]))
 * const data = [[0, 0], [2, 20], [4, 40]]
 * const e = lime(model, [2, 20], stream(1), { samples: 200, data })
 * print('w times sd =', [2 * Math.sqrt(8 / 3), 2 * Math.sqrt(800 / 3)])
 * print('coefficients =', e.coefficients)
 * print('without the ridge:', lime(model, [2, 20], stream(1), { samples: 200, data, ridge: 0 }).coefficients)
 */
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
