/**
 * PCA and ZCA whitening: a linear map $\zvec = \Wmat^\top (\xvec - \bar{\xvec})$ of each row to uncorrelated features
 * of unit variance.
 *
 * Both come from the eigendecomposition $\Cmat = \Vmat \Lambdamat \Vmat^\top$ of the sample covariance (divided by
 * $n - 1$). PCA whitening projects onto the leading eigenvectors and rescales each (scikit-learn's `PCA(whiten=True)`);
 * ZCA whitening rotates back to the original axes, giving the whitened data closest to the input (Bell and Sejnowski,
 * 1997; Kessy, Lewin and Strimmer, 2018, "Optimal whitening and decorrelation", The American Statistician 72).
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkColumns, matrix, values, type FittedTransform, type Invertible, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A fitted whitening transform. */
export interface Whitening extends FittedTransform, Invertible {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'whitening'
  /** `'pca'` (project onto the leading axes) or `'zca'` (rotate back to the input axes). */
  readonly method: 'pca' | 'zca'
  /** The column means $\bar{\xvec}$, $d$ values. */
  readonly mean: Tensor
  /**
   * The principal axes as columns, $d \times k$, in order of decreasing variance, each signed (as `eigh` signs them)
   * so that its entry of largest magnitude is positive.
   */
  readonly components: Tensor
  /** The variances along the kept axes (sample covariance eigenvalues, divided by $n - 1$), $k$ values, descending. */
  readonly explainedVariance: Tensor
  /**
   * The matrix $\Wmat$ with $\zvec^\top = (\xvec - \bar{\xvec})^\top \Wmat$ for each row: $d \times k$ for PCA,
   * $d \times d$ for ZCA.
   */
  readonly whitener: Tensor
  /**
   * True when a kept eigenvalue plus `epsilon` is not above $d \varepsilon \lambda_{\max}$ ($\varepsilon$ the machine
   * epsilon): the whitened values are then unreliable.
   */
  readonly singular: boolean
}

/**
 * Whitening: each row becomes $\zvec^\top = (\xvec - \bar{\xvec})^\top \Wmat$, with $\Wmat$ such that the whitened
 * training rows have identity sample covariance. With $\Cmat = \Vmat \Lambdamat \Vmat^\top$ the sample covariance
 * (divided by $n - 1$), PCA whitening keeps the top `components` axes,
 * $\Wmat = \Vmat_k (\Lambdamat_k + \epsilon \Imat)^{-1/2}$ (scikit-learn's `PCA(whiten=True)`, with the same component
 * signs); ZCA whitening rotates back, $\Wmat = \Vmat (\Lambdamat + \epsilon \Imat)^{-1/2} \Vmat^\top$, the whitening
 * closest to the identity (Bell and Sejnowski, 1997; Kessy, Lewin and Strimmer, 2018, "Optimal whitening and
 * decorrelation", The American Statistician 72). `inverseTransform` maps back (exactly when all axes are kept, else to
 * the rank-$k$ reconstruction). `fit` throws `DomainError` for fewer than two rows; a covariance too close to singular
 * is reported in `singular`, not thrown.
 *
 * @param options The kind of whitening, how many axes to keep, and the regularisation.
 * @param options.method `'pca'` (the $k$ leading axes, as rotated coordinates) or `'zca'` (all $d$ axes, rotated back
 *   to the input's).
 * @param options.components The number $k$ of axes PCA keeps (all $d$ when left out, and at most $d$); ignored by ZCA.
 * @param options.epsilon The $\epsilon \ge 0$ added to every eigenvalue before the inverse square root, which bounds
 *   the scaling of low-variance axes (the output covariance is then $\Lambdamat (\Lambdamat + \epsilon \Imat)^{-1}$,
 *   not $\Imat$).
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `Whitening`.
 *
 * @example PCA whitening of correlated points gives an identity covariance
 * const x = matmul(normal(stream(1), 0, 1, { shape: [300, 2] }), tensor([[2, 1], [0, 0.5]]))
 * const model = whitening().fit({ x })
 * print('explained variance =', model.explainedVariance)
 * const z = model.transform(x)
 * print('cov(z) =', div(matmul(transpose(z), z), 299))
 *
 * @example ZCA's whitener is symmetric, PCA's is not; the inverse recovers the inputs
 * const x = matmul(normal(stream(2), 0, 1, { shape: [300, 2] }), tensor([[2, 1], [0, 0.5]]))
 * const zca = whitening({ method: 'zca' }).fit({ x })
 * print('ZCA whitener =', zca.whitener)
 * print('PCA whitener =', whitening().fit({ x }).whitener)
 * print('first row:', slice(x, 0), ' back:', slice(zca.inverseTransform(zca.transform(x)), 0))
 */
export function whitening({
  method = 'pca',
  components,
  epsilon = 0,
}: { method?: 'pca' | 'zca'; components?: number; epsilon?: number } = {}): Transformer<Tensor, Whitening> {
  return {
    name: 'whitening',
    params: { method, components, epsilon },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'whitening')
      if (n < 2) throw new DomainError('whitening', 'whitening: needs at least two rows')
      const k = method === 'zca' ? d : Math.min(components ?? d, d)
      const mean = new Float64Array(d)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += v[i * d + j] / n
      const C = new Float64Array(d * d)
      for (let i = 0; i < n; i++) {
        for (let a = 0; a < d; a++) {
          const xa = v[i * d + a] - mean[a]
          for (let b = 0; b <= a; b++) C[a * d + b] += (xa * (v[i * d + b] - mean[b])) / (n - 1)
        }
      }
      const { values: lambda, vectors } = eigh(fromData(C, [d, d]))
      const L = values(lambda)
      const V = values(vectors)
      const Vk = new Float64Array(d * k)
      for (let a = 0; a < d; a++) for (let c = 0; c < k; c++) Vk[a * k + c] = V[a * d + c]
      const inv = Float64Array.from({ length: k }, (_, c) => 1 / Math.sqrt(L[c] + epsilon))
      const sqrt = Float64Array.from({ length: k }, (_, c) => Math.sqrt(L[c] + epsilon))
      const floor = d * Number.EPSILON * Math.max(L[0], 0)
      const singular = Array.from(L.subarray(0, k)).some((l) => !(l + epsilon > floor))
      // W and its left inverse M (x − x̄ = z M on the kept subspace).
      const W = new Float64Array(d * (method === 'zca' ? d : k))
      const M = new Float64Array((method === 'zca' ? d : k) * d)
      if (method === 'pca') {
        for (let a = 0; a < d; a++) for (let c = 0; c < k; c++) W[a * k + c] = Vk[a * k + c] * inv[c]
        for (let c = 0; c < k; c++) for (let a = 0; a < d; a++) M[c * d + a] = sqrt[c] * Vk[a * k + c]
      } else {
        for (let a = 0; a < d; a++) {
          for (let b = 0; b < d; b++) {
            let w = 0
            let m = 0
            for (let c = 0; c < d; c++) {
              w += V[a * d + c] * inv[c] * V[b * d + c]
              m += V[a * d + c] * sqrt[c] * V[b * d + c]
            }
            W[a * d + b] = w
            M[a * d + b] = m
          }
        }
      }
      const out = method === 'zca' ? d : k
      const multiply = (input: Float64Array, rows: number, A: Float64Array, inner: number, cols: number) => {
        const r = new Float64Array(rows * cols)
        for (let i = 0; i < rows; i++)
          for (let a = 0; a < inner; a++) {
            const e = input[i * inner + a]
            if (e !== 0) for (let c = 0; c < cols; c++) r[i * cols + c] += e * A[a * cols + c]
          }
        return r
      }
      return {
        kind: 'model',
        name: 'whitening',
        method,
        mean: fromData(mean, [d]),
        components: fromData(Vk, [d, k]),
        explainedVariance: fromData(L.slice(0, k), [k]),
        whitener: fromData(W, [d, out]),
        singular,
        transform(input) {
          const { n: rows, d: cols, v: z } = matrix(input, 'whitening.transform')
          checkColumns(cols, d, 'whitening')
          const centred = Float64Array.from(z, (e, idx) => e - mean[idx % d])
          return fromData(multiply(centred, rows, W, d, out), [rows, out])
        },
        inverseTransform(z) {
          const { n: rows, d: cols, v: w } = matrix(z, 'whitening.inverseTransform')
          checkColumns(cols, out, 'whitening.inverseTransform')
          const x = multiply(w, rows, M, out, d)
          for (let idx = 0; idx < x.length; idx++) x[idx] += mean[idx % d]
          return fromData(x, [rows, d])
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'whitening',
    module: 'learning/preprocessing',
    name: 'Whitening',
    summary: 'A linear map to identity covariance (PCA or ZCA whitening).',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ method: oneOf(['pca', 'zca']), epsilon: real(0, 1, { default: 0, label: 'ε' }) }),
    notes: ['principal-component-analysis'],
  },
  whitening,
)
