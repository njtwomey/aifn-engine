/** PCA and ZCA whitening. */

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
  readonly method: 'pca' | 'zca'
  /** Column means [d]. */
  readonly mean: Tensor
  /** Principal axes as columns [d, k], in order of decreasing variance, each signed so its largest entry is positive. */
  readonly components: Tensor
  /** Variances along the kept axes (sample covariance eigenvalues, ÷ (n − 1)), [k], descending. */
  readonly explainedVariance: Tensor
  /** The matrix W with z = (x − mean) W: [d, k] for PCA, [d, d] for ZCA. */
  readonly whitener: Tensor
  /** True when a kept eigenvalue plus `epsilon` is not above d·ε_machine·λ_max: the whitened values are then unreliable. */
  readonly singular: boolean
}

/**
 * Whitening: z = (x − x̄) W with W such that z has identity sample covariance. With C = V Λ Vᵀ the sample covariance
 * (÷ (n − 1)), PCA whitening keeps the top `components` axes, W = V_k (Λ_k + εI)^{−½} (scikit-learn's `PCA(whiten=True)`,
 * with the same component signs); ZCA whitening rotates back, W = V (Λ + εI)^{−½} Vᵀ, the whitening closest to the
 * identity (Bell and Sejnowski, 1997; Kessy, Lewin and Strimmer, 2018, "Optimal whitening and decorrelation", The
 * American Statistician 72). `inverse` maps back (exactly when all axes are kept, else to the rank-k reconstruction).
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
