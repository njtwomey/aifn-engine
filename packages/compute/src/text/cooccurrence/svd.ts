/**
 * The truncated singular value decomposition A ≈ U_k diag(S_k) V_kᵀ of a word × context or term × document matrix: the
 * best rank-k approximation in Frobenius norm (Eckart & Young 1936), the factorisation behind latent semantic analysis
 * (Deerwester et al. 1990) and SVD word vectors (Levy, Goldberg & Dagan 2015). Small matrices use the full SVD; larger
 * ones find the top k eigenpairs of the Gram matrix of the shorter side by Lanczos (`eigsh`), touching A only through
 * products, and recover the other side's vectors by one product with A.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, toFlat, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigsh, svd } from 'aifn-compute/numerics/linalg'

/** A truncated SVD with the share of the squared Frobenius norm each component carries. */
export interface TruncatedSvd {
  readonly kind: 'truncated-svd'
  /** Left singular vectors as columns (float64 [m, k]). */
  readonly U: Tensor
  /** The k largest singular values, descending (float64 [k]). */
  readonly S: Tensor
  /** Right singular vectors as columns (float64 [n, k]). */
  readonly V: Tensor
  /** σᵢ² / ‖A‖²_F: the share of the matrix's energy in component i (float64 [k]); the shares of all components sum to 1. */
  readonly energy: Tensor
  /** ‖A‖²_F, the sum of all squared singular values. */
  readonly total: number
}

/** Below this shorter side the full SVD is cheap and exact. */
const FULL = 64

/**
 * The k largest singular triplets of an m × n matrix. Each pair of singular vectors is signed so that the
 * largest-magnitude entry of its left vector is positive (scikit-learn's `svd_flip`), so results do not depend on the
 * method. A right (left) vector of a zero singular value is left zero when found from the Gram matrix of the other side.
 */
export function truncatedSvd(matrix: MatrixLike, k: number): TruncatedSvd {
  const { data, m, n } = dense.toMatrixF64(matrix, 'truncatedSvd')
  const s = Math.min(m, n)
  if (!(Number.isInteger(k) && k >= 1 && k <= s))
    throw new DomainError('truncatedSvd', `truncatedSvd: k must be an integer in [1, ${s}]`)
  let total = 0
  for (const x of data) total += x * x
  const U = new Float64Array(m * k)
  const V = new Float64Array(n * k)
  const S = new Float64Array(k)
  if (s <= FULL || k >= s - 1) {
    const full = svd(fromData(data, [m, n]))
    const [u, sv, v] = [toFlat(full.U), toFlat(full.S), toFlat(full.V)]
    for (let j = 0; j < k; j++) {
      S[j] = sv[j]
      for (let i = 0; i < m; i++) U[i * k + j] = u[i * s + j]
      for (let i = 0; i < n; i++) V[i * k + j] = v[i * s + j]
    }
  } else {
    // Eigenpairs of the Gram matrix of the shorter side: A Aᵀ (m ≤ n) gives U, Aᵀ A gives V.
    const rows = m <= n
    const op = (x: Tensor) => {
      const v = x.data as Float64Array
      if (rows) {
        const t = new Float64Array(n)
        for (let i = 0; i < m; i++) if (v[i] !== 0) for (let j = 0; j < n; j++) t[j] += data[i * n + j] * v[i]
        return fromData(dense.matVec(data, t, m, n))
      }
      const t = dense.matVec(data, v, m, n)
      const out = new Float64Array(n)
      for (let i = 0; i < m; i++) if (t[i] !== 0) for (let j = 0; j < n; j++) out[j] += data[i * n + j] * t[i]
      return fromData(out)
    }
    const { values, vectors } = eigsh(op, s, { k })
    const lam = toFlat(values)
    const vec = toFlat(vectors)
    for (let j = 0; j < k; j++) {
      const sigma = Math.sqrt(Math.max(lam[j], 0))
      S[j] = sigma
      const [known, other, a, b] = rows ? [U, V, m, n] : [V, U, n, m]
      for (let i = 0; i < a; i++) known[i * k + j] = vec[i * k + j]
      if (sigma === 0) continue
      // The other side: Aᵀu / σ or A v / σ.
      for (let i = 0; i < b; i++) {
        let acc = 0
        for (let r = 0; r < a; r++) acc += (rows ? data[r * n + i] : data[i * n + r]) * known[r * k + j]
        other[i * k + j] = acc / sigma
      }
    }
  }
  for (let j = 0; j < k; j++) {
    let best = 0
    for (let i = 0; i < m; i++) if (Math.abs(U[i * k + j]) > Math.abs(best)) best = U[i * k + j]
    if (best < 0) {
      for (let i = 0; i < m; i++) U[i * k + j] = -U[i * k + j]
      for (let i = 0; i < n; i++) V[i * k + j] = -V[i * k + j]
    }
  }
  return {
    kind: 'truncated-svd',
    U: fromData(U, [m, k]),
    S: fromData(S),
    V: fromData(V, [n, k]),
    energy: fromData(S.map((x) => (total > 0 ? (x * x) / total : 0))),
    total,
  }
}
