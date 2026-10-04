/**
 * Shared by the embeddings of `aifn-methods/unsupervised/embedding`: double centring and the classical-MDS core.
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { fromData } from 'aifn-compute/foundation/tensor'

/**
 * Classical MDS on squared distances D² [n, n] (Torgerson, 1952; Gower, 1966): B = −½ J D² J with J = I − 11ᵀ/n,
 * and the coordinates Y = V_r Λ_r^½ from B's top r eigenpairs (negative eigenvalues give zero columns).
 */
export function classicalCore(D2: Float64Array, n: number, r: number) {
  const B = new Float64Array(n * n)
  const row = new Float64Array(n)
  let all = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) row[i] += D2[i * n + j] / n
    all += row[i] / n
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = -0.5 * (D2[i * n + j] - row[i] - row[j] + all)
  const e = eigh(fromData(B, [n, n]))
  const lambda = e.values.data as Float64Array
  const V = e.vectors.data as Float64Array
  const Y = new Float64Array(n * r)
  for (let c = 0; c < r; c++) {
    const s = Math.sqrt(Math.max(lambda[c], 0))
    for (let i = 0; i < n; i++) Y[i * r + c] = V[i * n + c] * s
  }
  return { Y, eigenvalues: Float64Array.from(lambda) }
}
