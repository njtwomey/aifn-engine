/**
 * Shared by the embeddings of `aifn-methods/unsupervised/embedding`: double centring and the classical-MDS core.
 *
 * Squared distances $\Dmat^{(2)}$ become the Gram matrix $\Bmat = -\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$ with
 * $\Jmat = \Imat - \ones\ones^\top/n$, whose top eigenpairs give the coordinates (Torgerson, 1952; Gower, 1966).
 * `classicalMds` and `isomap` both embed through it.
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { fromData } from 'aifn-compute/foundation/tensor'

/**
 * Classical MDS on squared distances (Torgerson, 1952; Gower, 1966): double-centre them into
 * $\Bmat = -\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$ with $\Jmat = \Imat - \ones\ones^\top/n$, and take the coordinates
 * $\Ymat = \Vmat_{r}\Lambdamat_{r}^{1/2}$ from $\Bmat$'s top $r$ eigenpairs. A negative eigenvalue among the top $r$
 * gives a zero column.
 *
 * @param D2 The squared distances $D_{ij}^2$ as a row-major array of $n^2$ values (read, not modified).
 * @param n The number of points.
 * @param r The number of coordinates to keep, at most $n$.
 * @returns `Y`, the coordinates as a row-major array of $n \times r$ values (point `i` in entries `i * r` to
 *   `i * r + r - 1`), and `eigenvalues`, all $n$ eigenvalues of $\Bmat$ in descending order.
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
