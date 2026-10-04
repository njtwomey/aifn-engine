/**
 * Gram–Schmidt orthogonalisation, classical and modified, as a traceable algorithm (Golub and Van Loan, 2013, §5.2.7–
 * 5.2.8; Björck, 1967): one column per step, so a figure can show each projection being removed. `gramSchmidt` runs
 * the steps to the end.
 */

import type { MatrixLike, Status } from 'aifn-compute/foundation/contracts'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { EPS } from './dense'

/**
 * `classical` projects the original column $\avec_j$ on every earlier $\qvec_i$ ($r_{ij} = \qvec_i^\top \avec_j$);
 * `modified` projects the running remainder ($r_{ij} = \qvec_i^\top \vvec$, then
 * $\vvec \leftarrow \vvec - r_{ij} \qvec_i$), which loses orthogonality in proportion to $\kappa(\Amat)$ rather than
 * $\kappa(\Amat)^2$.
 */
export type GramSchmidtVariant = 'classical' | 'modified'

/** Options of `gramSchmidtSteps` and `gramSchmidt`. */
export type GramSchmidtOptions = {
  /** Default `modified`. */
  variant?: GramSchmidtVariant
  /**
   * A remainder with $\lVert \vvec \rVert \le \text{tolerance} \cdot \lVert \avec_j \rVert$ counts as dependent:
   * $\qvec_j = \zeros$, $r_{jj} = 0$ and `rankDeficient` is set. Default $n \cdot \varepsilon$ ($n$ the number of
   * columns).
   */
  tolerance?: number
}

/** One state of `gramSchmidtSteps`. */
export interface GramSchmidtState extends Status {
  /** Orthonormal columns $\qvec_0, \dots, \qvec_{\text{column} - 1}$; later columns are zero ($m \times n$). */
  Q: Tensor
  /**
   * The coefficients $r_{ij} = \qvec_i^\top (\cdot)$ found so far, upper triangular ($n \times n$);
   * $\Amat = \Qmat\Rmat$ once done.
   */
  R: Tensor
  /** The remainder $\vvec$ of the last column before normalisation (length $m$; zeros at $t = 0$). */
  remainder: Tensor
  /** The next column to orthogonalise; $n$ when done. */
  column: number
  /** $\max |\qvec_i^\top \qvec_j - \delta_{ij}|$ over the columns so far: the loss of orthogonality. */
  orthogonalityError: number
  /** True once a column was found (numerically) dependent on the earlier ones. */
  rankDeficient: boolean
  /** True once every column has been processed. */
  done: boolean
}

/**
 * Gram–Schmidt as a traceable algorithm on the columns of an $m \times n$ matrix $\Amat$. Step $j$ removes from column
 * $\avec_j$ its projections on $\qvec_0, \dots, \qvec_{j-1}$ (all from $\avec_j$ for `classical`, one after another
 * from the remainder for `modified`), sets $r_{jj} = \lVert \vvec \rVert$ and $\qvec_j = \vvec / r_{jj}$. After $n$
 * steps $\Amat = \Qmat\Rmat$ with $\Qmat$'s columns orthonormal (up to rounding) and $\Rmat$ upper triangular with a
 * non-negative diagonal: the thin QR factorisation, with $\Rmat$'s signs fixed positive.
 *
 * @param A The matrix $\Amat$ whose columns are orthogonalised, $m \times n$, as a tensor or nested rows of numbers.
 *   It is read once and not modified.
 * @param options The variant to run (`modified` by default) and the tolerance below which a column counts as
 *   dependent on the earlier ones (by default $n \cdot \varepsilon$).
 * @returns The algorithm, which takes no input: run it with `run(alg, undefined, steps)`. Each state holds `Q` and
 *   `R` after one more column, the `remainder` of that column, and the loss of orthogonality so far.
 *
 * @example Orthogonalise one column at a time
 * const A = tensor([[1, 1], [1, 0], [0, 1]])
 * const alg = gramSchmidtSteps(A)
 * const first = run(alg, undefined, 1)
 * print('after one column, Q =', first.Q)
 * const last = run(alg, undefined, 10)
 * print('finished =', last.done)
 * print('Q =', last.Q)
 */
export function gramSchmidtSteps(A: MatrixLike, options: GramSchmidtOptions = {}): Algorithm<void, GramSchmidtState> {
  const { data: a, m, n } = dense.toMatrixF64(A, 'gramSchmidtSteps')
  const variant = options.variant ?? 'modified'
  const tolerance = options.tolerance ?? Math.max(n, 1) * EPS
  const column = (x: Float64Array, cols: number, j: number) =>
    Float64Array.from({ length: m }, (_, i) => x[i * cols + j])
  const orthogonality = (q: Float64Array, upto: number) => {
    let worst = 0
    for (let i = 0; i < upto; i++)
      for (let j = i; j < upto; j++) {
        let s = 0
        for (let r = 0; r < m; r++) s += q[r * n + i] * q[r * n + j]
        worst = Math.max(worst, Math.abs(s - (i === j ? 1 : 0)))
      }
    return worst
  }
  return {
    name: `gram-schmidt-${variant}`,
    init: () => ({
      t: 0,
      Q: dense.mat(new Float64Array(m * n), m, n),
      R: dense.mat(new Float64Array(n * n), n, n),
      remainder: dense.vec(new Float64Array(m)),
      column: 0,
      orthogonalityError: 0,
      rankDeficient: false,
      done: n === 0,
    }),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const j = s.column
      const q = Float64Array.from(dense.data(s.Q))
      const r = Float64Array.from(dense.data(s.R))
      const aj = column(a, n, j)
      const v = Float64Array.from(aj)
      for (let i = 0; i < j; i++) {
        const qi = column(q, n, i)
        const rij = dense.dot(qi, variant === 'classical' ? aj : v)
        r[i * n + j] = rij
        for (let k = 0; k < m; k++) v[k] -= rij * qi[k]
      }
      const norm = dense.norm(v)
      const dependent = norm <= tolerance * Math.max(dense.norm(aj), Number.MIN_VALUE)
      r[j * n + j] = dependent ? 0 : norm
      if (!dependent) for (let k = 0; k < m; k++) q[k * n + j] = v[k] / norm
      return {
        t: s.t + 1,
        Q: dense.mat(q, m, n),
        R: dense.mat(r, n, n),
        remainder: dense.vec(v),
        column: j + 1,
        orthogonalityError: orthogonality(q, j + 1),
        rankDeficient: s.rankDeficient || dependent,
        done: j + 1 === n,
      }
    },
    done: (s) => s.done,
  }
}

/** The result of `gramSchmidt`. */
export type GramSchmidt = {
  /** Orthonormal columns ($m \times n$); a dependent column is zero. */
  Q: Tensor
  /** Upper triangular with a non-negative diagonal ($n \times n$); $\Amat = \Qmat\Rmat$. */
  R: Tensor
  /** $\max |\qvec_i^\top \qvec_j - \delta_{ij}|$ over the columns: the loss of orthogonality. */
  orthogonalityError: number
  /** True when a column was found (numerically) dependent on the earlier ones. */
  rankDeficient: boolean
}

/**
 * The thin QR factorisation $\Amat = \Qmat\Rmat$ of an $m \times n$ matrix by Gram–Schmidt: `gramSchmidtSteps` run
 * to the end.
 *
 * @param A The matrix $\Amat$ to factor, $m \times n$, as a tensor or nested rows of numbers. It is not modified.
 * @param options The variant to run (`modified` by default) and the tolerance below which a column counts as
 *   dependent on the earlier ones (by default $n \cdot \varepsilon$).
 * @returns The factors `Q` ($m \times n$) and `R` ($n \times n$), with `orthogonalityError` (the largest entry of
 *   $|\Qmat^\top\Qmat - \Imat|$, the loss of orthogonality) and `rankDeficient` (true when some column was dependent
 *   on the earlier ones, its column of `Q` then being zero).
 *
 * @example An orthonormal basis for the columns
 * const A = tensor([[1, 1], [1, 0], [0, 1]])
 * const { Q, R, orthogonalityError } = gramSchmidt(A)
 * print('Q =', Q)
 * print('R =', R)
 * print('Qᵀ Q =', matmul(transpose(Q), Q))
 * print('orthogonality error =', orthogonalityError)
 */
export function gramSchmidt(A: MatrixLike, options: GramSchmidtOptions = {}): GramSchmidt {
  const alg = gramSchmidtSteps(A, options)
  const s = run(alg, undefined, dense.toMatrixF64(A, 'gramSchmidt').n + 1)
  return { Q: s.Q, R: s.R, orthogonalityError: s.orthogonalityError, rankDeficient: s.rankDeficient }
}
