/**
 * Laplacian regularised least squares (LapRLS), the squared-loss case of manifold regularisation (Belkin, Niyogi and
 * Sindhwani, 2006, "Manifold regularization", JMLR 7, §4.2, eq. 8).
 *
 * Over $n = l + u$ points with Gram matrix $\Kmat$ ($n \times n$) and graph Laplacian $\Lmat$ ($n \times n$), the
 * function $f = \sum_i \alpha_i k(\xvec_i, \cdot)$ minimises
 *
 * $$
 *   \frac{1}{l}\sum_{i \in L} (y_i - f(\xvec_i))^2 + \gamma_A \lVert f \rVert_{\mathcal{H}}^2
 *   + \frac{\gamma_I}{n^2}\fvec^\top\Lmat\fvec,
 * $$
 *
 * whose coefficients solve
 * $(\Jmat\Kmat + \gamma_A l\,\Imat + \frac{\gamma_I l}{n^2}\Lmat\Kmat)\,\alphavec = \Jmat\yvec$, with $\Jmat$
 * the diagonal 0/1 indicator of the labelled points. With $\gamma_I = 0$ the coefficients of the unlabelled points
 * vanish and the labelled ones are kernel ridge regression's with $\lambda = \gamma_A l$. The
 * labelled points may sit anywhere in the ordering (given by index), and the intrinsic weight is divided by $n^2$ (the
 * paper's form) or used as given (`scale: 'none'`). The Laplacian is `laplacian` of `aifn-compute/graph/matrices`, of a
 * graph or of an affinity matrix.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, matmul, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'

/** Options of `laplacianRegularisedLeastSquares`. */
export interface LapRlsOptions {
  /** The ambient (RKHS norm) weight $\gamma_A > 0$. */
  gammaA: number
  /** The intrinsic (graph) weight $\gamma_I \ge 0$; 0 gives kernel ridge regression on the labelled points. */
  gammaI: number
  /** `'squared-count'` (default) divides $\gamma_I$ by $n^2$ as the paper; `'none'` uses it as given. */
  scale?: 'squared-count' | 'none'
}

/** A fitted LapRLS model over the $n$ training points. */
export interface LapRlsModel {
  /** The expansion coefficients $\alphavec$, one per labelled or unlabelled point ($n$). */
  readonly alpha: Tensor
  /** The fitted values $\fvec = \Kmat\alphavec$ at the $n$ points. */
  readonly fitted: Tensor
}

/**
 * Fit Laplacian regularised least squares; see the file comment. A new point $\xvec$ is predicted by
 * $f(\xvec) = \sum_i \alpha_i k(\xvec_i, \xvec)$, that is `matmul(gram(kernel, xNew, x), alpha)`. Throws
 * `DomainError` for $\gamma_A \le 0$, $\gamma_I < 0$, a bad index or no labelled point, and `ShapeError` when the
 * matrices and targets describe different numbers of points. Pure, worker-safe; $\mathcal{O}(n^3)$ (about 10 ms at
 * $n = 120$).
 *
 * @param K The Gram matrix $\Kmat$ of the kernel over all $n$ points, $n \times n$.
 * @param L The graph Laplacian $\Lmat = \Dmat - \Wmat$ over the same points, $n \times n$.
 * @param targets The target of each point ($n$ values); only the labelled entries are read.
 * @param labelled The indices of the $l$ labelled points.
 * @param options The weights $\gamma_A$ and $\gamma_I$, and how $\gamma_I$ is scaled.
 * @returns The coefficients $\alphavec$ and the fitted values $\Kmat\alphavec$.
 *
 * @example Six points on an arc, labelled at the ends: the graph term fills in the middle
 * import { gram, rbf } from 'aifn-compute/learning/kernels'
 * import { laplacian } from 'aifn-compute/graph/matrices'
 * const pts = [0, 1, 2, 3, 4, 5].map((i) => [Math.cos((i * 50 * Math.PI) / 180), Math.sin((i * 50 * Math.PI) / 180)])
 * const W = pts.map((_, i) => pts.map((__, j) => (Math.abs(i - j) === 1 ? 1 : 0)))
 * const K = gram(rbf({ lengthscale: 1 }), tensor(pts))
 * const y = [0, 1, 2, 3, 4, 5]
 * const fit = (gammaI) => laplacianRegularisedLeastSquares(K, laplacian(W), y, [0, 5], { gammaA: 1e-5, gammaI }).fitted
 * print('kernel ridge:', fit(0))
 * print('LapRLS:', fit(0.036))
 */
export function laplacianRegularisedLeastSquares(
  K: MatrixLike,
  L: MatrixLike,
  targets: ArrayLike<number>,
  labelled: ArrayLike<number>,
  options: LapRlsOptions,
): LapRlsModel {
  const where = 'laplacianRegularisedLeastSquares'
  const { gammaA, gammaI, scale = 'squared-count' } = options
  if (!(gammaA > 0)) throw new DomainError(where, `${where}: gammaA must be positive`)
  if (!(gammaI >= 0)) throw new DomainError(where, `${where}: gammaI must be non-negative`)
  const k = dense.toMatrixF64(K, where)
  const lap = dense.toMatrixF64(L, where)
  const n = k.m
  if (k.n !== n || lap.m !== n || lap.n !== n || targets.length !== n)
    throw new ShapeError(where, `${where}: K, L and targets must all describe the same n points`)
  const isLabelled = new Array<boolean>(n).fill(false)
  for (const i of Array.from(labelled)) {
    if (!(Number.isInteger(i) && i >= 0 && i < n)) throw new DomainError(where, `${where}: bad index ${i}`)
    isLabelled[i] = true
  }
  const l = isLabelled.filter(Boolean).length
  if (l === 0) throw new DomainError(where, `${where}: at least one point must be labelled`)
  const intrinsic = (gammaI * l) / (scale === 'none' ? 1 : n * n)
  // A = J K + γ_A l I + (γ_I l / n²) L K, and b = J y, accumulated row by row.
  const a = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const row = i * n
    if (isLabelled[i]) for (let j = 0; j < n; j++) a[row + j] += k.data[row + j]
    a[row + i] += gammaA * l
    if (intrinsic > 0)
      for (let m = 0; m < n; m++) {
        const c = lap.data[row + m]
        if (c === 0) continue
        for (let j = 0; j < n; j++) a[row + j] += intrinsic * c * k.data[m * n + j]
      }
  }
  const b = Float64Array.from({ length: n }, (_, i) => (isLabelled[i] ? Number(targets[i]) : 0))
  const alpha = solve(fromData(a, [n, n]), fromData(b, [n])) as Tensor
  const fitted = matmul(fromData(k.data, [n, n]), alpha) as Tensor
  return { alpha, fitted: tensor(toFlat(fitted)) }
}
