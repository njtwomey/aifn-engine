/**
 * One-class kernel methods: the one-class support vector machine (Schölkopf et al., 2001), which separates the data
 * from the origin in feature space with maximum margin, and support vector data description (Tax and Duin, 2004),
 * which encloses the data in the smallest ball. Both are quadratic programmes in the dual over weights $\alphavec$
 * with $0 \le \alpha_i \le 1/(\nu n)$ and $\ones^\top \alphavec = 1$, solved by
 * `aifn-compute/optim/programming`'s interior-point method and then polished by a few active-set rounds, with a
 * Gaussian kernel from `aifn-compute/learning/kernels`; with that kernel ($k(\xvec, \xvec) = 1$) the two give the same
 * boundary. The kernel matrix and the programme are dense, $n \times n$, so they suit a few hundred points.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { quadprog } from 'aifn-compute/optim/programming'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `oneClassSvm` and `supportVectorDataDescription`. */
export type OneClassOptions = {
  /**
   * $\nu \in (0, 1]$: an upper bound on the share of training points outside the boundary (default 0.1). Outside
   * $(0, 1]$ it throws `DomainError`.
   */
  nu?: number
  /**
   * The $\gamma$ of the Gaussian kernel $\exp(-\gamma \lVert \xvec - \yvec \rVert^2)$, as scikit-learn's (default
   * $1/(d \cdot \var \Xmat)$, its `'scale'`, with the variance over every entry of $\Xmat$, or $1/d$ when that is 0).
   */
  gamma?: number
}

/** A fitted one-class model: support points, their dual weights, the kernel's $\gamma$ and the offset. */
export type OneClassModel = {
  /** Which method fitted it, which sets how `oneClassScore` reads it. */
  kind: 'one-class-svm' | 'svdd'
  /** The support points (training points with a positive weight), $s \times d$. */
  support: Tensor
  /** Dual weights $\alpha_i$ of the support points (they sum to 1). */
  alpha: Float64Array
  /** The kernel's $\gamma$. */
  gamma: number
  /**
   * OCSVM: the offset $\rho$ of $f(\xvec) = \sum_i \alpha_i k(\xvec_i, \xvec) - \rho$. SVDD: the squared radius
   * $R^2$.
   */
  offset: number
  /** SVDD: $\alphavec^\top \Kmat \alphavec$, the squared norm of the centre (0 for the one-class SVM). */
  centreNorm: number
  /** The share of training points with $\alpha_i$ at its upper bound (outside the boundary). */
  boundShare: number
}

/**
 * The Gaussian kernel $\exp(-\gamma \lVert \xvec - \yvec \rVert^2)$ as an RBF kernel of lengthscale
 * $1/\sqrt{2\gamma}$.
 *
 * @param gamma The kernel's $\gamma$, positive.
 * @returns The kernel.
 */
const kernelOf = (gamma: number) => rbf({ lengthscale: 1 / Math.sqrt(2 * gamma) })

/**
 * The common start of both fits: the data as a tensor, $\nu$ checked, $\gamma$ chosen, and the kernel matrix.
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options $\nu$ and $\gamma$, as given to the fit.
 * @param where The caller's name for error messages.
 * @returns `X` ($n \times d$), `n`, `nu`, `gamma`, the kernel matrix `K` (row-major, $n \times n$) and the box bound
 *   `C` $= 1/(\nu n)$.
 */
function setup(x: MatrixLike, options: OneClassOptions, where: string) {
  const m = dense.toMatrixF64(x, where)
  const X = fromData(Float64Array.from(m.data), [m.m, m.n])
  const { nu = 0.1 } = options
  if (!(nu > 0 && nu <= 1)) throw new DomainError(where, `${where}: ν must lie in (0, 1], got ${nu}`)
  let gamma = options.gamma
  if (gamma === undefined) {
    let s = 0
    let s2 = 0
    for (const v of m.data) {
      s += v
      s2 += v * v
    }
    const variance = s2 / m.data.length - (s / m.data.length) ** 2
    gamma = 1 / (m.n * (variance > 0 ? variance : 1))
  }
  const K = Float64Array.from(toFlat(gram(kernelOf(gamma), X)))
  return { X, n: m.m, nu, gamma, K, C: 1 / (nu * m.m) }
}

/**
 * Solve $\min_{\alphavec} \frac{1}{2} \alphavec^\top \Qmat \alphavec + \cvec^\top \alphavec$ subject to
 * $0 \le \alpha_i \le C$ and $\ones^\top \alphavec = 1$ by the interior-point method, clip the result to the box,
 * and polish it (`polish`) when that settles.
 *
 * @param Q The quadratic term $\Qmat$, $n \times n$, row-major, symmetric positive semi-definite.
 * @param c The linear term $\cvec$, $n$ values.
 * @param n The number of weights.
 * @param C The upper bound of each weight.
 * @returns The weights $\alphavec$, $n$ values.
 */
function boxSimplexQp(Q: Float64Array, c: Float64Array, n: Size, C: number): Float64Array {
  const A = new Float64Array(2 * n * n)
  const b = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    A[i * n + i] = 1
    b[i] = C
    A[(n + i) * n + i] = -1
  }
  const r = quadprog(
    {
      Q: fromData(Q, [n, n]),
      c: fromData(c, [n]),
      A: fromData(A, [2 * n, n]),
      b: fromData(b, [2 * n]),
      E: fromData(new Float64Array(n).fill(1), [1, n]),
      e: fromData(Float64Array.of(1), [1]),
    },
    { method: 'interior-point', tolerance: 1e-10 },
  )
  const alpha = Float64Array.from(toFlat(r.x), (v) => Math.min(C, Math.max(0, v)))
  return polish(Q, c, alpha, n, C) ?? alpha
}

/**
 * The interior point stops strictly inside the box, so its weights at the bounds are only near them. Starting from the
 * bound set it found (weights within $10^{-4} C$ of a bound), solve the KKT system of the free weights exactly,
 * $\Qmat_{FF} \alphavec_F - \mu \ones = -\Qmat_{FB} \alphavec_B - \cvec_F$ with
 * $\ones^\top \alphavec_F = 1 - \ones^\top \alphavec_B$, moving a free weight that leaves the box to its bound and
 * freeing a bound weight whose multiplier has the wrong sign, until the set settles (a few rounds of an active-set
 * method from a good start). Returns null if it does not settle within 50 rounds, if no weight is free, or if the
 * system is singular (the interior point's weights are then kept).
 *
 * @param Q The quadratic term, $n \times n$, row-major.
 * @param c The linear term, $n$ values.
 * @param alpha The interior point's weights, clipped to the box; read, not modified.
 * @param n The number of weights.
 * @param C The upper bound of each weight.
 * @returns The polished weights, or null.
 */
function polish(Q: Float64Array, c: Float64Array, alpha: Float64Array, n: Size, C: number): Float64Array | null {
  const tol = 1e-4 * C
  // 0 or C for a weight held at a bound, NaN for a free one.
  const fixed = Float64Array.from(alpha, (a) => (a < tol ? 0 : a > C - tol ? C : NaN))
  for (let round = 0; round < 50; round++) {
    const free: number[] = []
    fixed.forEach((a, i) => Number.isNaN(a) && free.push(i))
    if (free.length === 0) return null
    const f = free.length
    const M = new Float64Array((f + 1) * (f + 1))
    const rhs = new Float64Array(f + 1)
    let budget = 1
    for (let i = 0; i < n; i++) if (!Number.isNaN(fixed[i])) budget -= fixed[i]
    free.forEach((i, a) => {
      free.forEach((j, b) => (M[a * (f + 1) + b] = Q[i * n + j]))
      M[a * (f + 1) + f] = -1
      M[f * (f + 1) + a] = 1
      let r = -c[i]
      for (let j = 0; j < n; j++) if (fixed[j] === C) r -= Q[i * n + j] * C
      rhs[a] = r
    })
    rhs[f] = budget
    const sol = solveDense(M, rhs, f + 1)
    if (!sol.x) return null
    const out = Float64Array.from(fixed, (a) => (Number.isNaN(a) ? 0 : a))
    free.forEach((i, a) => (out[i] = sol.x![a]))
    const mu = sol.x[f]
    let changed = false
    // A free weight outside the box goes to the bound it crossed.
    free.forEach((i) => {
      if (out[i] < 0) {
        fixed[i] = 0
        changed = true
      } else if (out[i] > C) {
        fixed[i] = C
        changed = true
      }
    })
    if (changed) continue
    // A bound weight whose gradient says it should move inward is freed.
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(fixed[i])) continue
      let g = c[i]
      for (let j = 0; j < n; j++) g += Q[i * n + j] * out[j]
      if ((fixed[i] === 0 && g - mu < -1e-12) || (fixed[i] === C && g - mu > 1e-12)) {
        fixed[i] = NaN
        changed = true
      }
    }
    if (!changed) return out
  }
  return null
}

/**
 * Indices with $0 < \alpha_i < C$ (free support vectors), allowing a tolerance of $10^{-9} C$ at each bound.
 *
 * @param alpha The weights.
 * @param C The upper bound of each weight.
 * @returns The indices of the free weights, ascending.
 */
function freeSupport(alpha: Float64Array, C: number): number[] {
  const tol = 1e-9 * C
  const out: number[] = []
  alpha.forEach((a, i) => a > tol && a < C - tol && out.push(i))
  return out
}

/**
 * Keep the support points: the rows whose weight exceeds $10^{-9} C$.
 *
 * @param X The training points, $n \times d$.
 * @param alpha The weights, $n$ values.
 * @param C The upper bound of each weight.
 * @returns `support`, the kept rows ($s \times d$); `alpha`, their weights; and `bound`, the share of all $n$
 *   weights at their upper bound.
 */
function keepSupport(X: Tensor, alpha: Float64Array, C: number) {
  const idx: number[] = []
  alpha.forEach((a, i) => a > 1e-9 * C && idx.push(i))
  const d = X.shape[1]
  const rows = new Float64Array(idx.length * d)
  idx.forEach((i, r) => rows.set(X.data.subarray(i * d, (i + 1) * d) as Float64Array, r * d))
  const bound = Array.from(alpha).filter((a) => a >= C * (1 - 1e-9)).length / alpha.length
  return { support: fromData(rows, [idx.length, d]), alpha: Float64Array.from(idx, (i) => alpha[i]), bound }
}

/**
 * The one-class SVM (Schölkopf, Platt, Shawe-Taylor, Smola and Williamson, 2001) in its dual: minimise
 * $\frac{1}{2} \alphavec^\top \Kmat \alphavec$ subject to $0 \le \alpha_i \le 1/(\nu n)$ and
 * $\sum_i \alpha_i = 1$. The decision function $f(\xvec) = \sum_i \alpha_i k(\xvec_i, \xvec) - \rho$ is positive
 * inside the boundary; $\rho$ is the mean of $(\Kmat\alphavec)_i$ over the free support vectors (over all support
 * vectors when none is free). $\nu$ bounds the share of training points outside from above and the share of support
 * vectors from below. scikit-learn's `OneClassSVM` scales $\alphavec$ and $\rho$ by $\nu n$.
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options $\nu$ and the kernel's $\gamma$.
 * @returns The model, to score points with `oneClassScore`.
 *
 * @example A cloud of twenty points: inside and outside
 * const x = normals(stream(0), [20, 2])
 * const model = oneClassSvm(x, { nu: 0.2, gamma: 0.5 })
 * print('support points', model.alpha.length, 'at the bound', model.boundShare)
 * print('scores', oneClassScore(model, [[0, 0], [4, 4]]))
 */
export function oneClassSvm(x: MatrixLike, options: OneClassOptions = {}): OneClassModel {
  const { X, n, gamma, K, C } = setup(x, options, 'oneClassSvm')
  const alpha = boxSimplexQp(K, new Float64Array(n), n, C)
  const Ka = dense.matVec(K, alpha, n, n)
  const free = freeSupport(alpha, C)
  const pool = free.length ? free : Array.from({ length: n }, (_, i) => i).filter((i) => alpha[i] > 0)
  const offset = pool.reduce((s, i) => s + Ka[i], 0) / pool.length
  const kept = keepSupport(X, alpha, C)
  return {
    kind: 'one-class-svm',
    support: kept.support,
    alpha: kept.alpha,
    gamma,
    offset,
    centreNorm: 0,
    boundShare: kept.bound,
  }
}

/**
 * Support vector data description (Tax and Duin, 2004): the smallest ball (centre $\avec$, radius $R$) in feature space
 * holding the data, with slack for a share of at most $\nu$ outside. Dual: minimise
 * $\alphavec^\top \Kmat \alphavec - \sum_i \alpha_i k(\xvec_i, \xvec_i)$ subject to $0 \le \alpha_i \le 1/(\nu n)$
 * and $\sum_i \alpha_i = 1$; then $\avec = \sum_i \alpha_i \phi(\xvec_i)$ and $R^2$ is the mean squared distance
 * of the free support vectors from $\avec$ (of all support vectors when none is free).
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options $\nu$ and the kernel's $\gamma$.
 * @returns The model, to score points with `oneClassScore`.
 *
 * @example The ball around twenty points
 * const x = normals(stream(0), [20, 2])
 * const model = supportVectorDataDescription(x, { nu: 0.2, gamma: 0.5 })
 * print('squared radius', model.offset)
 * print('scores', oneClassScore(model, [[0, 0], [4, 4]]))
 */
export function supportVectorDataDescription(x: MatrixLike, options: OneClassOptions = {}): OneClassModel {
  const { X, n, gamma, K, C } = setup(x, options, 'supportVectorDataDescription')
  const Q = K.map((v) => 2 * v)
  const c = Float64Array.from({ length: n }, (_, i) => -K[i * n + i])
  const alpha = boxSimplexQp(Q, c, n, C)
  const Ka = dense.matVec(K, alpha, n, n)
  const centreNorm = alpha.reduce((s, a, i) => s + a * Ka[i], 0)
  const free = freeSupport(alpha, C)
  const pool = free.length ? free : Array.from({ length: n }, (_, i) => i).filter((i) => alpha[i] > 0)
  const offset = pool.reduce((s, i) => s + K[i * n + i] - 2 * Ka[i] + centreNorm, 0) / pool.length
  const kept = keepSupport(X, alpha, C)
  return { kind: 'svdd', support: kept.support, alpha: kept.alpha, gamma, offset, centreNorm, boundShare: kept.bound }
}

/**
 * The anomaly score of each row of `x` under a one-class model, positive outside the boundary and negative inside:
 * $-f(\xvec)$ for the one-class SVM, and $\lVert \phi(\xvec) - \avec \rVert^2 - R^2$ for SVDD (using
 * $k(\xvec, \xvec) = 1$ of the Gaussian kernel).
 *
 * @param model The model, as `oneClassSvm` or `supportVectorDataDescription` returns it.
 * @param x The points to score, $m \times d$: nested arrays or a rank-2 tensor.
 * @returns The score of each row ($m$ values).
 *
 * @example The two methods agree on the sign
 * const x = normals(stream(0), [20, 2])
 * const q = [[0, 0], [1.5, 0], [4, 4]]
 * print('one-class SVM', oneClassScore(oneClassSvm(x, { gamma: 0.5 }), q))
 * print('SVDD', oneClassScore(supportVectorDataDescription(x, { gamma: 0.5 }), q))
 */
export function oneClassScore(model: OneClassModel, x: MatrixLike): Float64Array {
  const m = dense.toMatrixF64(x, 'oneClassScore')
  const Q = fromData(Float64Array.from(m.data), [m.m, m.n])
  const Kq = toFlat(gram(kernelOf(model.gamma), Q, model.support))
  const s = model.alpha.length
  const out = new Float64Array(m.m)
  for (let q = 0; q < m.m; q++) {
    let f = 0
    for (let j = 0; j < s; j++) f += model.alpha[j] * Kq[q * s + j]
    out[q] = model.kind === 'one-class-svm' ? model.offset - f : 1 - 2 * f + model.centreNorm - model.offset
  }
  return out
}
