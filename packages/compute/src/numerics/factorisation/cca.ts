/**
 * Canonical correlation analysis (Hotelling, 1936): pairs of directions $\mathbf{a}_c, \mathbf{b}_c$ whose projections $\mathbf{X}\mathbf{a}_c$ and $\mathbf{Y}\mathbf{b}_c$ are
 * maximally correlated, each pair uncorrelated with the earlier ones. With the sample covariances $\mathbf{C}_{xx}, \mathbf{C}_{yy}, \mathbf{C}_{xy}$, the
 * pairs are the singular vectors of the whitened cross-covariance $\mathbf{M} = \mathbf{C}_{xx}^{-1/2} \mathbf{C}_{xy} \mathbf{C}_{yy}^{-1/2} = \mathbf{U} \mathbf{S} \mathbf{V}^\top$, mapped
 * back: $\mathbf{a}_c = \mathbf{C}_{xx}^{-1/2} \mathbf{u}_c$, $\mathbf{b}_c = \mathbf{C}_{yy}^{-1/2} \mathbf{v}_c$, and the canonical correlations are the singular values. Scores
 * have unit sample variance.
 *
 * Regularised CCA (Vinod, 1976; Hardoon, Szedmak and Shawe-Taylor, 2004) adds ridge terms, $\mathbf{C}_{xx} + r_x \mathbf{I}$ and
 * $\mathbf{C}_{yy} + r_y \mathbf{I}$, which makes the problem well posed when a block has more columns than rows or near-collinear columns;
 * the reported correlations are then the regularised ones (shrunk towards 0).
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { svd, symmetricInverseSqrt } from 'aifn-compute/numerics/linalg'

/** Options for configuring `canonicalCorrelation`. */
export type CcaOptions = {
  /** Number of canonical pairs $r$ (default $\min(p, q)$). */
  components?: Size
  /** Ridge added to $\mathbf{C}_{xx}$ and $\mathbf{C}_{yy}$: one value for both, or $[r_x, r_y]$ (default 0, classical CCA). */
  regularisation?: number | readonly [number, number]
}

/** A fitted CCA model containing canonical directions, correlations, and projection transforms. */
export type Cca = {
  /** Canonical directions for $\mathbf{X}$ as columns of shape $[p, r]$; each pair signed so the largest-magnitude entry of $\mathbf{a}_c$ is positive. */
  xWeights: Tensor
  /** Canonical directions for $\mathbf{Y}$ as columns of shape $[q, r]$. */
  yWeights: Tensor
  /** Canonical correlations of length $r$, sorted in descending order. */
  correlations: Tensor
  /** Column means of $\mathbf{X}$ ($p$ values). */
  xMean: Tensor
  /** Column means of $\mathbf{Y}$ ($q$ values). */
  yMean: Tensor
  /**
   * Project new data $\mathbf{X}$ onto canonical variates: $(\mathbf{X} - \bar{\mathbf{x}}) \mathbf{A}$ of shape $[m, r]$.
   *
   * @param X Data matrix of shape $[m, p]$.
   * @returns Canonical scores tensor of shape $[m, r]$.
   */
  transformX(X: MatrixLike): Tensor
  /**
   * Project new data $\mathbf{Y}$ onto canonical variates: $(\mathbf{Y} - \bar{\mathbf{y}}) \mathbf{B}$ of shape $[m, r]$.
   *
   * @param Y Data matrix of shape $[m, q]$.
   * @returns Canonical scores tensor of shape $[m, r]$.
   */
  transformY(Y: MatrixLike): Tensor
}

/**
 * Compute the column means and zero-centred copy of a row-major matrix.
 *
 * @param data Row-major matrix elements of length $n \times d$.
 * @param n Number of rows.
 * @param d Number of columns.
 * @returns An object containing column means array `mean` and zero-centred array `Xc`.
 */
function centred(data: Float64Array, n: Size, d: Size) {
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n
  return { mean, Xc: Float64Array.from(data, (v, t) => v - mean[t % d]) }
}

/**
 * Compute $S^{-1/2}$ of a covariance block $S$ ($d \times d$); throws a `NumericalError` with code `'singular'` if
 * $S$ is not positive definite.
 *
 * @param S The $d \times d$ covariance matrix as a row-major array.
 * @param d The dimension $d$ of the covariance matrix.
 * @returns The inverse square root matrix $S^{-1/2}$ as a row-major array.
 */
function inverseSqrt(S: Float64Array, d: Size): Float64Array {
  try {
    return dense.data(symmetricInverseSqrt(fromData(S, [d, d])))
  } catch (e) {
    if (!(e instanceof NumericalError)) throw e
    throw new NumericalError(
      'canonicalCorrelation',
      'canonicalCorrelation: a covariance block is singular; add regularisation',
      'singular',
    )
  }
}

/**
 * Fit CCA (or regularised CCA) to paired rows $X$ ($n \times p$) and $Y$ ($n \times q$).
 *
 * @param X First data matrix of shape $[n, p]$ with rows as observations.
 * @param Y Second data matrix of shape $[n, q]$ with rows as observations.
 * @param options Configuration for components count and ridge regularisation.
 * @returns A fitted `Cca` object containing weights, canonical correlations, means, and projection methods.
 *
 * @example Fit canonical correlation analysis
 * const X = [[1, 0], [0, 1], [-1, 0], [0, -1]]
 * const Y = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
 * const res = canonicalCorrelation(X, Y, { components: 1 })
 * print('correlation =', res.correlations)
 */
export function canonicalCorrelation(X: MatrixLike, Y: MatrixLike, options: CcaOptions = {}): Cca {
  const x = dense.toMatrixF64(X, 'canonicalCorrelation')
  const y = dense.toMatrixF64(Y, 'canonicalCorrelation')
  const n = x.m
  if (y.m !== n) throw new ShapeError('canonicalCorrelation', `canonicalCorrelation: X has ${n} rows and Y ${y.m}`)
  if (n < 2) throw new DomainError('canonicalCorrelation', 'canonicalCorrelation: need at least two rows')
  const p = x.n
  const q = y.n
  const r = Math.min(options.components ?? Math.min(p, q), p, q)
  const reg = options.regularisation ?? 0
  const [rx, ry] = typeof reg === 'number' ? [reg, reg] : reg
  if (!(rx >= 0 && ry >= 0))
    throw new DomainError('canonicalCorrelation', 'canonicalCorrelation: regularisation must be non-negative')
  const cx = centred(x.data, n, p)
  const cy = centred(y.data, n, q)
  const cov = (a: Float64Array, da: Size, b: Float64Array, db: Size, ridge: number) => {
    const out = dense.matMul(dense.transpose(a, n, da), b, da, n, db)
    for (let i = 0; i < out.length; i++) out[i] /= n - 1
    if (ridge > 0) for (let i = 0; i < da; i++) out[i * db + i] += ridge
    return out
  }
  const Wx = inverseSqrt(cov(cx.Xc, p, cx.Xc, p, rx), p)
  const Wy = inverseSqrt(cov(cy.Xc, q, cy.Xc, q, ry), q)
  const Cxy = cov(cx.Xc, p, cy.Xc, q, 0)
  const M = dense.matMul(dense.matMul(Wx, Cxy, p, p, q), Wy, p, q, q)
  const { U, S, V } = svd(fromData(M, [p, q]))
  const k = Math.min(p, q)
  const u = dense.data(U)
  const v = dense.data(V)
  const s = dense.data(S)
  const A = new Float64Array(p * r)
  const B = new Float64Array(q * r)
  for (let c = 0; c < r; c++) {
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) A[i * r + c] += Wx[i * p + j] * u[j * k + c]
    for (let i = 0; i < q; i++) for (let j = 0; j < q; j++) B[i * r + c] += Wy[i * q + j] * v[j * k + c]
    let big = 0
    for (let i = 0; i < p; i++) if (Math.abs(A[i * r + c]) > Math.abs(big)) big = A[i * r + c]
    if (big < 0) {
      for (let i = 0; i < p; i++) A[i * r + c] = -A[i * r + c]
      for (let i = 0; i < q; i++) B[i * r + c] = -B[i * r + c]
    }
  }
  const scores = (Z: MatrixLike, mean: Float64Array, W: Float64Array, d: Size, where: string) => {
    const z = dense.toMatrixF64(Z, where)
    if (z.n !== d) throw new ShapeError(where, `${where}: fitted on ${d} columns, given ${z.n}`)
    const Zc = Float64Array.from(z.data, (val, t) => val - mean[t % d])
    return fromData(dense.matMul(Zc, W, z.m, d, r), [z.m, r])
  }
  return {
    xWeights: fromData(A, [p, r]),
    yWeights: fromData(B, [q, r]),
    correlations: fromData(Float64Array.from(s.slice(0, r)), [r]),
    xMean: fromData(cx.mean, [p]),
    yMean: fromData(cy.mean, [q]),
    transformX: (Z) => scores(Z, cx.mean, A, p, 'canonicalCorrelation.transformX'),
    transformY: (Z) => scores(Z, cy.mean, B, q, 'canonicalCorrelation.transformY'),
  }
}
