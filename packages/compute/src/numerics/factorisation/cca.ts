/**
 * Canonical correlation analysis (Hotelling, 1936): pairs of directions a_c, b_c whose projections Xa_c and Yb_c are
 * maximally correlated, each pair uncorrelated with the earlier ones. With the sample covariances C_xx, C_yy, C_xy, the
 * pairs are the singular vectors of the whitened cross-covariance M = C_xx^{−1/2} C_xy C_yy^{−1/2} = U S Vᵀ, mapped
 * back: a_c = C_xx^{−1/2} u_c, b_c = C_yy^{−1/2} v_c, and the canonical correlations are the singular values. Scores
 * have unit sample variance.
 *
 * Regularised CCA (Vinod, 1976; Hardoon, Szedmak and Shawe-Taylor, 2004) adds ridge terms, C_xx + r_x I and
 * C_yy + r_y I, which makes the problem well posed when a block has more columns than rows or near-collinear columns;
 * the reported correlations are then the regularised ones (shrunk towards 0).
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { svd, symmetricInverseSqrt } from 'aifn-compute/numerics/linalg'

/** Options of `canonicalCorrelation`. */
export type CcaOptions = {
  /** Number of canonical pairs r (default min(p, q)). */
  components?: Size
  /** Ridge added to C_xx and C_yy: one value for both, or [r_x, r_y] (default 0, classical CCA). */
  regularisation?: number | readonly [number, number]
}

/** A fitted CCA. */
export type Cca = {
  /** Canonical directions for X as columns [p, r]; each pair signed so the largest-magnitude entry of a_c is positive. */
  xWeights: Tensor
  /** Canonical directions for Y as columns [q, r]. */
  yWeights: Tensor
  /** Canonical correlations [r], descending. */
  correlations: Tensor
  xMean: Tensor
  yMean: Tensor
  /** Canonical scores (X − x̄) A [m, r]. */
  transformX(X: MatrixLike): Tensor
  /** Canonical scores (Y − ȳ) B [m, r]. */
  transformY(Y: MatrixLike): Tensor
}

function centred(data: Float64Array, n: Size, d: Size) {
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n
  return { mean, Xc: Float64Array.from(data, (v, t) => v - mean[t % d]) }
}

/** S^{−1/2} of a covariance block S [d, d]; a singular block is refused with the remedy. */
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

/** Fit CCA (or regularised CCA) to paired rows X [n, p] and Y [n, q]. */
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
