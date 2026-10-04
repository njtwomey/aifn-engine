/**
 * Covariance and precision ellipses of 2-D Gaussians: the level set $(\xvec - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\xvec - \boldsymbol{\mu}) = k^2$,
 * drawn at $k$ standard deviations or at the radius that encloses a chosen probability mass
 * (Johnson & Wichern, 2007, §4.2: the contours of constant density and the $\chi^2_2$ mass they enclose).
 */

import { eigh2, type Mat2 } from 'aifn-compute/numerics/linalg'
import {
  dense,
  fromData,
  log1p,
  mul,
  neg,
  sqrt,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** An ellipse: its outline, centre, semi-axes and orientation. */
export interface Ellipse {
  /** Closed outline coordinates tensor of shape $[(\text{points} + 1), 2]$ (the last point repeats the first). */
  points: Tensor
  /** Centre point coordinates $(c_x, c_y)$. */
  center: [number, number]
  /** Semi-axis lengths $[a, b]$, major axis first. */
  radii: [number, number]
  /** Angle of the major axis from the x-axis in radians, within $(-\pi/2, \pi/2]$. */
  angle: number
  /** The Mahalanobis radius $k$ of the level set. */
  k: number
  /** Probability mass of a 2-D Gaussian enclosed by the ellipse, $1 - \exp(-k^2/2)$. */
  mass: number
  /** Whether the underlying matrix was not positive definite (leading to NaN or infinite radii). */
  degenerate: boolean
}

/** Options for ellipse extraction: specifies the contour scale as $k$ (Mahalanobis radius) or probability `mass`. */
export interface EllipseOptions {
  /** Mahalanobis radius $k$. Default 1 (one standard deviation) unless `mass` is given. */
  k?: number
  /** Probability mass enclosed in $(0, 1)$; sets $k = \sqrt{-2 \ln(1 - \text{mass})}$, the $\chi^2_2$ quantile. */
  mass?: number
  /** Number of discretization points on the outline (default 100). */
  points?: number
}

/**
 * Parse a $2 \times 2$ matrix-like argument into a `Mat2` tuple and validate dimensions.
 *
 * @param m - Matrix-like data of shape $[2, 2]$.
 * @param what - Function or context name for error messages.
 * @returns 2x2 matrix tuple `Mat2`.
 */
export function readMatrix2(m: MatrixLike, what: string): Mat2 {
  const v = dense.toMatrixF64(m, what, 2, 2).data
  return [
    [v[0], v[1]],
    [v[2], v[3]],
  ]
}

/**
 * Parse a 2-element vector-like argument into an $[x, y]$ coordinate pair and validate length.
 *
 * @param p - Vector-like data of length 2.
 * @param what - Function or context name for error messages.
 * @returns 2-element number array $[x, y]$.
 */
export function readPoint2(p: VectorLike, what: string): [number, number] {
  const v = dense.toF64(p, what)
  if (v.length !== 2) throw new ShapeError(what, `${what}: expected a point of length 2`)
  return [v[0], v[1]]
}

/**
 * Compute the Mahalanobis radius that encloses a given probability `mass` for a 2-D Gaussian:
 * $k = \sqrt{-2 \ln(1 - \text{mass})}$ (the square-rooted $\chi^2_2$ quantile).
 * A number outside $(0, 1)$ throws `DomainError`; tensor entries outside it yield NaN or $\infty$.
 *
 * @param mass - Enclosed probability mass in $(0, 1)$.
 * @returns Mahalanobis radius $k$.
 * @example Mahalanobis radius from mass
 * const r = massToRadius(0.95)
 * print('95% radius =', r)
 */
export function massToRadius(mass: Scalar): Scalar
export function massToRadius(mass: Tensor): Tensor
export function massToRadius(mass: Traced): Traced
export function massToRadius(mass: Value): Value
export function massToRadius(mass: Value): Value {
  if (typeof mass === 'number' && !(mass > 0 && mass < 1))
    throw new DomainError('massToRadius', `massToRadius: mass must be in (0, 1), got ${mass}`)
  return sqrt(mul(-2, log1p(neg(mass))))
}

/**
 * Determine the Mahalanobis radius $k$ from options.
 *
 * @param options - Ellipse configuration options.
 * @returns Mahalanobis radius $k$.
 */
function level(options: EllipseOptions): number {
  if (options.mass !== undefined) return massToRadius(options.mass)
  return options.k ?? 1
}

/**
 * Construct an ellipse with semi-axes aligned with the eigenvectors of a symmetric $2 \times 2$ matrix,
 * with semi-axis lengths scaled by `axisScale(lambda)`.
 *
 * @param mean - Ellipse centre coordinates $[c_x, c_y]$.
 * @param m - Symmetric $2 \times 2$ matrix.
 * @param axisScale - Function mapping each eigenvalue $\lambda$ to its corresponding semi-axis scale.
 * @param options - Ellipse options controlling $k$ or mass and outline resolution.
 * @param what - Calling function name for diagnostics.
 * @returns Fitted `Ellipse` geometry object.
 */
function ellipseOf(
  mean: VectorLike,
  m: MatrixLike,
  axisScale: (lambda: number) => number,
  options: EllipseOptions,
  what: string,
): Ellipse {
  const [cx, cy] = readPoint2(mean, what)
  const mat = readMatrix2(m, what)
  const k = level(options)
  const count = options.points ?? 100
  const { values, vectors } = eigh2(mat)
  // eigh2 returns eigenvalues in descending order, so the first axis is the major one for a covariance.
  let [a, b] = [k * axisScale(values[0]), k * axisScale(values[1])]
  let [u, w] = vectors
  if (b > a) {
    ;[a, b] = [b, a]
    ;[u, w] = [w, u]
  }
  const degenerate = !(values[0] > 0 && values[1] > 0)
  let angle = Math.atan2(u[1], u[0])
  if (angle > Math.PI / 2) angle -= Math.PI
  if (angle <= -Math.PI / 2) angle += Math.PI
  const out = new Float64Array((count + 1) * 2)
  for (let i = 0; i <= count; i++) {
    const t = (2 * Math.PI * (i % count)) / count
    const p = a * Math.cos(t)
    const q = b * Math.sin(t)
    out[2 * i] = cx + p * u[0] + q * w[0]
    out[2 * i + 1] = cy + p * u[1] + q * w[1]
  }
  return {
    points: fromData(out, [count + 1, 2]),
    center: [cx, cy],
    radii: [a, b],
    angle,
    k,
    mass: -Math.expm1((-k * k) / 2),
    degenerate,
  }
}

/**
 * Construct the covariance ellipse $\{\xvec : (\xvec - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\xvec - \boldsymbol{\mu}) = k^2\}$
 * of a 2-D Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$,
 * having semi-axes $k \sqrt{\lambda_i}$ aligned with the eigenvectors of $\boldsymbol{\Sigma}$.
 *
 * @param mean - 2D mean vector $\boldsymbol{\mu} = [\mu_x, \mu_y]$.
 * @param covariance - $2 \times 2$ covariance matrix $\boldsymbol{\Sigma}$.
 * @param options - Ellipse scaling and resolution options.
 * @returns Geometric `Ellipse` object.
 * @example Covariance ellipse
 * const mean = [0, 0]
 * const cov = [
 *   [2, 0.5],
 *   [0.5, 1],
 * ]
 * const ell = covarianceEllipse(mean, cov, { mass: 0.95 })
 * print('radii =', ell.radii)
 */
export function covarianceEllipse(mean: VectorLike, covariance: MatrixLike, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, covariance, (l) => Math.sqrt(l), options, 'covarianceEllipse')
}

/**
 * Construct the confidence ellipse from a precision matrix $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ (e.g. a Hessian or Fisher information matrix):
 * $\{\xvec : (\xvec - \boldsymbol{\mu})^\top \boldsymbol{\Lambda} (\xvec - \boldsymbol{\mu}) = k^2\}$,
 * having semi-axes $k / \sqrt{\lambda_i}$ aligned with the eigenvectors of $\boldsymbol{\Lambda}$.
 *
 * @param mean - 2D mean vector $\boldsymbol{\mu} = [\mu_x, \mu_y]$.
 * @param precision - $2 \times 2$ precision matrix $\boldsymbol{\Lambda}$.
 * @param options - Ellipse scaling and resolution options.
 * @returns Geometric `Ellipse` object.
 * @example Precision ellipse
 * const mean = [0, 0]
 * const prec = [
 *   [2, 0],
 *   [0, 1],
 * ]
 * const ell = precisionEllipse(mean, prec, { k: 2 })
 * print('radii =', ell.radii)
 */
export function precisionEllipse(mean: VectorLike, precision: MatrixLike, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, precision, (l) => 1 / Math.sqrt(l), options, 'precisionEllipse')
}
