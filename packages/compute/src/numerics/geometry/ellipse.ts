/**
 * Covariance and precision ellipses of 2-D Gaussians: the level set (x − μ)ᵀ Σ⁻¹ (x − μ) = k², drawn at k standard
 * deviations or at the radius that encloses a chosen probability mass (Johnson & Wichern, 2007, "Applied Multivariate
 * Statistical Analysis", 6th ed., §4.2: the contours of constant density and the χ²₂ mass they enclose).
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
  /** Closed outline, (points + 1) × 2 (the last point repeats the first). */
  points: Tensor
  center: [number, number]
  /** Semi-axis lengths, major first. */
  radii: [number, number]
  /** Angle of the major axis from the x-axis, radians in (−π/2, π/2]. */
  angle: number
  /** The Mahalanobis radius k of the level set. */
  k: number
  /** Probability mass of a 2-D Gaussian inside the ellipse, 1 − exp(−k²/2). */
  mass: number
  /** True when the matrix was not positive definite (a radius is then NaN or infinite). */
  degenerate: boolean
}

/** Options for the ellipse functions: the level as k (standard deviations) or as a probability mass. */
export interface EllipseOptions {
  /** Mahalanobis radius. Default 1 (one standard deviation) unless `mass` is given. */
  k?: number
  /** Probability mass enclosed, in (0, 1); sets k = √(−2 ln(1 − mass)), the χ²₂ quantile. */
  mass?: number
  /** Points on the outline. Default 100. */
  points?: number
}

/** A 2 × 2 matrix argument as a `Mat2` tuple (checks the shape). */
export function readMatrix2(m: MatrixLike, what: string): Mat2 {
  const v = dense.toMatrixF64(m, what, 2, 2).data
  return [
    [v[0], v[1]],
    [v[2], v[3]],
  ]
}

/** A 2-vector argument as a pair (checks the length). */
export function readPoint2(p: VectorLike, what: string): [number, number] {
  const v = dense.toF64(p, what)
  if (v.length !== 2) throw new ShapeError(what, `${what}: expected a point of length 2`)
  return [v[0], v[1]]
}

/**
 * The Mahalanobis radius that encloses probability `mass` of a 2-D Gaussian: √(−2 ln(1 − mass)), elementwise (the χ²₂
 * quantile, square-rooted). A number outside (0, 1) throws `DomainError`; tensor entries outside it give NaN or ∞.
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

function level(options: EllipseOptions): number {
  if (options.mass !== undefined) return massToRadius(options.mass)
  return options.k ?? 1
}

/** Ellipse with semi-axes along the eigenvectors of a symmetric matrix with eigenvalues `axisScale(λ)`. */
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
 * The covariance ellipse {x : (x − μ)ᵀ Σ⁻¹ (x − μ) = k²} of a 2-D Gaussian with mean μ and covariance Σ: semi-axes
 * k√λᵢ along the eigenvectors of Σ.
 */
export function covarianceEllipse(mean: VectorLike, covariance: MatrixLike, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, covariance, (l) => Math.sqrt(l), options, 'covarianceEllipse')
}

/**
 * The same level set given the precision Λ = Σ⁻¹ (e.g. a Hessian or an information matrix): semi-axes k/√λᵢ along the
 * eigenvectors of Λ.
 */
export function precisionEllipse(mean: VectorLike, precision: MatrixLike, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, precision, (l) => 1 / Math.sqrt(l), options, 'precisionEllipse')
}
