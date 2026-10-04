/**
 * Closed forms for 2×2 matrices on plain tuples, for per-frame work such as handle drags. A matrix is written
 * [[a, b], [c, d]] row by row. These are plain-number functions (not traced). The symmetric eigenproblem is the 2×2
 * Jacobi rotation (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., §8.5.2), the general one the roots of the
 * characteristic quadratic in the cancellation-free form (Press et al., 2007, "Numerical Recipes", 3rd ed., §5.6), and
 * the SVD the eigendecomposition of MᵀM (Blinn, 1996, "Consider the lowly 2×2 matrix", IEEE CG&A 16(2)).
 */

import { EPS } from './dense'

/** A 2-vector. */
export type Vec2 = [number, number]
/** A 2×2 matrix [[a, b], [c, d]], row by row. */
export type Mat2 = [[number, number], [number, number]]

/** ad − bc. */
export function det2([[a, b], [c, d]]: Mat2): number {
  return a * d - b * c
}

/** M v. */
export function apply2([[a, b], [c, d]]: Mat2, [x, y]: Vec2): Vec2 {
  return [a * x + b * y, c * x + d * y]
}

/**
 * The inverse, or null when the matrix is singular to working precision: |ad − bc| ≤ 4ε(|ad| + |bc|), where the
 * determinant is lost to cancellation.
 */
export function inv2(m: Mat2): Mat2 | null {
  const [[a, b], [c, d]] = m
  const det = a * d - b * c
  if (!(Math.abs(det) > 4 * EPS * (Math.abs(a * d) + Math.abs(b * c)))) return null
  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ]
}

/** Flip v so that its largest-magnitude component (the first of equals) is positive. */
function signed([x, y]: Vec2): Vec2 {
  const lead = Math.abs(y) > Math.abs(x) ? y : x
  return lead < 0 ? [-x, -y] : [x, y]
}

/**
 * Eigendecomposition of a symmetric matrix [[a, c], [c, d]] (the lower entry c is used): eigenvalues descending and
 * unit eigenvectors (`vectors[j]` for `values[j]`), signed as `eigh` signs them.
 */
export function eigh2(m: Mat2): { values: Vec2; vectors: [Vec2, Vec2] } {
  const [[a], [c, d]] = m
  const mid = (a + d) / 2
  const radius = Math.hypot((a - d) / 2, c)
  // The eigenvector of the larger eigenvalue is at angle ½·atan2(2c, a − d).
  const angle = 0.5 * Math.atan2(2 * c, a - d)
  const v1: Vec2 = [Math.cos(angle), Math.sin(angle)]
  return { values: [mid + radius, mid - radius], vectors: [signed(v1), signed([-v1[1], v1[0]])] }
}

/**
 * Eigenvalues of a general 2×2 matrix, from the characteristic polynomial λ² − tr·λ + det. Real ones come with unit
 * eigenvectors, larger first; a complex pair λ = re ± i·im comes without eigenvectors.
 */
export type Eig2 = { kind: 'real'; values: Vec2; vectors: [Vec2, Vec2] } | { kind: 'complex'; re: number; im: number }

/** Eigenvalues (and real eigenvectors) of a general 2×2 matrix; see `Eig2`. */
export function eig2(m: Mat2): Eig2 {
  const [[a, b], [c, d]] = m
  const half = (a + d) / 2
  // (tr/2)² − det, written as ((a − d)/2)² + bc to avoid cancellation when the eigenvalues are close.
  const disc = ((a - d) / 2) ** 2 + b * c
  if (disc < 0) return { kind: 'complex', re: half, im: Math.sqrt(-disc) }
  const root = Math.sqrt(disc)
  const values: Vec2 = [half + root, half - root]
  // (A − λI)v = 0: v is perpendicular to the larger row of A − λI.
  const eigenvector = (j: 0 | 1): Vec2 => {
    const l = values[j]
    const r1: Vec2 = [a - l, b]
    const r2: Vec2 = [c, d - l]
    const row = Math.hypot(...r1) >= Math.hypot(...r2) ? r1 : r2
    const n = Math.hypot(...row)
    // A − λI = 0 (A = λI): every vector is an eigenvector.
    if (n === 0) return j === 0 ? [1, 0] : [0, 1]
    return signed([-row[1] / n, row[0] / n])
  }
  return { kind: 'real', values, vectors: [eigenvector(0), eigenvector(1)] }
}

/**
 * Lower Cholesky factor of a symmetric positive-definite [[a, c], [c, d]] (the lower entry c is used), or null when
 * a pivot is not positive. Use `cholesky` for jitter.
 */
export function cholesky2(m: Mat2): Mat2 | null {
  const [[a], [c, d]] = m
  if (!(a > 0)) return null
  const l11 = Math.sqrt(a)
  const l21 = c / l11
  const s = d - l21 * l21
  if (!(s > 0)) return null
  return [
    [l11, 0],
    [l21, Math.sqrt(s)],
  ]
}

/**
 * Closed-form SVD A = U diag(s) Vᵀ of a 2×2 matrix, with s descending and non-negative, `u[j]` and `v[j]` the j-th left
 * and right singular vectors, each pair signed so that v's largest component is positive. It works from the rotation
 * and reflection parts of A (Blinn, 1996, "Consider the lowly 2×2 matrix"), not from AᵀA, so small singular values
 * keep their accuracy.
 */
export function svd2(m: Mat2): { s: Vec2; u: [Vec2, Vec2]; v: [Vec2, Vec2] } {
  const [[a, b], [c, d]] = m
  // A = [[E + F, −H + G], [H + G, E − F]] splits into a scaled rotation (E, H) and a scaled reflection (F, G).
  const E = (a + d) / 2
  const F = (a - d) / 2
  const G = (c + b) / 2
  const H = (c - b) / 2
  const Q = Math.hypot(E, H)
  const R = Math.hypot(F, G)
  const a1 = Math.atan2(G, F)
  const a2 = Math.atan2(H, E)
  const theta = (a2 - a1) / 2
  const phi = (a2 + a1) / 2
  // A = Rot(φ) diag(Q + R, Q − R) Rot(θ).
  let s2 = Q - R
  const u1: Vec2 = [Math.cos(phi), Math.sin(phi)]
  let u2: Vec2 = [-Math.sin(phi), Math.cos(phi)]
  const v1: Vec2 = [Math.cos(theta), -Math.sin(theta)]
  const v2: Vec2 = [Math.sin(theta), Math.cos(theta)]
  if (s2 < 0) {
    s2 = -s2
    u2 = [-u2[0], -u2[1]]
  }
  const pair = (u: Vec2, v: Vec2): [Vec2, Vec2] => {
    const w = signed(v)
    return w[0] === v[0] && w[1] === v[1] ? [u, v] : [[-u[0], -u[1]], w]
  }
  const [U1, V1] = pair(u1, v1)
  const [U2, V2] = pair(u2, v2)
  return { s: [Q + R, s2], u: [U1, U2], v: [V1, V2] }
}
