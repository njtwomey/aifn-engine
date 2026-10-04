/**
 * Projective geometry of one and two views (Hartley & Zisserman, 2004, "Multiple View Geometry in Computer Vision",
 * 2nd ed.): pinhole cameras $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$, the plane-to-plane homography by the normalised direct linear transform
 * (§4.4), the fundamental matrix by the normalised eight-point algorithm (Hartley, 1997, "In defense of the eight-point
 * algorithm", IEEE TPAMI 19(6)), epipolar lines, the Sampson distance, and linear triangulation (§12.2).
 *
 * Image points are rows $(x, y)$ of an $n \times 2$ matrix; scene points rows $(X, Y, Z)$ of an $n \times 3$ matrix.
 */

import { dense, fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { svd } from 'aifn-compute/numerics/linalg'

type F64 = dense.F64

/**
 * Extract and validate 2D or 3D point data from a matrix-like container.
 *
 * @param x - Input coordinate matrix.
 * @param d - Expected dimensionality ($2$ or $3$).
 * @param where - Calling function name for errors.
 * @returns Flat coordinate buffer and number of points $n$.
 */
function points(x: MatrixLike, d: number, where: string): { p: F64; n: number } {
  const m = dense.toMatrixF64(x, where)
  if (m.n !== d) throw new ShapeError(where, `${where}: points need ${d} columns, got ${m.n}`)
  return { p: m.data, n: m.m }
}

/**
 * The unit vector minimising $\|\mathbf{A}\mathbf{v}\|$ for $\mathbf{A}$ ($m \times k$): the right singular vector of the smallest singular value ($\mathbf{A}$ is padded
 * with zero rows to at least $k$ rows, so a wide system still yields all $k$ right singular vectors).
 *
 * @param A - Flat matrix buffer of size $m \times k$.
 * @param m - Number of rows.
 * @param k - Number of columns.
 * @returns Unit right singular vector of length $k$.
 */
function nullVector(A: F64, m: number, k: number): F64 {
  const rows = Math.max(m, k)
  const padded = new Float64Array(rows * k)
  padded.set(A)
  const V = dense.data(svd(fromData(padded, [rows, k])).V)
  return Float64Array.from({ length: k }, (_, i) => V[i * k + (k - 1)])
}

/**
 * Hartley's normalisation of 2-D points: the similarity $\mathbf{T}$ that moves their centroid to the origin and scales their
 * root-mean-square distance from it to $\sqrt{2}$ (as scikit-image; Hartley, 1997, scales the mean distance, which is nearly the
 * same). Returns the normalised points and $\mathbf{T}$ ($3 \times 3$).
 *
 * @param x - Input 2D coordinates as an $n \times 2$ matrix.
 * @returns Object containing normalised points and the $3 \times 3$ transformation matrix $\mathbf{T}$.
 *
 * @example Hartley point normalisation
 * const pts = [
 *   [10, 20],
 *   [12, 22],
 *   [8, 18],
 *   [14, 24],
 * ]
 * const { points, T } = normalisePoints(pts)
 * print('Normalised points:\n' + points)
 * print('Transform T:\n' + T)
 */
export function normalisePoints(x: MatrixLike): { points: Matrix; T: Matrix } {
  const { p, n } = points(x, 2, 'normalisePoints')
  if (n < 1) throw new DomainError('normalisePoints', 'normalisePoints: no points')
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) {
    cx += p[2 * i] / n
    cy += p[2 * i + 1] / n
  }
  let ms = 0
  for (let i = 0; i < n; i++) ms += ((p[2 * i] - cx) ** 2 + (p[2 * i + 1] - cy) ** 2) / n
  const s = ms > 0 ? Math.SQRT2 / Math.sqrt(ms) : 1
  const out = Float64Array.from(p, (v, k) => s * (v - (k % 2 ? cy : cx)))
  return { points: fromData(out, [n, 2]), T: fromData(Float64Array.of(s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1), [3, 3]) }
}

/**
 * Invert a $3 \times 3$ matrix stored in a flat array.
 *
 * @param m - 9-element flat array representing a $3 \times 3$ matrix.
 * @returns Inverted $3 \times 3$ matrix as a 9-element array.
 */
const inv3 = (m: F64): F64 => {
  const [a, b, c, d, e, f, g, h, i] = m
  const A = e * i - f * h
  const B = -(d * i - f * g)
  const C = d * h - e * g
  const det = a * A + b * B + c * C
  if (det === 0) throw new DomainError('geometry', 'geometry: a singular 3 × 3 matrix')
  return Float64Array.of(
    A / det,
    -(b * i - c * h) / det,
    (b * f - c * e) / det,
    B / det,
    (a * i - c * g) / det,
    -(a * f - c * d) / det,
    C / det,
    -(a * h - b * g) / det,
    (a * e - b * d) / det,
  )
}

/**
 * Multiply two $3 \times 3$ matrices stored in flat arrays.
 *
 * @param a - First $3 \times 3$ matrix.
 * @param b - Second $3 \times 3$ matrix.
 * @returns Product matrix $\mathbf{A}\mathbf{B}$.
 */
const mul3 = (a: F64, b: F64): F64 => dense.matMul(a, b, 3, 3, 3)

/**
 * The homography $\mathbf{H}$ ($3 \times 3$, $\mathbf{H}_{33} = 1$) with $\mathbf{x}' \propto \mathbf{H}\mathbf{x}$ from $n \ge 4$ correspondences, by the normalised direct linear
 * transform (Hartley & Zisserman, 2004, Alg. 4.2): normalise both point sets, stack the two equations
 * $\mathbf{x}' \times \mathbf{H}\mathbf{x} = \mathbf{0}$ contributed per correspondence into $\mathbf{A}$ ($2n \times 9$), take the null vector of $\mathbf{A}$ (the smallest singular vector),
 * and undo the normalisations, $\mathbf{H} = \mathbf{T}'^{-1} \mathbf{\tilde{H}} \mathbf{T}$. With exactly four points in general position the fit is exact.
 *
 * @param src - Source 2D points as an $n \times 2$ matrix.
 * @param dst - Destination 2D points as an $n \times 2$ matrix.
 * @returns The estimated $3 \times 3$ homography matrix.
 *
 * @example Estimate homography between planar points
 * const src = [
 *   [0, 0],
 *   [1, 0],
 *   [1, 1],
 *   [0, 1],
 * ]
 * const dst = [
 *   [1, 1],
 *   [3, 1],
 *   [2.5, 3],
 *   [0.5, 2.5],
 * ]
 * const H = homography(src, dst)
 * print('Homography matrix:\n' + H)
 */
export function homography(src: MatrixLike, dst: MatrixLike): Matrix {
  const a = normalisePoints(src)
  const b = normalisePoints(dst)
  const n = a.points.shape[0]
  if (n !== b.points.shape[0]) throw new ShapeError('homography', 'homography: unequal numbers of points')
  if (n < 4) throw new DomainError('homography', 'homography: needs at least 4 correspondences')
  const p = dense.data(a.points)
  const q = dense.data(b.points)
  const A = new Float64Array(2 * n * 9)
  for (let i = 0; i < n; i++) {
    const [x, y] = [p[2 * i], p[2 * i + 1]]
    const [u, v] = [q[2 * i], q[2 * i + 1]]
    A.set([0, 0, 0, -x, -y, -1, v * x, v * y, v], 18 * i)
    A.set([x, y, 1, 0, 0, 0, -u * x, -u * y, -u], 18 * i + 9)
  }
  const h = nullVector(A, 2 * n, 9)
  const H = mul3(mul3(inv3(dense.data(b.T)), h), dense.data(a.T))
  const s = H[8] !== 0 ? H[8] : 1
  return fromData(
    H.map((v) => v / s),
    [3, 3],
  )
}

/**
 * Points mapped by a homography: $\mathbf{x}' = \mathbf{H}\mathbf{x}$ in homogeneous coordinates, returned dehomogenised ($n \times 2$).
 *
 * @param H - $3 \times 3$ homography matrix.
 * @param x - Input 2D points as an $n \times 2$ matrix.
 * @returns Transformed 2D points as an $n \times 2$ matrix.
 *
 * @example Apply homography to 2D coordinates
 * const H = [
 *   [2, 0, 1],
 *   [0, 2, 3],
 *   [0, 0, 1],
 * ]
 * const pts = [
 *   [0, 0],
 *   [1, 1],
 * ]
 * const mapped = applyHomography(H, pts)
 * print('Mapped points:\n' + mapped)
 */
export function applyHomography(H: MatrixLike, x: MatrixLike): Matrix {
  const h = dense.toMatrixF64(H, 'applyHomography H', 3, 3).data
  const { p, n } = points(x, 2, 'applyHomography')
  const out = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const [u, v] = [p[2 * i], p[2 * i + 1]]
    const w = h[6] * u + h[7] * v + h[8]
    out[2 * i] = (h[0] * u + h[1] * v + h[2]) / w
    out[2 * i + 1] = (h[3] * u + h[4] * v + h[5]) / w
  }
  return fromData(out, [n, 2])
}

/**
 * The transfer error $\|\mathbf{x}' - \mathbf{H}\mathbf{x}\|$ of each correspondence (pixels), $n$ values.
 *
 * @param H - $3 \times 3$ homography matrix.
 * @param src - Source 2D points as an $n \times 2$ matrix.
 * @param dst - Target 2D points as an $n \times 2$ matrix.
 * @returns Vector of Euclidean transfer errors for each correspondence.
 *
 * @example Compute transfer error of homography
 * const pts = [
 *   [0, 0],
 *   [1, 0],
 *   [1, 1],
 *   [0, 1],
 * ]
 * const H = homography(pts, pts)
 * const err = transferError(H, pts, pts)
 * print('Errors:', err)
 */
export function transferError(H: MatrixLike, src: MatrixLike, dst: MatrixLike): Vector {
  const m = dense.data(applyHomography(H, src))
  const { p, n } = points(dst, 2, 'transferError')
  return fromData(
    Float64Array.from({ length: n }, (_, i) => Math.hypot(m[2 * i] - p[2 * i], m[2 * i + 1] - p[2 * i + 1])),
    [n],
  )
}

/**
 * The fundamental matrix $\mathbf{F}$ ($3 \times 3$, rank 2, $\|\mathbf{F}\|_F = 1$) with $\mathbf{x}_2^\top \mathbf{F}\mathbf{x}_1 = 0$, from $n \ge 8$ correspondences by the normalised
 * eight-point algorithm (Hartley, 1997): normalise each image's points, solve the linear system for the null vector,
 * enforce rank 2 by zeroing the smallest singular value, and denormalise, $\mathbf{F} = \mathbf{T}_2^\top \mathbf{\tilde{F}} \mathbf{T}_1$.
 *
 * @param x1 - Points in first image as an $n \times 2$ matrix ($n \ge 8$).
 * @param x2 - Corresponding points in second image as an $n \times 2$ matrix.
 * @returns Estimated $3 \times 3$ rank-2 fundamental matrix.
 *
 * @example Fundamental matrix estimation
 * const x1 = [
 *   [0, 0], [10, 0], [10, 10], [0, 10],
 *   [5, 5], [2, 8], [8, 2], [7, 3],
 * ]
 * const x2 = [
 *   [1, 2], [11, 2], [11, 12], [1, 12],
 *   [6, 7], [3, 10], [9, 4], [8, 5],
 * ]
 * const F = fundamentalMatrix(x1, x2)
 * print('Fundamental matrix:\n' + F)
 */
export function fundamentalMatrix(x1: MatrixLike, x2: MatrixLike): Matrix {
  const a = normalisePoints(x1)
  const b = normalisePoints(x2)
  const n = a.points.shape[0]
  if (n !== b.points.shape[0]) throw new ShapeError('fundamentalMatrix', 'fundamentalMatrix: unequal numbers of points')
  if (n < 8) throw new DomainError('fundamentalMatrix', 'fundamentalMatrix: needs at least 8 correspondences')
  const p = dense.data(a.points)
  const q = dense.data(b.points)
  const A = new Float64Array(n * 9)
  for (let i = 0; i < n; i++) {
    const [x, y] = [p[2 * i], p[2 * i + 1]]
    const [u, v] = [q[2 * i], q[2 * i + 1]]
    A.set([u * x, u * y, u, v * x, v * y, v, x, y, 1], 9 * i)
  }
  const f = nullVector(A, n, 9)
  // Rank 2: F̃ = U diag(s₁, s₂, 0) Vᵀ.
  const d = svd(fromData(f, [3, 3]))
  const U = dense.data(d.U)
  const S = dense.data(d.S)
  const V = dense.data(d.V)
  const F2 = new Float64Array(9)
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) F2[3 * r + c] = U[3 * r] * S[0] * V[3 * c] + U[3 * r + 1] * S[1] * V[3 * c + 1]
  const T1 = dense.data(a.T)
  const T2t = dense.transpose(dense.data(b.T), 3, 3)
  const F = mul3(mul3(T2t, F2), T1)
  const norm = Math.hypot(...F)
  // Sign convention: the largest-magnitude entry positive.
  let big = 0
  for (let k = 1; k < 9; k++) if (Math.abs(F[k]) > Math.abs(F[big])) big = k
  const s = (F[big] < 0 ? -1 : 1) / norm
  return fromData(
    F.map((v) => v * s),
    [3, 3],
  )
}

/**
 * Epipolar lines $(a, b, c)$ with $ax + by + c = 0$, scaled so $a^2 + b^2 = 1$: in image 2 for points of image 1 ($\mathbf{l}_2 = \mathbf{F}\mathbf{x}_1$),
 * or in image 1 for points of image 2 (`image: 1`, $\mathbf{l}_1 = \mathbf{F}^\top \mathbf{x}_2$). $n \times 3$.
 *
 * @param F - $3 \times 3$ fundamental matrix.
 * @param x - Input 2D points as an $n \times 2$ matrix.
 * @param options - Configuration options.
 * @param options.image - Target image for lines ($1$ or $2$, default $2$).
 * @returns Epipolar line coefficients as an $n \times 3$ matrix.
 *
 * @example Epipolar lines
 * const F = [
 *   [0, 0, 0],
 *   [0, 0, -1],
 *   [0, 1, 0],
 * ]
 * const pts = [[10, 20]]
 * const lines = epipolarLines(F, pts)
 * print('Epipolar line:', lines)
 */
export function epipolarLines(F: MatrixLike, x: MatrixLike, { image = 2 }: { image?: 1 | 2 } = {}): Matrix {
  const f = dense.toMatrixF64(F, 'epipolarLines F', 3, 3).data
  const M = image === 2 ? f : dense.transpose(f, 3, 3)
  const { p, n } = points(x, 2, 'epipolarLines')
  const out = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) {
    const v = [p[2 * i], p[2 * i + 1], 1]
    const l = [0, 1, 2].map((r) => M[3 * r] * v[0] + M[3 * r + 1] * v[1] + M[3 * r + 2] * v[2])
    const s = Math.hypot(l[0], l[1]) || 1
    out.set(
      l.map((c) => c / s),
      3 * i,
    )
  }
  return fromData(out, [n, 3])
}

/**
 * The Sampson distance of each correspondence to $\mathbf{F}$, the first-order geometric error
 * $(\mathbf{x}_2^\top \mathbf{F}\mathbf{x}_1)^2 / ((\mathbf{F}\mathbf{x}_1)_1^2 + (\mathbf{F}\mathbf{x}_1)_2^2 + (\mathbf{F}^\top \mathbf{x}_2)_1^2 + (\mathbf{F}^\top \mathbf{x}_2)_2^2)$ (Hartley & Zisserman, 2004, §11.4.3), in squared pixels.
 *
 * @param F - $3 \times 3$ fundamental matrix.
 * @param x1 - Points in first view as an $n \times 2$ matrix.
 * @param x2 - Corresponding points in second view as an $n \times 2$ matrix.
 * @returns Vector of Sampson distances (squared geometric error) for each correspondence.
 *
 * @example Sampson distance for correspondences
 * const F = [
 *   [0, 0, 0],
 *   [0, 0, -1],
 *   [0, 1, 0],
 * ]
 * const x1 = [[10, 20]]
 * const x2 = [[10, 20]]
 * const d = sampsonDistance(F, x1, x2)
 * print('Sampson distance:', d)
 */
export function sampsonDistance(F: MatrixLike, x1: MatrixLike, x2: MatrixLike): Vector {
  const f = dense.toMatrixF64(F, 'sampsonDistance F', 3, 3).data
  const a = points(x1, 2, 'sampsonDistance x1')
  const b = points(x2, 2, 'sampsonDistance x2')
  if (a.n !== b.n) throw new ShapeError('sampsonDistance', 'sampsonDistance: unequal numbers of points')
  const out = new Float64Array(a.n)
  for (let i = 0; i < a.n; i++) {
    const u = [a.p[2 * i], a.p[2 * i + 1], 1]
    const v = [b.p[2 * i], b.p[2 * i + 1], 1]
    const Fu = [0, 1, 2].map((r) => f[3 * r] * u[0] + f[3 * r + 1] * u[1] + f[3 * r + 2] * u[2])
    const Ftv = [0, 1, 2].map((c) => f[c] * v[0] + f[3 + c] * v[1] + f[6 + c] * v[2])
    const e = v[0] * Fu[0] + v[1] * Fu[1] + v[2] * Fu[2]
    out[i] = (e * e) / (Fu[0] ** 2 + Fu[1] ** 2 + Ftv[0] ** 2 + Ftv[1] ** 2)
  }
  return fromData(out, [a.n])
}

/**
 * The pinhole camera matrix $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$ ($3 \times 4$) from intrinsics $\mathbf{K}$ ($3 \times 3$), rotation $\mathbf{R}$ and translation $\mathbf{t}$.
 *
 * @param K - $3 \times 3$ camera intrinsics calibration matrix.
 * @param R - $3 \times 3$ camera rotation matrix.
 * @param t - 3-element translation vector.
 * @returns $3 \times 4$ projection camera matrix $\mathbf{P}$.
 *
 * @example Compose pinhole camera matrix
 * const K = [
 *   [1000, 0, 320],
 *   [0, 1000, 240],
 *   [0, 0, 1],
 * ]
 * const R = [
 *   [1, 0, 0],
 *   [0, 1, 0],
 *   [0, 0, 1],
 * ]
 * const t = [0, 0, 10]
 * const P = cameraMatrix(K, R, t)
 * print('Camera matrix shape:', P.shape)
 */
export function cameraMatrix(K: MatrixLike, R: MatrixLike, t: ArrayLike<number>): Matrix {
  const k = dense.toMatrixF64(K, 'cameraMatrix K', 3, 3).data
  const r = dense.toMatrixF64(R, 'cameraMatrix R', 3, 3).data
  const Rt = new Float64Array(12)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) Rt[4 * i + j] = r[3 * i + j]
    Rt[4 * i + 3] = t[i]
  }
  return fromData(dense.matMul(k, Rt, 3, 3, 4), [3, 4])
}

/**
 * Scene points ($n \times 3$) projected by a camera $\mathbf{P}$ ($3 \times 4$) to image points ($n \times 2$); also their depths $w$.
 *
 * @param P - $3 \times 4$ camera projection matrix.
 * @param X - 3D scene coordinates as an $n \times 3$ matrix.
 * @returns Object with projected 2D coordinates `points` ($n \times 2$) and projective `depth` vector ($n$).
 *
 * @example Project 3D points to 2D
 * const P = [
 *   [100, 0, 50, 0],
 *   [0, 100, 50, 0],
 *   [0, 0, 1, 0],
 * ]
 * const X = [[0, 0, 5]]
 * const { points, depth } = projectPoints(P, X)
 * print('Projected:', points)
 * print('Depth:', depth)
 */
export function projectPoints(P: MatrixLike, X: MatrixLike): { points: Matrix; depth: Vector } {
  const p = dense.toMatrixF64(P, 'projectPoints P', 3, 4).data
  const { p: x, n } = points(X, 3, 'projectPoints')
  const out = new Float64Array(2 * n)
  const depth = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const v = [x[3 * i], x[3 * i + 1], x[3 * i + 2], 1]
    const h = [0, 1, 2].map((r) => p[4 * r] * v[0] + p[4 * r + 1] * v[1] + p[4 * r + 2] * v[2] + p[4 * r + 3] * v[3])
    out[2 * i] = h[0] / h[2]
    out[2 * i + 1] = h[1] / h[2]
    depth[i] = h[2]
  }
  return { points: fromData(out, [n, 2]), depth: fromData(depth, [n]) }
}

/**
 * Linear triangulation (Hartley & Zisserman, 2004, §12.2): for each correspondence the scene point $\mathbf{X}$ with $\mathbf{x}_1 \times \mathbf{P}_1 \mathbf{X} = \mathbf{0}$
 * and $\mathbf{x}_2 \times \mathbf{P}_2 \mathbf{X} = \mathbf{0}$, the null vector of the $4 \times 4$ system, dehomogenised. Exact for noiseless data. $n \times 3$.
 *
 * @param P1 - $3 \times 4$ camera matrix for the first view.
 * @param P2 - $3 \times 4$ camera matrix for the second view.
 * @param x1 - Points in first view as an $n \times 2$ matrix.
 * @param x2 - Points in second view as an $n \times 2$ matrix.
 * @returns Triangulated 3D points as an $n \times 3$ matrix.
 *
 * @example Linear triangulation of stereo points
 * const P1 = [
 *   [100, 0, 0, 0],
 *   [0, 100, 0, 0],
 *   [0, 0, 1, 0],
 * ]
 * const P2 = [
 *   [100, 0, 0, -100],
 *   [0, 100, 0, 0],
 *   [0, 0, 1, 0],
 * ]
 * const x1 = [[0, 0]]
 * const x2 = [[-10, 0]]
 * const X = triangulate(P1, P2, x1, x2)
 * print('Triangulated 3D point:\n' + X)
 */
export function triangulate(P1: MatrixLike, P2: MatrixLike, x1: MatrixLike, x2: MatrixLike): Matrix {
  const a = dense.toMatrixF64(P1, 'triangulate P1', 3, 4).data
  const b = dense.toMatrixF64(P2, 'triangulate P2', 3, 4).data
  const u = points(x1, 2, 'triangulate x1')
  const v = points(x2, 2, 'triangulate x2')
  if (u.n !== v.n) throw new ShapeError('triangulate', 'triangulate: unequal numbers of points')
  const out = new Float64Array(3 * u.n)
  for (let i = 0; i < u.n; i++) {
    const A = new Float64Array(16)
    const rows: [F64, number, number][] = [
      [a, u.p[2 * i], u.p[2 * i + 1]],
      [b, v.p[2 * i], v.p[2 * i + 1]],
    ]
    rows.forEach(([P, x, y], k) => {
      for (let j = 0; j < 4; j++) {
        A[2 * k * 4 + j] = x * P[8 + j] - P[j]
        A[(2 * k + 1) * 4 + j] = y * P[8 + j] - P[4 + j]
      }
    })
    // Scale rows to unit norm so the null vector is not dominated by large pixel coordinates.
    for (let r = 0; r < 4; r++) {
      const s = Math.hypot(A[4 * r], A[4 * r + 1], A[4 * r + 2], A[4 * r + 3]) || 1
      for (let j = 0; j < 4; j++) A[4 * r + j] /= s
    }
    const X = nullVector(A, 4, 4)
    out.set([X[0] / X[3], X[1] / X[3], X[2] / X[3]], 3 * i)
  }
  return fromData(out, [u.n, 3])
}

/**
 * The rotation by angle $\theta$ (radians) about a unit axis (Rodrigues' formula), $3 \times 3$.
 *
 * @param axis - 3-element rotation axis vector $[x, y, z]$.
 * @param angle - Rotation angle $\theta$ in radians.
 * @returns $3 \times 3$ orthogonal rotation matrix.
 *
 * @example Compute 3D rotation matrix
 * const R = rotationMatrix([0, 0, 1], Math.PI / 2)
 * print('Rotation by 90 deg around Z:\n' + R)
 */
export function rotationMatrix(axis: ArrayLike<number>, angle: Scalar): Matrix {
  const n = Math.hypot(axis[0], axis[1], axis[2])
  if (!(n > 0)) throw new DomainError('rotationMatrix', 'rotationMatrix: the axis must be nonzero')
  const [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n]
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const C = 1 - c
  return fromData(
    Float64Array.of(
      c + x * x * C,
      x * y * C - z * s,
      x * z * C + y * s,
      y * x * C + z * s,
      c + y * y * C,
      y * z * C - x * s,
      z * x * C - y * s,
      z * y * C + x * s,
      c + z * z * C,
    ),
    [3, 3],
  )
}
