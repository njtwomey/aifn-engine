/**
 * A two-view scene with known ground truth: two pinhole cameras looking at scene points (on a plane, or spread in
 * depth), their image correspondences with pixel noise, and a fraction replaced by outliers. The true homography (for
 * a planar scene) and fundamental matrix come with it, so a figure can compare what RANSAC recovers with the truth.
 * The RANSAC problems for a homography and a fundamental matrix connect `aifn-compute/numerics/geometry` to
 * `aifn-compute/numerics/robust`.
 */

import {
  cameraMatrix,
  fundamentalMatrix,
  homography,
  projectPoints,
  rotationMatrix,
  sampsonDistance,
  transferError,
} from 'aifn-compute/numerics/geometry'
import type { RansacProblem } from 'aifn-compute/numerics/robust'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike } from 'aifn-compute/foundation/contracts'

/** Options for `twoViewScene`. */
export interface TwoViewOptions {
  /** Number of correspondences. Default 80. */
  count?: number
  /** Fraction replaced by random point pairs. Default 0.3. */
  outlierFraction?: number
  /** Pixel noise σ on the inliers. Default 0.5. */
  noise?: number
  /** `plane` (all scene points on one plane: a homography relates the views) or `depth` (spread in depth). */
  kind?: 'plane' | 'depth'
  /** The second camera's rotation about the vertical axis (radians) and sideways baseline. Defaults 0.25 and 1. */
  rotation?: number
  baseline?: number
  /** Image size in pixels (square). Default 400. */
  size?: number
}

/** A two-view scene and its correspondences. */
export interface TwoViewScene {
  K: Matrix
  P1: Matrix
  P2: Matrix
  /** Scene points (n × 3) and their images in each view (n × 2), outliers included. */
  X: Matrix
  x1: Matrix
  x2: Matrix
  /** 1 for an outlier, 0 for an inlier. */
  outlier: number[]
  /** The true homography (plane scenes only, else null) and the true fundamental matrix. */
  H: Matrix | null
  F: Matrix
  size: number
}

const mat3 = (m: Tensor) => dense.data(m)

/**
 * A synthetic two-view scene. Camera 1 is at the origin looking down +Z; camera 2 is moved sideways by `baseline` and
 * turned by `rotation` about the vertical axis toward the scene. Scene points lie on a plane tilted in depth (or fill a
 * box in depth), at Z ≈ 5 … 8. Correspondences are projected, perturbed by Gaussian noise, and a fraction are replaced
 * by uniform random pairs. The true F = K⁻ᵀ[t]ₓRK⁻¹ and, for a plane n·X = d (camera-1 frame), H = K(R + tnᵀ/d)K⁻¹.
 */
export function twoViewScene(s: Stream, options: TwoViewOptions = {}): TwoViewScene {
  const n = options.count ?? 80
  const size = options.size ?? 400
  const f = size * 0.9
  const K = [
    [f, 0, size / 2],
    [0, f, size / 2],
    [0, 0, 1],
  ]
  const I3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
  const R = rotationMatrix([0, 1, 0], -(options.rotation ?? 0.25))
  const C2 = [options.baseline ?? 1, 0, 0]
  // t = −R C for a camera at C.
  const r = mat3(R)
  const t = [0, 1, 2].map((i) => -(r[3 * i] * C2[0] + r[3 * i + 1] * C2[1] + r[3 * i + 2] * C2[2]))
  const P1 = cameraMatrix(K, I3, [0, 0, 0])
  const P2 = cameraMatrix(K, R, t)
  const kind = options.kind ?? 'plane'
  const u = toFlat(uniform(child(s, 'points'), -1, 1, { shape: [n, 3] }) as Tensor)
  const X = new Float64Array(3 * n)
  // The plane Z = 6.5 + 0.4X − 0.3Y, i.e. nᵀX = d with n = (−0.4, 0.3, 1), d = 6.5.
  const plane = { n: [-0.4, 0.3, 1], d: 6.5 }
  for (let i = 0; i < n; i++) {
    const x = 2.2 * u[3 * i]
    const y = 2.2 * u[3 * i + 1]
    const z = kind === 'plane' ? 6.5 + 0.4 * x - 0.3 * y : 6.5 + 1.5 * u[3 * i + 2]
    X.set([x, y, z], 3 * i)
  }
  const Xm = fromData(X, [n, 3])
  const p1 = dense.data(projectPoints(P1, Xm).points)
  const p2 = dense.data(projectPoints(P2, Xm).points)
  const sigma = options.noise ?? 0.5
  const e1 = toFlat(normal(child(s, 'noise1'), 0, 1, { shape: [2 * n] }) as Tensor)
  const e2 = toFlat(normal(child(s, 'noise2'), 0, 1, { shape: [2 * n] }) as Tensor)
  const x1 = Float64Array.from(p1, (v, k) => v + sigma * e1[k])
  const x2 = Float64Array.from(p2, (v, k) => v + sigma * e2[k])
  const frac = Math.min(1, Math.max(0, options.outlierFraction ?? 0.3))
  const nOut = Math.round(frac * n)
  const order = toFlat(uniform(child(s, 'outliers'), 0, 1, { shape: [n] }) as Tensor)
    .map((v, i) => [v, i] as const)
    .sort((a, b) => a[0] - b[0])
    .slice(0, nOut)
    .map(([, i]) => i)
  const junk = toFlat(uniform(child(s, 'junk'), 0, size, { shape: [nOut, 2] }) as Tensor)
  const outlier = new Array<number>(n).fill(0)
  order.forEach((i, k) => {
    outlier[i] = 1
    x2[2 * i] = junk[2 * k]
    x2[2 * i + 1] = junk[2 * k + 1]
  })
  // True F = K⁻ᵀ [t]ₓ R K⁻¹ and, for the plane, H = K (R + t nᵀ/d) K⁻¹.
  const k = mat3(fromData(Float64Array.from(K.flat()), [3, 3]))
  const kInv = Float64Array.of(1 / f, 0, -size / 2 / f, 0, 1 / f, -size / 2 / f, 0, 0, 1)
  const tx = Float64Array.of(0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0)
  const m3 = (a: ArrayLike<number>, b: ArrayLike<number>) => dense.matMul(a, b, 3, 3, 3)
  const Fm = m3(m3(dense.transpose(kInv, 3, 3), m3(tx, r)), kInv)
  const fn = Math.hypot(...Fm)
  let big = 0
  for (let j = 1; j < 9; j++) if (Math.abs(Fm[j]) > Math.abs(Fm[big])) big = j
  const F = fromData(
    Fm.map((v) => (v * (Fm[big] < 0 ? -1 : 1)) / fn),
    [3, 3],
  )
  let H: Matrix | null = null
  if (kind === 'plane') {
    const A = Float64Array.from(r, (v, j) => v + (t[Math.floor(j / 3)] * plane.n[j % 3]) / plane.d)
    const Hm = m3(m3(k, A), kInv)
    H = fromData(
      Hm.map((v) => v / Hm[8]),
      [3, 3],
    )
  }
  return {
    K: fromData(Float64Array.from(K.flat()), [3, 3]),
    P1,
    P2,
    X: Xm,
    x1: fromData(x1, [n, 2]),
    x2: fromData(x2, [n, 2]),
    outlier,
    H,
    F,
    size,
  }
}

/** The rows of an n × 2 matrix at the given indices. */
function rows(x: Float64Array, idx: readonly number[]): Matrix {
  const out = new Float64Array(2 * idx.length)
  idx.forEach((i, k) => out.set([x[2 * i], x[2 * i + 1]], 2 * k))
  return fromData(out, [idx.length, 2])
}

/** The RANSAC problem of a homography between correspondences x1 → x2: four-point DLT, transfer error in pixels. */
export function homographyProblem(x1: MatrixLike, x2: MatrixLike): RansacProblem<Matrix> {
  const a = dense.toMatrixF64(x1, 'homographyProblem x1').data
  const b = dense.toMatrixF64(x2, 'homographyProblem x2').data
  const n = a.length / 2
  const fit = (idx: readonly number[]) => {
    try {
      const H = homography(rows(a, idx), rows(b, idx))
      return dense.allFinite(dense.data(H)) ? H : null
    } catch {
      return null
    }
  }
  return {
    count: n,
    sampleSize: 4,
    fit,
    residuals: (H) => dense.data(transferError(H, fromData(a, [n, 2]), fromData(b, [n, 2]))),
  }
}

/** The RANSAC problem of a fundamental matrix: eight-point fits, the square root of the Sampson distance in pixels. */
export function fundamentalProblem(x1: MatrixLike, x2: MatrixLike): RansacProblem<Matrix> {
  const a = dense.toMatrixF64(x1, 'fundamentalProblem x1').data
  const b = dense.toMatrixF64(x2, 'fundamentalProblem x2').data
  const n = a.length / 2
  const fit = (idx: readonly number[]) => {
    try {
      const F = fundamentalMatrix(rows(a, idx), rows(b, idx))
      return dense.allFinite(dense.data(F)) ? F : null
    } catch {
      return null
    }
  }
  return {
    count: n,
    sampleSize: 8,
    fit,
    residuals: (F) => dense.data(sampsonDistance(F, fromData(a, [n, 2]), fromData(b, [n, 2]))).map(Math.sqrt),
  }
}
