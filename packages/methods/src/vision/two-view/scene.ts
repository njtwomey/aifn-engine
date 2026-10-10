/**
 * A synthetic two-view scene with known ground truth, and the RANSAC problems that fit a homography or a fundamental
 * matrix to its correspondences.
 *
 * Two pinhole cameras $\Pmat_1 = \Kmat[\Imat \mid \zeros]$ and $\Pmat_2 = \Kmat[\Rmat \mid \tvec]$ look at scene
 * points on a plane or spread in depth; their images are perturbed by Gaussian pixel noise, and a fraction of the
 * correspondences are replaced by outliers. The true fundamental matrix
 * $\Fmat = \Kmat^{-\top}[\tvec]_{\times}\Rmat\Kmat^{-1}$ and, for a planar scene, the true homography come with it
 * (Hartley and Zisserman, 2004, §9.2 and §13.1), so a figure can compare what RANSAC recovers with the truth. The
 * problems connect the estimators of `aifn-compute/numerics/geometry` (four-point DLT, the normalised eight-point
 * algorithm) to the `ransac` of `aifn-compute/numerics/robust`.
 *
 * Points are $n \times 2$ matrices of pixel coordinates, row $i$ of the first view corresponding to row $i$ of the
 * second, and the world frame is camera 1's.
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
  /** Number of correspondences $n$. Default 80. */
  count?: number
  /**
   * Fraction of the correspondences that are outliers (clamped to $[0, 1]$; default 0.3): $\operatorname{round}(fn)$
   * of them, chosen at random, have their second-view point replaced by a uniform draw over the image.
   */
  outlierFraction?: number
  /**
   * Standard deviation $\sigma$, in pixels, of the Gaussian noise added to each coordinate of every point in both
   * views (before the outliers are drawn). Default 0.5.
   */
  noise?: number
  /**
   * `'plane'` (default): the scene points lie on one plane, so a homography relates the views. `'depth'`: they fill a
   * box in depth, and only the fundamental matrix does.
   */
  kind?: 'plane' | 'depth'
  /**
   * The angle $\theta$, in radians, of the second camera's turn about the vertical ($Y$) axis: $\Rmat$ is the rotation
   * by $\theta$. Default 0.25.
   */
  rotation?: number
  /** The second camera's sideways offset: its centre is at $(b, 0, 0)$. Default 1. */
  baseline?: number
  /**
   * Image width and height in pixels (default 400): the focal length is $0.9$ times it and the principal point is the
   * image centre.
   */
  size?: number
}

/** A two-view scene and its correspondences. */
export interface TwoViewScene {
  /** The shared intrinsics $\Kmat$ ($3 \times 3$): focal length $0.9 \cdot$ `size`, principal point at the centre. */
  K: Matrix
  /** Camera 1, $\Pmat_1 = \Kmat[\Imat \mid \zeros]$ ($3 \times 4$), at the origin looking down $+Z$. */
  P1: Matrix
  /** Camera 2, $\Pmat_2 = \Kmat[\Rmat \mid \tvec]$ ($3 \times 4$), with $\tvec = -\Rmat\cvec$ ($\cvec$ its centre). */
  P2: Matrix
  /** The scene points ($n \times 3$), in camera 1's frame; an outlier keeps its true scene point. */
  X: Matrix
  /** The noisy images of the scene points in view 1 ($n \times 2$, pixels). */
  x1: Matrix
  /** The noisy images in view 2 ($n \times 2$, pixels), with the outliers' rows replaced by uniform draws. */
  x2: Matrix
  /** Per correspondence, 1 for an outlier and 0 for an inlier. */
  outlier: number[]
  /**
   * The true homography from view 1 to view 2, $\Hmat = \Kmat(\Rmat + \tvec\nvec^\top / d)\Kmat^{-1}$ for the plane
   * $\nvec^\top\mathbf{X} = d$, scaled so that $H_{33} = 1$; null for a `'depth'` scene.
   */
  H: Matrix | null
  /**
   * The true fundamental matrix, $\xvec_2^\top\Fmat\xvec_1 = 0$ for homogeneous image points, scaled to unit
   * Frobenius norm with its largest-magnitude entry positive.
   */
  F: Matrix
  /** The image size in pixels, as given. */
  size: number
}

/**
 * The entries of a $3 \times 3$ matrix as a flat row-major array.
 *
 * @param m The matrix.
 * @returns Its nine entries, row by row.
 */
const mat3 = (m: Tensor) => dense.data(m)

/**
 * A synthetic two-view scene. Camera 1 is at the origin looking down $+Z$; camera 2 is moved sideways to $(b, 0, 0)$
 * ($b$ the `baseline`) and turned about the vertical axis, $\Rmat$ being the rotation by $\theta$ ($\theta$ the
 * `rotation`), turning its optical axis inwards towards the scene. The scene points
 * have $X, Y$ uniform on $[-2.2, 2.2]$ and lie on the plane $Z = 6.5 + 0.4X - 0.3Y$ (that is,
 * $\nvec^\top\mathbf{X} = d$ with $\nvec = (-0.4, 0.3, 1)$, $d = 6.5$) or have $Z$ uniform on $[5, 8]$. They are
 * projected into both views, Gaussian noise is added to every coordinate, and a fraction of the second-view points are
 * replaced by uniform draws over the image. The true $\Fmat = \Kmat^{-\top}[\tvec]_{\times}\Rmat\Kmat^{-1}$ and, for
 * the plane, $\Hmat = \Kmat(\Rmat + \tvec\nvec^\top / d)\Kmat^{-1}$ come with it.
 *
 * @param s The random stream; the points, the noise of each view, the choice of outliers and their positions each
 *   draw from their own child of it.
 * @param options The size of the scene, its noise and outliers, its kind and the second camera's pose; see
 *   `TwoViewOptions`.
 * @returns The cameras, the scene points, the correspondences with their outlier flags, and the true $\Hmat$ and
 *   $\Fmat$.
 *
 * @example Noise-free correspondences satisfy $\xvec_2^\top\Fmat\xvec_1 = 0$
 * const scene = twoViewScene(stream(0), { count: 4, kind: 'depth', noise: 0, outlierFraction: 0 })
 * const [x1, x2, F] = [toArray(scene.x1), toArray(scene.x2), toArray(scene.F)]
 * const epipolar = x1.map(([u, v], i) => {
 *   const Fx = F.map((row) => row[0] * u + row[1] * v + row[2])
 *   return x2[i][0] * Fx[0] + x2[i][1] * Fx[1] + Fx[2]
 * })
 * print('x2ᵀ F x1 =', epipolar)
 *
 * @example A planar scene has a homography; outliers are flagged
 * const scene = twoViewScene(stream(0), { count: 6, outlierFraction: 0.5 })
 * print('H =', scene.H)
 * print('outlier =', scene.outlier)
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
  const R = rotationMatrix([0, 1, 0], options.rotation ?? 0.25)
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

/**
 * The rows of an $n \times 2$ matrix at the given indices, as a new matrix.
 *
 * @param x The matrix as a row-major array of $2n$ values (row $i$ is entries $2i$ and $2i + 1$); not modified.
 * @param idx The row indices to take, in order.
 * @returns A matrix of `idx.length` rows and 2 columns.
 */
function rows(x: Float64Array, idx: readonly number[]): Matrix {
  const out = new Float64Array(2 * idx.length)
  idx.forEach((i, k) => out.set([x[2 * i], x[2 * i + 1]], 2 * k))
  return fromData(out, [idx.length, 2])
}

/**
 * The RANSAC problem of a homography $\Hmat$ mapping the points of view 1 to those of view 2: a sample of four
 * correspondences is fitted by the DLT of `homography`, and each correspondence is scored by its transfer error
 * $\lVert \xvec_2 - \Hmat\xvec_1 \rVert$ in pixels. A sample whose fit throws or is not finite is degenerate (`fit`
 * returns null).
 *
 * @param x1 The points of view 1 ($n \times 2$, pixels).
 * @param x2 The corresponding points of view 2 ($n \times 2$, pixels), row by row.
 * @returns The problem for `ransac`: $n$ data, samples of 4, the fit and the residuals.
 *
 * @example Four noise-free inliers give the homography; the outliers stand out
 * const scene = twoViewScene(stream(0), { count: 10, outlierFraction: 0.2, noise: 0 })
 * const problem = homographyProblem(scene.x1, scene.x2)
 * const inliers = scene.outlier.flatMap((o, i) => (o ? [] : [i]))
 * const H = problem.fit(inliers.slice(0, 4))
 * print('outlier =', scene.outlier)
 * print('transfer error (px) =', problem.residuals(H))
 */
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

/**
 * The RANSAC problem of a fundamental matrix $\Fmat$ with $\xvec_2^\top\Fmat\xvec_1 = 0$: a sample of eight
 * correspondences is fitted by the normalised eight-point algorithm of `fundamentalMatrix`, and each correspondence is
 * scored by the square root of its Sampson distance, in pixels. A sample whose fit throws or is not finite is
 * degenerate (`fit` returns null).
 *
 * @param x1 The points of view 1 ($n \times 2$, pixels).
 * @param x2 The corresponding points of view 2 ($n \times 2$, pixels), row by row.
 * @returns The problem for `ransac`: $n$ data, samples of 8, the fit and the residuals.
 *
 * @example The fundamental matrix from eight point correspondences
 * const scene = twoViewScene(stream(0), { count: 12, kind: 'depth', noise: 0 })
 * const problem = fundamentalProblem(scene.x1, scene.x2)
 * const inliers = scene.outlier.flatMap((o, i) => (o ? [] : [i]))
 * const F = problem.fit(inliers.slice(0, 8))
 * print('F =', F)
 * print('true F =', scene.F)
 * print('outlier =', scene.outlier)
 * print('residual (px) =', problem.residuals(F))
 */
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
