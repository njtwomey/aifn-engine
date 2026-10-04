import { describe, expect, test } from 'vitest'
import {
  applyHomography,
  cameraMatrix,
  epipolarLines,
  fundamentalMatrix,
  homography,
  normalisePoints,
  projectPoints,
  sampsonDistance,
  transferError,
  triangulate,
} from 'aifn-compute/numerics/geometry'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type M = number[][]
const F = fixture<{
  homography: { src: M; dst: M; H: M }
  fundamentalMatrix: { x1: M; x2: M; F: M; K: M; R: M; t: number[]; X: M }
}>('numerics/geometry')

const closeTo = (a: number[], b: number[], tol: number) =>
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol))

describe('homography', () => {
  test('normalised DLT matches scikit-image on noisy correspondences', () => {
    const c = F.homography
    closeTo(toFlat(homography(c.src, c.dst)), c.H.flat(), 1e-8)
  })

  test('four points give an exact fit, and the transfer error vanishes', () => {
    const src = [
      [0, 0],
      [100, 0],
      [100, 80],
      [0, 80],
    ]
    const dst = [
      [10, 5],
      [120, 12],
      [110, 95],
      [3, 85],
    ]
    const H = homography(src, dst)
    closeTo(toFlat(applyHomography(H, src)), dst.flat(), 1e-9)
    expect(Math.max(...toFlat(transferError(H, src, dst)))).toBeLessThan(1e-9)
  })

  test('Hartley normalisation centres the points at root-mean-square distance √2', () => {
    const { points } = normalisePoints(F.homography.src)
    const rows = toRows(points)
    const ms = rows.reduce((s, [x, y]) => s + x * x + y * y, 0) / rows.length
    expect(Math.sqrt(ms)).toBeCloseTo(Math.SQRT2, 12)
    expect(rows.reduce((s, r) => s + r[0], 0)).toBeCloseTo(0, 10)
  })
})

describe('two views', () => {
  const c = F.fundamentalMatrix
  test('the eight-point fundamental matrix matches scikit-image, with rank 2', () => {
    const Fm = fundamentalMatrix(c.x1, c.x2)
    closeTo(toFlat(Fm), c.F.flat(), 1e-8)
    const f = toFlat(Fm)
    const det =
      f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6]) + f[2] * (f[3] * f[7] - f[4] * f[6])
    expect(Math.abs(det)).toBeLessThan(1e-12)
    // Each point lies within a pixel of its epipolar line; the Sampson distance is small for noise σ = 0.2.
    const lines = toRows(epipolarLines(Fm, c.x1))
    lines.forEach(([a, b, d], i) => expect(Math.abs(a * c.x2[i][0] + b * c.x2[i][1] + d)).toBeLessThan(1.5))
    expect(Math.max(...toFlat(sampsonDistance(Fm, c.x1, c.x2)))).toBeLessThan(1)
  })

  test('triangulation recovers noiseless scene points exactly', () => {
    const I = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    const P1 = cameraMatrix(c.K, I, [0, 0, 0])
    const P2 = cameraMatrix(c.K, c.R, c.t)
    const x1 = projectPoints(P1, c.X).points
    const x2 = projectPoints(P2, c.X).points
    closeTo(toFlat(triangulate(P1, P2, x1, x2)), c.X.flat(), 1e-8)
    expect(Math.min(...toFlat(projectPoints(P2, c.X).depth))).toBeGreaterThan(0)
  })
})
