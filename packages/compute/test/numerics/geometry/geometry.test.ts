import { describe, expect, it } from 'vitest'
import {
  barycentricToCartesian,
  cartesianToBarycentric,
  contourLevels,
  contourLines,
  contourSegments,
  convexHull,
  covarianceEllipse,
  evaluateGrid,
  grid2d,
  massToRadius,
  pointInPolygon,
  polygonArea,
  polygonCentroid,
  precisionEllipse,
  simplexGrid,
  simplexVertices,
} from 'aifn-compute/numerics/geometry'
import { logspace, meshgrid } from 'aifn-compute/foundation/tensor'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('numerics/geometry')

describe('ellipses', () => {
  it('a covariance ellipse has semi-axes k√λ and every point at Mahalanobis radius k', () => {
    const cov = [
      [4, 1.2],
      [1.2, 1],
    ]
    const e = covarianceEllipse([1, -1], cov, { k: 2, points: 60 })
    expect(e.points.shape).toEqual([61, 2])
    const det = 4 - 1.44
    const inv = [
      [1 / det, -1.2 / det],
      [-1.2 / det, 4 / det],
    ]
    const p = toFlat(e.points)
    for (let i = 0; i < 60; i++) {
      const [dx, dy] = [p[2 * i] - 1, p[2 * i + 1] + 1]
      const m = dx * dx * inv[0][0] + 2 * dx * dy * inv[0][1] + dy * dy * inv[1][1]
      expect(m).toBeCloseTo(4, 10)
    }
    const trace = 5
    const disc = Math.sqrt((4 - 1) ** 2 / 4 + 1.44)
    expect(e.radii[0]).toBeCloseTo(2 * Math.sqrt(trace / 2 + disc), 10)
    expect(e.mass).toBeCloseTo(1 - Math.exp(-2), 12)
  })

  it('a precision ellipse equals the covariance ellipse of the inverse; mass sets the radius', () => {
    const a = covarianceEllipse(
      [0, 0],
      [
        [2, 0],
        [0, 0.5],
      ],
    )
    const b = precisionEllipse(
      [0, 0],
      [
        [0.5, 0],
        [0, 2],
      ],
    )
    expect(b.radii[0]).toBeCloseTo(a.radii[0], 12)
    expect(b.radii[1]).toBeCloseTo(a.radii[1], 12)
    expect(massToRadius(0.95)).toBeCloseTo(Math.sqrt(5.991464547107979), 10)
    expect(
      covarianceEllipse(
        [0, 0],
        [
          [1, 0],
          [0, 1],
        ],
        { mass: 0.5 },
      ).k,
    ).toBeCloseTo(massToRadius(0.5), 12)
  })
})

describe('polygons', () => {
  it('the convex hull matches scipy', () => {
    const h = convexHull(F.hull.points)
    expect([...toFlat(h.indices)].sort((a, b) => a - b)).toEqual(F.hull.vertices)
    expect(h.area).toBeCloseTo(F.hull.area, 12)
    expect(polygonArea(h.points)).toBeGreaterThan(0)
  })

  it('area, centroid and point-in-polygon of a square', () => {
    const sq = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ]
    expect(polygonArea(sq)).toBe(4)
    expect(polygonArea([...sq].reverse())).toBe(-4)
    expect(polygonCentroid(sq)).toEqual([1, 1])
    expect(pointInPolygon([1, 1], sq)).toBe(true)
    expect(pointInPolygon([3, 1], sq)).toBe(false)
  })
})

describe('contours', () => {
  it('the contour of x² + y² at 0.81 is the circle of radius 0.9', () => {
    const x = linspace(-2, 2, 81)
    const z = evaluateGrid((a, b) => a * a + b * b, x, x)
    const segs = toFlat(contourSegments(x, x, z, 0.81))
    for (let k = 0; k < segs.length; k += 2) expect(Math.hypot(segs[k], segs[k + 1])).toBeCloseTo(0.9, 2)
    const lines = contourLines(x, x, z, 0.81)
    expect(lines.length).toBe(1)
    const l = toFlat(lines[0])
    expect(l[0]).toBeCloseTo(l[l.length - 2], 12)
    expect(l[1]).toBeCloseTo(l[l.length - 1], 12)
    expect(Math.abs(polygonArea(lines[0]))).toBeCloseTo(Math.PI * 0.81, 2)
  })

  it('levels sit strictly inside the range', () => {
    expect(
      toFlat(
        contourLevels(
          [
            [0, 1],
            [2, 3],
          ],
          2,
        ),
      ),
    ).toEqual([1, 2])
  })
})

describe('grids and decimation', () => {
  it('meshgrid and logspace match numpy', () => {
    const [X, Y] = meshgrid(F.meshgrid.x, F.meshgrid.y)
    expect(toFlat(X)).toEqual((F.meshgrid.xx as number[][]).flat())
    expect(toFlat(Y)).toEqual((F.meshgrid.yy as number[][]).flat())
    const [Xi, Yi] = meshgrid(F.meshgrid.x, F.meshgrid.y, { indexing: 'ij' })
    expect(toFlat(Xi)).toEqual((F.meshgrid.xi as number[][]).flat())
    expect(toFlat(Yi)).toEqual((F.meshgrid.yi as number[][]).flat())
    toFlat(logspace(-2, 1, 7)).forEach((v, i) => expect(v).toBeCloseTo(F.logspace[i], 12))
    const g = grid2d([0, 1], [0, 2], 3, 2)
    expect(g.points.shape).toEqual([6, 2])
    expect(toFlat(g.points).slice(6, 8)).toEqual([0, 2])
  })
})

describe('simplex', () => {
  it('barycentric and Cartesian coordinates invert each other', () => {
    const w = [
      [1, 0, 0],
      [0.2, 0.3, 0.5],
    ]
    const p = barycentricToCartesian(w)
    expect(toFlat(p).slice(0, 2)).toEqual([0, 0])
    toFlat(cartesianToBarycentric(p)).forEach((v, i) => expect(v).toBeCloseTo(w.flat()[i], 12))
    expect(simplexVertices().shape).toEqual([3, 2])
  })

  it('the simplex grid has (r+1)(r+2)/2 points and r² triangles', () => {
    const g = simplexGrid(6)
    expect(g.weights.shape).toEqual([28, 3])
    expect(g.triangles.shape).toEqual([36, 3])
    const inner = simplexGrid(4, { interior: true })
    expect(Math.min(...toFlat(inner.weights))).toBeGreaterThan(0)
  })
})
