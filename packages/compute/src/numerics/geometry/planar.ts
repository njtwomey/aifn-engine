/**
 * Planar geometry on point sets: convex hulls (Andrew, 1979, "Another efficient algorithm for convex hulls in two
 * dimensions", Inf. Process. Lett. 9(5)), polygon area and centroid (the shoelace formula) and point-in-polygon tests
 * (the even–odd crossing rule; O'Rourke, 1998, "Computational Geometry in C", 2nd ed., §7.4).
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/** Read n × 2 points (an n × 2 matrix, or an array of [x, y] pairs) into a flat array [x0, y0, x1, y1, …]. */
export function readPoints(p: MatrixLike, what: string): Float64Array {
  if (dense.toMatrixF64(p, what).m === 0) return new Float64Array(0)
  return dense.toMatrixF64(p, what, undefined, 2).data
}

/** A convex hull. */
export interface Hull {
  /** Indices of the hull's vertices in the input, anticlockwise from the lowest-x (then lowest-y) point; int32. */
  indices: Tensor
  /** The vertices in that order, h × 2 (not closed). */
  points: Tensor
  /** Enclosed area. */
  area: number
}

/**
 * The convex hull by Andrew's monotone chain (Andrew, 1979, "Another efficient algorithm for convex hulls in two
 * dimensions", Information Processing Letters 9(5)), O(n log n). Collinear points on an edge are dropped.
 */
export function convexHull(points: MatrixLike): Hull {
  const p = readPoints(points, 'convexHull')
  const n = p.length / 2
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (i, j) => p[2 * i] - p[2 * j] || p[2 * i + 1] - p[2 * j + 1],
  )
  const cross = (o: number, a: number, b: number) =>
    (p[2 * a] - p[2 * o]) * (p[2 * b + 1] - p[2 * o + 1]) - (p[2 * a + 1] - p[2 * o + 1]) * (p[2 * b] - p[2 * o])
  const chain = (idx: number[]) => {
    const out: number[] = []
    for (const i of idx) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], i) <= 0) out.pop()
      out.push(i)
    }
    out.pop()
    return out
  }
  let hull: number[]
  if (n < 3)
    hull = order.filter((i, k) => k === 0 || p[2 * i] !== p[2 * order[0]] || p[2 * i + 1] !== p[2 * order[0] + 1])
  else hull = [...chain(order), ...chain([...order].reverse())]
  const pts = new Float64Array(hull.length * 2)
  hull.forEach((i, k) => {
    pts[2 * k] = p[2 * i]
    pts[2 * k + 1] = p[2 * i + 1]
  })
  return {
    indices: fromData(Int32Array.from(hull)),
    points: fromData(pts, [hull.length, 2]),
    area: Math.abs(shoelace(pts)),
  }
}

function shoelace(p: Float64Array): number {
  const n = p.length / 2
  let s = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    s += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1]
  }
  return s / 2
}

/** Signed area of a simple polygon by the shoelace formula: positive when its vertices run anticlockwise. */
export function polygonArea(polygon: MatrixLike): number {
  return shoelace(readPoints(polygon, 'polygonArea'))
}

/** Centroid (centre of mass) of a simple polygon's area; NaN for a polygon of zero area. */
export function polygonCentroid(polygon: MatrixLike): [number, number] {
  const p = readPoints(polygon, 'polygonCentroid')
  const n = p.length / 2
  const a = shoelace(p)
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const c = p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1]
    cx += (p[2 * i] + p[2 * j]) * c
    cy += (p[2 * i + 1] + p[2 * j + 1]) * c
  }
  return [cx / (6 * a), cy / (6 * a)]
}

/**
 * Whether a point lies inside a polygon, by the even–odd crossing rule (a ray to +x crosses the boundary an odd
 * number of times). Points exactly on the boundary may go either way.
 */
export function pointInPolygon(point: VectorLike, polygon: MatrixLike): boolean {
  const [x, y] = dense.toF64(point, 'pointInPolygon')
  const p = readPoints(polygon, 'pointInPolygon')
  const n = p.length / 2
  let inside = false
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi, xj, yj] = [p[2 * i], p[2 * i + 1], p[2 * j], p[2 * j + 1]]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
