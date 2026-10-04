/**
 * The probability simplex drawn in the plane: barycentric ↔ Cartesian coordinates on a triangle, the standard
 * equilateral triangle, and a triangular grid of the simplex for densities such as the Dirichlet's (barycentric
 * coordinates after Möbius; Coxeter, 1969, "Introduction to Geometry", 2nd ed., §13.7).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/**
 * Vertices of the equilateral triangle of side 1 used to draw the 2-simplex: e₁ at (0, 0), e₂ at (1, 0) and e₃ at
 * (1/2, √3/2), as a 3 × 2 tensor.
 */
export function simplexVertices(): Tensor {
  return fromData(Float64Array.from([0, 0, 1, 0, 0.5, Math.sqrt(3) / 2]), [3, 2])
}

function readVertices(v: Tensor | undefined): Float64Array {
  const out = Float64Array.from(toFlat(v ?? simplexVertices()))
  if (out.length !== 6) throw new DomainError('simplex', 'simplex: vertices must be 3 × 2')
  return out
}

function rows(x: MatrixLike | VectorLike, width: number, what: string) {
  const flat = isTensor(x)
    ? toFlat(x)
    : Array.from(x as ArrayLike<ArrayLike<number> | number>).flatMap((r) =>
        typeof r === 'number' ? [r] : Array.from(r),
      )
  if (flat.length % width !== 0) throw new DomainError(what, `${what}: expected rows of ${width} values`)
  return {
    flat,
    n: flat.length / width,
    batched: isTensor(x) ? x.shape.length === 2 : typeof (x as ArrayLike<unknown>)[0] !== 'number',
  }
}

/**
 * Cartesian points from barycentric weights on a triangle: p = Σ λᵢ vᵢ. Accepts one weight vector (length 3, giving a
 * length-2 tensor) or n × 3 rows (giving n × 2). Weights are used as given; normalise them first if they are not
 * probabilities.
 */
export function barycentricToCartesian(weights: MatrixLike | VectorLike, vertices?: Tensor): Tensor {
  const v = readVertices(vertices)
  const { flat, n, batched } = rows(weights, 3, 'barycentricToCartesian')
  const out = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const [a, b, c] = [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]]
    out[2 * i] = a * v[0] + b * v[2] + c * v[4]
    out[2 * i + 1] = a * v[1] + b * v[3] + c * v[5]
  }
  return batched ? fromData(out, [n, 2]) : fromData(out)
}

/**
 * Barycentric weights of Cartesian points relative to a triangle (they sum to one; a point outside the triangle has a
 * negative weight). One point (length 2) or n × 2 rows. A degenerate triangle gives NaN weights.
 */
export function cartesianToBarycentric(points: MatrixLike | VectorLike, vertices?: Tensor): Tensor {
  const v = readVertices(vertices)
  const { flat, n, batched } = rows(points, 2, 'cartesianToBarycentric')
  const [x1, y1, x2, y2, x3, y3] = v
  const det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
  const out = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) {
    const [x, y] = [flat[2 * i], flat[2 * i + 1]]
    const a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    const b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    out[3 * i] = a
    out[3 * i + 1] = b
    out[3 * i + 2] = 1 - a - b
  }
  return batched ? fromData(out, [n, 3]) : fromData(out)
}

/** A triangular grid of the 2-simplex. */
export interface SimplexGrid {
  /** Barycentric coordinates of the grid points, m × 3. */
  weights: Tensor
  /** The same points in the plane (on `vertices`), m × 2. */
  points: Tensor
  /** Triangles of the grid as triples of point indices, t × 3 (int32), for drawing a density as a mesh. */
  triangles: Tensor
}

/**
 * The points (i, j, k)/resolution with i + j + k = resolution, and the triangles between them. With `interior`, the
 * weights are shifted off the boundary to (i + δ, j + δ, k + δ)/(resolution + 3δ) with δ = 1/2 so densities that are
 * infinite at the edges (a Dirichlet with α < 1) stay finite at every point.
 */
export function simplexGrid(resolution: number, options: { interior?: boolean; vertices?: Tensor } = {}): SimplexGrid {
  if (!Number.isInteger(resolution) || resolution < 1)
    throw new DomainError('simplexGrid', 'simplexGrid: resolution must be ≥ 1')
  const r = resolution
  const delta = options.interior ? 0.5 : 0
  const index = new Map<string, number>()
  const w: number[] = []
  for (let i = 0; i <= r; i++)
    for (let j = 0; j <= r - i; j++) {
      const k = r - i - j
      index.set(`${i},${j}`, w.length / 3)
      const total = r + 3 * delta
      w.push((i + delta) / total, (j + delta) / total, (k + delta) / total)
    }
  const tris: number[] = []
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r - i; j++) {
      const a = index.get(`${i},${j}`)!
      const b = index.get(`${i + 1},${j}`)!
      const c = index.get(`${i},${j + 1}`)!
      tris.push(a, b, c)
      if (j < r - i - 1) tris.push(b, index.get(`${i + 1},${j + 1}`)!, c)
    }
  const weights = fromData(Float64Array.from(w), [w.length / 3, 3])
  return {
    weights,
    points: barycentricToCartesian(weights, options.vertices),
    triangles: fromData(Int32Array.from(tris), [tris.length / 3, 3]),
  }
}
