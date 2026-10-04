/**
 * Dynamic time warping (Sakoe and Chiba 1978, "Dynamic programming algorithm optimization for spoken word
 * recognition", IEEE TASSP 26(1)): the cheapest monotone, continuous alignment of two series. The accumulated cost
 * D(i, j) = c(xᵢ, yⱼ) + min{D(i − 1, j), D(i, j − 1), D(i − 1, j − 1)} fills an n × m table in O(nm) (on the shared
 * dynamic-programming engine, so it can be stepped); the
 * Sakoe–Chiba band |i − j| ≤ w restricts the warping and the work to O(nw). With the squared cost the distance is
 * √D(n, m), as in the UCR suite and tslearn; with the absolute cost it is D(n, m).
 *
 * Lower bounds prune a nearest-neighbour search under DTW without computing it (Keogh and Ratanamahatana 2005, "Exact
 * indexing of dynamic time warping", KAIS 7(3); Rakthanmanon et al. 2012, "Searching and mining trillions of time
 * series subsequences under dynamic time warping", KDD): LB_Kim from the first, last, largest and smallest values in
 * O(1), and LB_Keogh from the query's distance outside the candidate's band envelope in O(n).
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DIAGONAL, dp, LEFT, STOP, UP, type DynamicProgram } from 'aifn-compute/optim/programming'

/** The local cost of matching two values. */
export type DtwCost = 'squared' | 'absolute'

/** Options of {@link dtw}. */
export interface DtwOptions {
  /** The Sakoe–Chiba band half-width w: only |i − j| ≤ w is allowed (default: unconstrained). */
  window?: Size
  /** Default `squared`. */
  cost?: DtwCost
}

/** A DTW alignment. */
export interface DtwResult {
  /** √D(n, m) for the squared cost, D(n, m) for the absolute cost. */
  readonly distance: number
  /** The accumulated cost table D (n × m); Infinity outside the band. */
  readonly accumulated: Tensor
  /** The optimal warping path from (0, 0) to (n − 1, m − 1), as [i, j] pairs. */
  readonly path: readonly (readonly [number, number])[]
}

const local = (a: number, b: number, cost: DtwCost) => (cost === 'absolute' ? Math.abs(a - b) : (a - b) ** 2)

function inputs(x: VectorLike, y: VectorLike, options: DtwOptions, where: string) {
  const a = dense.toF64(x, where)
  const b = dense.toF64(y, where)
  const n = a.length
  const m = b.length
  if (n === 0 || m === 0) throw new DomainError(where, `${where}: both series must be non-empty`)
  const w = options.window ?? Math.max(n, m)
  if (!(Number.isInteger(w) && w >= Math.abs(n - m)))
    throw new DomainError(where, `${where}: the window ${w} must be an integer ≥ |n − m| = ${Math.abs(n - m)}`)
  return { a, b, n, m, w, cost: options.cost ?? 'squared' }
}

/**
 * DTW as a dynamic program on `aifn-compute/optim/programming`'s engine (so `dynamicProgram` steps it row by row): the table
 * is D (n × m), Infinity outside the band, with each cell's choice the predecessor taken (DIAGONAL, UP or LEFT; ties
 * prefer the diagonal).
 */
export function dtwProgram(x: VectorLike, y: VectorLike, options: DtwOptions = {}): DynamicProgram {
  const { a, b, n, m, w, cost } = inputs(x, y, options, 'dtwProgram')
  return {
    shape: [n, m],
    cell: (i, j, get) => {
      if (Math.abs(i - j) > w) return { value: Infinity, choice: STOP }
      const c = local(a[i], b[j], cost)
      if (i === 0 && j === 0) return { value: c, choice: STOP }
      const diag = i > 0 && j > 0 ? get(i - 1, j - 1) : Infinity
      const up = i > 0 ? get(i - 1, j) : Infinity
      const left = j > 0 ? get(i, j - 1) : Infinity
      if (diag <= up && diag <= left) return { value: c + diag, choice: DIAGONAL }
      return up <= left ? { value: c + up, choice: UP } : { value: c + left, choice: LEFT }
    },
  }
}

/** The DTW distance, accumulated cost table and warping path of two series (module notes). */
export function dtw(x: VectorLike, y: VectorLike, options: DtwOptions = {}): DtwResult {
  const { n, m, cost } = inputs(x, y, options, 'dtw')
  const { table, choice } = dp(dtwProgram(x, y, options))
  const C = dense.data(choice)
  const path: [number, number][] = [[n - 1, m - 1]]
  let i = n - 1
  let j = m - 1
  while (i > 0 || j > 0) {
    const move = C[i * m + j]
    if (move === DIAGONAL) {
      i--
      j--
    } else if (move === UP) i--
    else j--
    path.push([i, j])
  }
  path.reverse()
  const total = dense.data(table)[n * m - 1]
  return { distance: cost === 'squared' ? Math.sqrt(total) : total, accumulated: table, path }
}

/** The upper and lower envelope of a series under a band of half-width w: Uᵢ = max x[i−w … i+w], Lᵢ the min. */
export function keoghEnvelope(x: VectorLike, window: Size): { upper: Tensor; lower: Tensor } {
  const v = dense.toF64(x, 'keoghEnvelope')
  if (!(Number.isInteger(window) && window >= 0))
    throw new DomainError('keoghEnvelope', 'keoghEnvelope: the window must be a non-negative integer')
  const upper = new Float64Array(v.length)
  const lower = new Float64Array(v.length)
  for (let i = 0; i < v.length; i++) {
    let hi = -Infinity
    let lo = Infinity
    for (let j = Math.max(0, i - window); j <= Math.min(v.length - 1, i + window); j++) {
      hi = Math.max(hi, v[j])
      lo = Math.min(lo, v[j])
    }
    upper[i] = hi
    lower[i] = lo
  }
  return { upper: fromData(upper), lower: fromData(lower) }
}

/**
 * LB_Keogh(q, c): the cost of q outside c's envelope, Σ (qᵢ − Uᵢ)² where qᵢ > Uᵢ and (qᵢ − Lᵢ)² where qᵢ < Lᵢ (square
 * rooted for the squared cost). A lower bound on {@link dtw} with the same window, for series of equal length.
 */
export function lbKeogh(
  query: VectorLike,
  candidate: VectorLike,
  window: Size,
  options: { cost?: DtwCost } = {},
): number {
  const q = dense.toF64(query, 'lbKeogh')
  const c = dense.toF64(candidate, 'lbKeogh')
  if (q.length !== c.length) throw new ShapeError('lbKeogh', 'lbKeogh: the series must have equal lengths')
  const { upper, lower } = keoghEnvelope(c, window)
  const U = dense.data(upper)
  const L = dense.data(lower)
  const cost = options.cost ?? 'squared'
  let s = 0
  for (let i = 0; i < q.length; i++) {
    if (q[i] > U[i]) s += local(q[i], U[i], cost)
    else if (q[i] < L[i]) s += local(q[i], L[i], cost)
  }
  return cost === 'squared' ? Math.sqrt(s) : s
}

/**
 * LB_Kim(q, c) (Kim, Park and Chu 2001): the largest of the differences between the two series' first values, last
 * values, maxima and minima. Every warping path matches the first and last pairs and takes each maximum and minimum
 * to some value of the other series, so this is at most {@link dtw}'s distance, for either cost.
 */
export function lbKim(query: VectorLike, candidate: VectorLike): number {
  const q = dense.toF64(query, 'lbKim')
  const c = dense.toF64(candidate, 'lbKim')
  if (q.length === 0 || c.length === 0) throw new DomainError('lbKim', 'lbKim: both series must be non-empty')
  const max = (v: Float64Array) => v.reduce((a, b) => Math.max(a, b), -Infinity)
  const min = (v: Float64Array) => v.reduce((a, b) => Math.min(a, b), Infinity)
  return Math.max(
    Math.abs(q[0] - c[0]),
    Math.abs(q[q.length - 1] - c[c.length - 1]),
    Math.abs(max(q) - max(c)),
    Math.abs(min(q) - min(c)),
  )
}
