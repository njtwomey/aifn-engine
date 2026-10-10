/**
 * Dynamic time warping (DTW) and the lower bounds that prune a search under it.
 *
 * DTW (Sakoe and Chiba 1978, "Dynamic programming algorithm optimization for spoken word recognition", IEEE TASSP
 * 26(1)) is the cheapest monotone, continuous alignment of two series $x$ (length $n$) and $y$ (length $m$). The
 * accumulated cost $D(i, j) = c(x_i, y_j) + \min\{D(i - 1, j), D(i, j - 1), D(i - 1, j - 1)\}$ fills an $n \times m$
 * table in $O(nm)$, on the shared dynamic-programming engine of `aifn-compute/optim/programming`, so it can be stepped.
 * The Sakoe–Chiba band $\lvert i - j \rvert \le w$ restricts the warping, and the cells that can be finite, to a
 * diagonal band. The local cost is $c(a, b) = (a - b)^2$ (the default) or $\lvert a - b \rvert$; with the squared cost
 * the distance is $\sqrt{D(n - 1, m - 1)}$ (indices from 0), as in the UCR suite and tslearn, and with the absolute
 * cost it is $D(n - 1, m - 1)$.
 *
 * Lower bounds prune a nearest-neighbour search under DTW without computing it (Keogh and Ratanamahatana 2005, "Exact
 * indexing of dynamic time warping", KAIS 7(3); Rakthanmanon et al. 2012, "Searching and mining trillions of time
 * series subsequences under dynamic time warping", KDD): LB_Kim from the first, last, largest and smallest values
 * ($O(1)$ once the extremes are known; here they are found in $O(n)$), and LB_Keogh from the query's distance outside
 * the candidate's band envelope (here $O(nw)$, the envelope recomputed on each call).
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DIAGONAL, dp, LEFT, STOP, UP, type DynamicProgram } from 'aifn-compute/optim/programming'

/** The local cost of matching values $a$ and $b$: `'squared'` is $(a - b)^2$, `'absolute'` is $\lvert a - b \rvert$. */
export type DtwCost = 'squared' | 'absolute'

/** Options of {@link dtw}. */
export interface DtwOptions {
  /**
   * The Sakoe–Chiba band half-width $w$: only cells with $\lvert i - j \rvert \le w$ may be on the path. It must be
   * an integer of at least $\lvert n - m \rvert$, or no path reaches the last cell (default: $\max(n, m)$,
   * unconstrained).
   */
  window?: Size
  /** The local cost of matching two values (default `'squared'`). */
  cost?: DtwCost
}

/** A DTW alignment. */
export interface DtwResult {
  /** $\sqrt{D(n - 1, m - 1)}$ for the squared cost, $D(n - 1, m - 1)$ for the absolute cost. */
  readonly distance: number
  /**
   * The accumulated cost table $D$ ($n \times m$) of summed local costs (not square-rooted); Infinity outside the
   * band.
   */
  readonly accumulated: Tensor
  /** The optimal warping path from $(0, 0)$ to $(n - 1, m - 1)$, as `[i, j]` pairs in order. */
  readonly path: readonly (readonly [number, number])[]
}

/**
 * The local cost $c(a, b)$ of matching two values.
 *
 * @param a A value of the first series.
 * @param b A value of the second series.
 * @param cost Which cost: `'squared'` gives $(a - b)^2$, `'absolute'` gives $\lvert a - b \rvert$.
 * @returns The cost of matching `a` with `b`.
 */
const local = (a: number, b: number, cost: DtwCost) => (cost === 'absolute' ? Math.abs(a - b) : (a - b) ** 2)

/**
 * Reads and checks the two series and the options of a DTW: throws `DomainError` for an empty series, or for a window
 * that is not an integer of at least $\lvert n - m \rvert$.
 *
 * @param x The first series, of length $n$.
 * @param y The second series, of length $m$.
 * @param options The band half-width and the local cost; the window defaults to $\max(n, m)$ and the cost to
 *   `'squared'`.
 * @param where The caller's name for error messages.
 * @returns The series as float64 arrays `a` and `b`, their lengths `n` and `m`, the window `w` and the `cost`.
 */
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
 * DTW as a dynamic program on the engine of `aifn-compute/optim/programming` (so its `dynamicProgram` steps it row by
 * row, and `dp` fills it at once): the table is $D$ ($n \times m$), Infinity outside the band, with each cell's choice
 * the predecessor taken (`DIAGONAL`, `UP` or `LEFT`, and `STOP` at $(0, 0)$ and outside the band). Ties prefer the
 * diagonal, then `UP`. Throws `DomainError` as `dtw` does.
 *
 * @param x The first series, of length $n$: the rows of the table.
 * @param y The second series, of length $m$: the columns of the table.
 * @param options The band half-width `window` and the local `cost`, as for `dtw`.
 * @returns The program: its `shape` $[n, m]$ and its `cell` rule, which reads the cells above, to the left and
 *   diagonally before.
 *
 * @example Fill the table cell by cell, as the engine does
 * const x = [0, 1, 2]
 * const y = [0, 2]
 * const prog = dtwProgram(x, y)
 * const D = [[], [], []]
 * for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) D[i][j] = prog.cell(i, j, (a, b) => D[a][b]).value
 * print('shape =', prog.shape)
 * print('filled by hand =', D)
 * print('dtw accumulated =', dtw(x, y).accumulated)
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

/**
 * The DTW distance, accumulated cost table and warping path of two series (see the file notes). The path is traced
 * back from $(n - 1, m - 1)$ by the choices of `dtwProgram`. Throws `DomainError` for an empty series or a window
 * narrower than $\lvert n - m \rvert$.
 *
 * @param x The first series, of length $n$.
 * @param y The second series, of length $m$; the lengths may differ.
 * @param options The band half-width `window` (default unconstrained) and the local `cost` (default `'squared'`).
 * @returns The `distance`, the accumulated cost table and the warping path.
 *
 * @example A signal and its copy stretched to twice the length
 * const x = [0, 1, 2, 1, 0]
 * const y = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
 * const r = dtw(x, y)
 * print('distance =', r.distance)
 * print('path =', r.path)
 *
 * @example A band narrower than the shift forbids the free alignment
 * const x = [0, 0, 1, 0, 0, 0]
 * const y = [0, 0, 0, 0, 1, 0]
 * print('unconstrained =', dtw(x, y).distance)
 * print('window 1, squared cost =', dtw(x, y, { window: 1 }).distance)
 * print('window 1, absolute cost =', dtw(x, y, { window: 1, cost: 'absolute' }).distance)
 */
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

/**
 * The upper and lower envelope of a series under a band of half-width $w$: $U_i = \max_{\lvert k - i \rvert \le w} x_k$
 * and $L_i = \min_{\lvert k - i \rvert \le w} x_k$, the window clipped at the ends. Throws `DomainError` unless the
 * window is a non-negative integer.
 *
 * @param x The series, of length $n$.
 * @param window The band half-width $w$, in samples.
 * @returns `upper` ($U$) and `lower` ($L$), each of length $n$.
 *
 * @example The envelope of a short series under a band of half-width 1
 * const { upper, lower } = keoghEnvelope([0, 3, 1, 0, 2], 1)
 * print('upper =', upper)
 * print('lower =', lower)
 */
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
 * LB_Keogh$(q, c)$: the cost of $q$ outside $c$'s envelope (`keoghEnvelope`), $\sum_i c(q_i, U_i)$ over the $i$ with
 * $q_i > U_i$ plus $\sum_i c(q_i, L_i)$ over those with $q_i < L_i$, square-rooted for the squared cost. A lower bound
 * on {@link dtw} with the same window and cost, for series of equal length; throws `ShapeError` when the lengths
 * differ.
 *
 * @param query The query series $q$.
 * @param candidate The candidate series $c$, whose envelope is taken; the same length as `query`.
 * @param window The band half-width $w$, as passed to `dtw`.
 * @param options `cost`, the local cost (default `'squared'`).
 * @returns The lower bound, in the units of `dtw`'s distance.
 *
 * @example The bound beside the banded distance it bounds
 * const q = [0, 2, 0, 0, 3, 0]
 * const c = [0, 0, 1, 0, 0, 1]
 * print('LB_Keogh =', lbKeogh(q, c, 1))
 * print('DTW, window 1 =', dtw(q, c, { window: 1 }).distance)
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
 * LB_Kim$(q, c)$ (Kim, Park and Chu 2001): the largest of the absolute differences between the two series' first
 * values, last values, maxima and minima. Every warping path matches the first and last pairs and takes each maximum
 * and minimum to some value of the other series, so this is at most {@link dtw}'s distance, for either cost and any
 * window. The series may differ in length; throws `DomainError` when one is empty.
 *
 * @param query The query series $q$.
 * @param candidate The candidate series $c$.
 * @returns The lower bound, in the units of `dtw`'s distance.
 *
 * @example A cheap bound beside the distance it bounds
 * const q = [0, 2, 0, 0, 3, 0]
 * const c = [0, 0, 1, 0, 0, 1]
 * print('LB_Kim =', lbKim(q, c))
 * print('DTW =', dtw(q, c).distance)
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
