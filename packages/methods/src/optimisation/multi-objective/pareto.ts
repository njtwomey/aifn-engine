/**
 * Pareto dominance and the non-dominated front of a set of objective vectors.
 *
 * With $M$ objectives, all minimised, a point $\yvec$ dominates $\yvec'$, written $\yvec \prec \yvec'$, when it is no
 * worse in every objective and better in at least one: $y_j \le y'_j$ for all $j$ and $y_j < y'_j$ for some $j$. The
 * Pareto front of a set is the subset that no member dominates (Zitzler and Thiele, 1999). Every function of the
 * module takes a `minimise` option, a flag for all objectives or one per objective; a maximised objective is negated
 * on the way in, so the code works in minimisation throughout.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'

const { mat, toMatrixF64 } = dense

/** The direction of each objective: true to minimise all (the default), false to maximise all, or one flag each. */
export type Directions = boolean | readonly boolean[]

/** Options of `paretoFront`. */
export interface ParetoOptions {
  /** Minimise every objective (default true), maximise every one (false), or one flag per objective. */
  minimise?: Directions
}

/**
 * The sign that turns each objective into one to minimise: $+1$ for a minimised objective and $-1$ for a maximised
 * one. Throws `ShapeError` when a list of flags has the wrong length.
 *
 * @param minimise The directions, as `Directions`.
 * @param M The number of objectives.
 * @param where The caller's name, for error messages.
 * @returns The $M$ signs.
 */
export function signs(minimise: Directions, M: number, where: string): Float64Array {
  if (typeof minimise === 'boolean') return new Float64Array(M).fill(minimise ? 1 : -1)
  if (minimise.length !== M) throw new ShapeError(where, `${where}: ${minimise.length} directions for ${M} objectives`)
  return Float64Array.from(minimise, (m) => (m ? 1 : -1))
}

/**
 * The rows of an $n \times M$ matrix of objective values as arrays, each multiplied by its objective's sign so that
 * every objective is minimised. Throws `DomainError` for a NaN value and `ShapeError` for a matrix with no columns or
 * the wrong number of them. An empty set (no rows) gives no rows, whatever its columns.
 *
 * @param points The values, one row per point: a rank-2 tensor or an array of rows.
 * @param sign The sign of each objective, from `signs`.
 * @param where The caller's name, for error messages.
 * @returns The rows, in minimisation form.
 */
export function rowsOf(points: MatrixLike, sign: Float64Array | null, where: string): Float64Array[] {
  const { data, m, n } = toMatrixF64(points, where)
  if (m === 0) return []
  if (sign && n !== sign.length) throw new ShapeError(where, `${where}: ${n} objectives, expected ${sign.length}`)
  if (n === 0) throw new ShapeError(where, `${where}: the points have no objectives`)
  const rows: Float64Array[] = []
  for (let i = 0; i < m; i++) {
    const row = data.slice(i * n, (i + 1) * n)
    for (let j = 0; j < n; j++) {
      if (Number.isNaN(row[j])) throw new DomainError(where, `${where}: a value is NaN`)
      if (sign) row[j] *= sign[j]
    }
    rows.push(row)
  }
  return rows
}

/**
 * Whether $\avec \prec \bvec$ in minimisation: $\avec$ is no worse in every objective and better in at least one.
 *
 * @param a The first point.
 * @param b The second point, of the same length.
 * @returns True when `a` dominates `b`; false for equal points.
 */
export function dominates(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  let better = false
  for (let j = 0; j < a.length; j++) {
    if (a[j] > b[j]) return false
    if (a[j] < b[j]) better = true
  }
  return better
}

/**
 * The indices of the rows that no other row dominates, in minimisation form, by comparing every pair ($O(n^2 M)$).
 * Equal rows do not dominate each other, so each copy is kept.
 *
 * @param rows The points, every objective minimised.
 * @returns The indices of the non-dominated rows, in increasing order.
 */
export function nonDominated(rows: readonly ArrayLike<number>[]): number[] {
  const keep: number[] = []
  for (let i = 0; i < rows.length; i++) {
    let dominated = false
    for (let k = 0; k < rows.length && !dominated; k++) if (k !== i && dominates(rows[k], rows[i])) dominated = true
    if (!dominated) keep.push(i)
  }
  return keep
}

/**
 * Lexicographic order of two points: by the first objective, then the second, and so on.
 *
 * @param a The first point.
 * @param b The second point.
 * @returns Negative when `a` comes first, positive when `b` does, 0 when they are equal.
 */
export function lexicographic(a: ArrayLike<number>, b: ArrayLike<number>): number {
  for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) return a[j] - b[j]
  return 0
}

/**
 * The Pareto front of a set of points: the points that no other point dominates (Zitzler and Thiele, 1999), by
 * comparing every pair. Points that are equal do not dominate each other, so each copy of a non-dominated point is
 * kept. The front is ordered lexicographically from the best value of the first objective (with two objectives, along
 * the staircase from the best first to the best second objective). Throws `DomainError` for a NaN value and
 * `ShapeError` for directions of the wrong length.
 *
 * @param points The objective values, $n \times M$: one row per point, one column per objective.
 * @param options The direction of each objective, as `ParetoOptions`.
 * @returns `indices`, the rows of `points` on the front in the order described, and `front`, those rows
 *   ($k \times M$, in the units given).
 *
 * @example Two objectives to minimise: the front and the dominated points
 * const points = tensor([[1, 5], [2, 3], [3, 4], [4, 1], [2.5, 2.5], [5, 5]])
 * const { indices, front } = paretoFront(points)
 * print('on the front:', indices)
 * print('front:', front)
 *
 * @example Accuracy is maximised and latency minimised
 * const models = [[0.91, 120], [0.88, 40], [0.93, 300], [0.87, 60], [0.9, 150]]
 * const { indices } = paretoFront(models, { minimise: [false, true] })
 * print('worth considering:', indices.map((i) => models[i]))
 */
export function paretoFront(points: MatrixLike, options: ParetoOptions = {}): { indices: number[]; front: Tensor } {
  const raw = rowsOf(points, null, 'paretoFront')
  const M = raw.length > 0 ? raw[0].length : typeof options.minimise === 'object' ? options.minimise.length : 0
  const sign = signs(options.minimise ?? true, M, 'paretoFront')
  const rows = raw.map((r) => r.map((v, j) => v * sign[j]))
  const indices = nonDominated(rows).sort((a, b) => lexicographic(rows[a], rows[b]) || a - b)
  const out = new Float64Array(indices.length * M)
  indices.forEach((i, k) => out.set(raw[i], k * M))
  return { indices, front: mat(out, indices.length, M) }
}
