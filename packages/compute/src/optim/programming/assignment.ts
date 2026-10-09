/**
 * The assignment problem by the Hungarian algorithm (Kuhn, 1955, "The Hungarian method for the assignment problem",
 * Naval Research Logistics Quarterly 2), in Munkres' star-and-prime form (Munkres, 1957, J. SIAM 5(1)) extended to
 * rectangular matrices (Bourgeois and Lassalle, 1971, CACM 14(12)). Each step is one of Munkres' steps, so a figure
 * can show the reduced matrix, the starred and primed zeros, the covered lines and each augmenting path. Row and
 * column potentials $\uvec$, $\vvec$ are kept so that the reduced matrix is $c_{ij} - u_i - v_j$ throughout; at the
 * end they are an optimal dual solution.
 *
 * A matrix with more rows than columns is transposed first, so the algorithm always works on $k \le l$ rows and
 * columns; `hungarian` maps the result back to the input's rows.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { intTensor, matrix, readMatrix, vector } from './input'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A step of Munkres' algorithm. */
export type HungarianPhase =
  /** Subtract each row's minimum. */
  | 'reduce-rows'
  /** Subtract each column's minimum (square matrices only). */
  | 'reduce-columns'
  /** Star a zero in each row and column where possible, greedily. */
  | 'star'
  /** Cover the columns holding starred zeros; done when all rows are assigned. */
  | 'cover'
  /**
   * Prime an uncovered zero; cover its row and uncover its star's column, or go to augment if its row has no star.
   */
  | 'prime'
  /** Flip stars and primes along the alternating path from the last primed zero: one more assignment. */
  | 'augment'
  /** Add the smallest uncovered value to covered rows and subtract it from uncovered columns: a new zero appears. */
  | 'adjust'
  /** Every row holds a starred zero: the assignment is optimal. */
  | 'done'

/** Options for `hungarianSteps` and `hungarian`. */
export interface HungarianOptions {
  /** Maximise the total instead of minimising it (the algorithm then runs on the negated costs). */
  maximize?: boolean
}

/**
 * One state of the Hungarian algorithm, on the matrix oriented so that it has no more rows than columns. `converged`
 * once every row holds a starred zero (an optimal assignment).
 */
export interface HungarianState extends Status {
  /** The reduced cost matrix $c_{ij} - u_i - v_j$, $k \times l$ with $k \le l$. */
  reduced: Tensor
  /** The (sign-adjusted, oriented) cost matrix the algorithm works on, $k \times l$. */
  cost: Tensor
  /** Column of the starred zero in each row, or $-1$; int32, length $k$. */
  starred: Tensor
  /** Column of the primed zero in each row, or $-1$; int32, length $k$. */
  primed: Tensor
  /** 1 for each covered row, else 0; int32, length $k$. */
  rowCovered: Tensor
  /** 1 for each covered column, else 0; int32, length $l$. */
  columnCovered: Tensor
  /** Row potentials $\uvec$, length $k$, with the reduced matrix $c_{ij} - u_i - v_j$. */
  rowPotential: Tensor
  /** Column potentials $\vvec$, length $l$. */
  columnPotential: Tensor
  /** What the last step did (`start` initially). */
  last: HungarianPhase | 'start'
  /** What the next step will do. */
  next: HungarianPhase
  /**
   * The last augmenting path as (row, column) cells, alternating primed and starred zeros; shape `[p, 2]`; int32.
   * After a `prime` step that finds a row with no star, it holds only that primed zero, from which `augment` starts.
   */
  path: Tensor
  /** The value added and subtracted by the last step when it was `adjust`; 0 after any other step. */
  delta: Scalar
  /** True when the input had more rows than columns and the algorithm works on its transpose. */
  transposed: boolean
  /** True once `next` is `done`. */
  converged: boolean
}

/**
 * Whether a reduced cost counts as zero.
 *
 * @param v The reduced cost.
 * @param tol The absolute tolerance ($10^{-12}$ times the largest absolute cost, or 1 if larger).
 * @returns True when $\lvert v \rvert \le$ `tol`.
 */
const isZero = (v: number, tol: number) => Math.abs(v) <= tol

/**
 * The Hungarian algorithm (Kuhn, 1955; Munkres' steps, 1957) on the cost matrix `cost` ($n \times m$, finite;
 * rectangular matrices assign every row, or every column if fewer), as a traceable algorithm with no start. Each step
 * performs one `HungarianPhase` (priming one zero at a time). It converges when every row of the oriented matrix holds
 * a starred zero; the stars are then an optimal assignment. `hungarian` runs it to the end. Throws `DomainError` for
 * a non-finite cost.
 *
 * @param cost The cost matrix, $n \times m$, as a tensor or a list of rows.
 * @param options Whether to maximise instead of minimise.
 * @returns The algorithm. Its start is ignored; once converged, a step returns the state unchanged.
 *
 * @example Munkres' steps on a 3 by 3 matrix
 * const tr = trace(hungarianSteps([[4, 1, 3], [2, 0, 5], [3, 2, 2]]), {}, 100)
 * print('steps =', tr.steps.map((s) => s.last))
 * const s = tr.steps.at(-1)
 * print('reduced =', s.reduced)
 * print('starred column of each row =', s.starred)
 */
export function hungarianSteps(cost: MatrixLike, options: HungarianOptions = {}): Algorithm<object, HungarianState> {
  const rowsIn = 'shape' in cost ? cost.shape[0] : cost.length
  const colsIn = 'shape' in cost ? cost.shape[1] : rowsIn ? cost[0].length : 0
  const C = readMatrix(cost, 'hungarian: cost', colsIn)
  for (const v of C.a) if (!Number.isFinite(v)) throw new DomainError('hungarian', 'hungarian: costs must be finite')
  const transposed = C.m > C.n
  const k = Math.min(C.m, C.n)
  const l = Math.max(C.m, C.n)
  const a = new Float64Array(k * l)
  for (let i = 0; i < C.m; i++)
    for (let j = 0; j < C.n; j++) {
      const v = (options.maximize ? -1 : 1) * C.a[i * C.n + j]
      if (transposed) a[j * l + i] = v
      else a[i * l + j] = v
    }
  let scale = 1
  for (const v of a) scale = Math.max(scale, Math.abs(v))
  const tol = 1e-12 * scale
  return {
    name: 'hungarian',
    init: () => {
      return {
        reduced: matrix(a, k, l),
        cost: matrix(a, k, l),
        starred: intTensor(new Int32Array(k).fill(-1)),
        primed: intTensor(new Int32Array(k).fill(-1)),
        rowCovered: intTensor(new Int32Array(k)),
        columnCovered: intTensor(new Int32Array(l)),
        rowPotential: vector(new Float64Array(k)),
        columnPotential: vector(new Float64Array(l)),
        last: 'start',
        next: k === 0 ? 'done' : 'reduce-rows',
        path: intTensor([], [0, 2]),
        delta: 0,
        transposed,
        t: 0,
        converged: k === 0,
      }
    },
    step: (s) => {
      if (s.converged) return s
      const [k, l] = s.reduced.shape
      const r = Float64Array.from(s.reduced.data)
      const star = Int32Array.from(s.starred.data)
      const prime = Int32Array.from(s.primed.data)
      const rowCov = Int32Array.from(s.rowCovered.data)
      const colCov = Int32Array.from(s.columnCovered.data)
      const u = Float64Array.from(s.rowPotential.data)
      const v = Float64Array.from(s.columnPotential.data)
      const phase = s.next
      let next: HungarianPhase = phase
      let path = s.path
      let delta = 0
      const starInColumn = (j: number) => star.indexOf(j)
      if (phase === 'reduce-rows') {
        for (let i = 0; i < k; i++) {
          let min = Infinity
          for (let j = 0; j < l; j++) min = Math.min(min, r[i * l + j])
          for (let j = 0; j < l; j++) r[i * l + j] -= min
          u[i] += min
        }
        next = k === l ? 'reduce-columns' : 'star'
      } else if (phase === 'reduce-columns') {
        for (let j = 0; j < l; j++) {
          let min = Infinity
          for (let i = 0; i < k; i++) min = Math.min(min, r[i * l + j])
          for (let i = 0; i < k; i++) r[i * l + j] -= min
          v[j] += min
        }
        next = 'star'
      } else if (phase === 'star') {
        for (let i = 0; i < k; i++)
          for (let j = 0; j < l; j++)
            if (isZero(r[i * l + j], tol) && star[i] < 0 && starInColumn(j) < 0) {
              star[i] = j
              break
            }
        next = 'cover'
      } else if (phase === 'cover') {
        let count = 0
        for (let i = 0; i < k; i++)
          if (star[i] >= 0) {
            colCov[star[i]] = 1
            count++
          }
        next = count === k ? 'done' : 'prime'
      } else if (phase === 'prime') {
        let found: [number, number] | null = null
        for (let i = 0; i < k && !found; i++) {
          if (rowCov[i]) continue
          for (let j = 0; j < l; j++)
            if (!colCov[j] && isZero(r[i * l + j], tol)) {
              found = [i, j]
              break
            }
        }
        if (!found) next = 'adjust'
        else {
          const [i, j] = found
          prime[i] = j
          if (star[i] >= 0) {
            rowCov[i] = 1
            colCov[star[i]] = 0
            next = 'prime'
          } else {
            path = intTensor([i, j], [1, 2])
            next = 'augment'
          }
        }
      } else if (phase === 'augment') {
        // Alternate: primed zero (i, j) → the starred zero in column j → the primed zero in that star's row → …
        const cells: number[] = [s.path.data[0], s.path.data[1]]
        for (;;) {
          const j = cells[cells.length - 1]
          const i = starInColumn(j)
          if (i < 0) break
          cells.push(i, j, i, prime[i])
        }
        for (let q = 0; q < cells.length; q += 4) star[cells[q]] = cells[q + 1]
        prime.fill(-1)
        rowCov.fill(0)
        colCov.fill(0)
        path = intTensor(cells, [cells.length / 2, 2])
        next = 'cover'
      } else if (phase === 'adjust') {
        let min = Infinity
        for (let i = 0; i < k; i++)
          if (!rowCov[i]) for (let j = 0; j < l; j++) if (!colCov[j]) min = Math.min(min, r[i * l + j])
        for (let i = 0; i < k; i++)
          for (let j = 0; j < l; j++) {
            if (rowCov[i]) r[i * l + j] += min
            if (!colCov[j]) r[i * l + j] -= min
          }
        for (let i = 0; i < k; i++) if (rowCov[i]) u[i] -= min
        for (let j = 0; j < l; j++) if (!colCov[j]) v[j] += min
        delta = min
        next = 'prime'
      }
      return {
        ...s,
        reduced: matrix(r, k, l),
        starred: intTensor(star),
        primed: intTensor(prime),
        rowCovered: intTensor(rowCov),
        columnCovered: intTensor(colCov),
        rowPotential: vector(u),
        columnPotential: vector(v),
        last: phase,
        next,
        path,
        delta,
        t: s.t + 1,
        converged: next === 'done',
      }
    },
  }
}

/** The result of `hungarian`. */
export interface AssignmentResult {
  /**
   * The column assigned to each row of the input, or $-1$ for an unassigned row (more rows than columns); int32.
   */
  assignment: Tensor
  /** Total cost of the assignment (in the input's sign, also when maximising). */
  cost: Scalar
  /**
   * Dual potentials $\uvec$ of the input's rows, for the minimised matrix $\Cmat$ (the costs, negated when
   * maximising), with $c_{ij} \ge u_i + v_j$.
   */
  rowPotential: Tensor
  /** Dual potentials $\vvec$ of the input's columns; see `rowPotential`. */
  columnPotential: Tensor
  /** Munkres steps taken. */
  steps: Size
}

/**
 * Solve the linear assignment problem: assign each row to a distinct column (or each column to a row, when there are
 * fewer columns) minimising (or, with `maximize`, maximising) the total cost, by the Hungarian algorithm in $O(n^3)$,
 * as `scipy.optimize.linear_sum_assignment`. Throws `DomainError` for a non-finite cost.
 *
 * @param cost The cost matrix, $n \times m$, as a tensor or a list of rows.
 * @param options Whether to maximise instead of minimise.
 * @returns The assignment, its total cost, the dual potentials and the steps taken.
 *
 * @example Assign three workers to three jobs
 * const r = hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
 * print('job of each worker =', r.assignment, ' cost =', r.cost)
 * // At the optimum the potentials add up to the cost (strong duality).
 * print('u =', r.rowPotential, ' v =', r.columnPotential)
 *
 * @example Maximise, and a rectangular matrix
 * print('maximised:', hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]], { maximize: true }).assignment)
 * const r = hungarian([[1, 2], [2, 4], [3, 1]])
 * print('three rows, two columns:', r.assignment, ' cost =', r.cost)
 */
export function hungarian(cost: MatrixLike, options: HungarianOptions = {}): AssignmentResult {
  const s = run(hungarianSteps(cost, options), {}, 1_000_000)
  const [k] = s.reduced.shape
  const rows = s.transposed ? s.reduced.shape[1] : k
  const assignment = new Int32Array(rows).fill(-1)
  let total = 0
  for (let i = 0; i < k; i++) {
    const j = s.starred.data[i]
    if (j < 0) continue
    if (s.transposed) assignment[j] = i
    else assignment[i] = j
    total += s.cost.data[i * s.reduced.shape[1] + j]
  }
  const sign = options.maximize ? -1 : 1
  return {
    assignment: intTensor(assignment),
    cost: sign * total,
    rowPotential: s.transposed ? s.columnPotential : s.rowPotential,
    columnPotential: s.transposed ? s.rowPotential : s.columnPotential,
    steps: s.t,
  }
}
