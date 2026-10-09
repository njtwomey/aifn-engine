/**
 * The two-phase primal simplex method on a dense tableau (Dantzig, 1951; the two-phase method as in Chvátal, 1983,
 * "Linear Programming", ch. 8; Bland's rule from Bland, 1977, "New finite pivoting rules for the simplex method",
 * Mathematics of Operations Research 2(2)). Every state exposes the tableau, the basis, the next entering column and
 * leaving row, and the ratio test, so a figure can walk through the pivots.
 *
 * The method works on the standard form of `./lp` (minimise $\cvec^\top\zvec$ subject to $\Amat\zvec = \bvec$,
 * $\zvec \ge \zeros$, $\bvec \ge \zeros$). Rows with a slack of coefficient $+1$ start with it basic; the others get an
 * artificial column, and phase 1 minimises the sum of the artificials to find a feasible basis. The tableau is a
 * row-major array whose rows are the constraints and, last, the reduced costs; its last column is the right-hand side.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { Index, Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import type { RunOptions } from '../options'
import { intTensor, matrix, solve, vector } from './input'
import {
  dualityReport,
  parseLP,
  recoverDuals,
  standardForm,
  toOriginal,
  type DualityReport,
  type LinearProgram,
  type LinearProgramDuals,
  type StandardForm,
} from './lp'

/**
 * How the entering column is chosen. `bland`: the lowest-index column with a negative reduced cost, and the leaving
 * row by lowest basic-variable index among ties; it never cycles. `dantzig`: the most negative reduced cost, and the
 * first row among ties; faster in practice, but it can cycle on degenerate problems (Beale, 1955).
 */
export type SimplexRule = 'bland' | 'dantzig'

/**
 * What a simplex step did: `start` (the initial tableau), `pivot` (one basis change) or `phase-2` (left phase 1 and
 * built the phase-2 tableau).
 */
export type SimplexEvent = 'start' | 'pivot' | 'phase-2'

/** Where a simplex run stands: still running, or finished with one of the four outcomes. */
export type SimplexStatus = 'running' | 'optimal' | 'infeasible' | 'unbounded' | 'cycling'

/** Options for `simplex`. */
export interface SimplexOptions {
  /** Pivot rule (default `bland`). */
  rule?: SimplexRule
  /** Tolerance for reduced costs, pivots and feasibility (default 1e-9). */
  tolerance?: Scalar
}

/**
 * One state of the simplex method: the tableau after some number of pivots. `converged` when `status` is `optimal`;
 * `terminated` when it is `infeasible`, `unbounded` or `cycling`.
 */
export interface SimplexState extends Status {
  /**
   * 1 while finding a feasible basis (minimising the sum of artificial variables), 2 while optimising
   * $\cvec^\top\xvec$.
   */
  phase: 1 | 2
  /**
   * The tableau, $(m + 1) \times (k + 1)$: rows 0 to $m - 1$ are the constraints
   * $[\Bmat^{-1}\Amat \mid \Bmat^{-1}\bvec]$, the last row holds the reduced costs and, in its last entry, minus the
   * current phase's objective (in standard form).
   */
  tableau: Tensor
  /** The basic column of each constraint row, int32, length m. */
  basis: Tensor
  /**
   * The label of each of the $k$ columns: `x1`, `x1+`/`x1-` (a free variable's parts), `s1` (slacks of the $\le$
   * rows), `t1` (slacks of upper bounds), `a1` (artificials, numbered by standard-form row).
   */
  labels: readonly string[]
  /** The standard-form row of each tableau row (rows found redundant in phase 1 are removed), int32. */
  rowsKept: Tensor
  /** Number of artificial columns (the last columns before the right-hand side); 0 in phase 2. */
  artificial: Size
  /**
   * The column that enters at the next pivot, or $-1$ when there is none. When `unbounded`, the column along which the
   * objective decreases without bound.
   */
  entering: Index
  /** The tableau row that leaves at the next pivot, or $-1$. */
  leaving: Index
  /**
   * The ratio test for the entering column, length $m$: $\max(0, r_i) / a_{ie}$ ($r_i$ the right-hand side, $a_{ie}$
   * the entering column's entry) where $a_{ie}$ exceeds the tolerance, else `Infinity`.
   */
  ratios: Tensor
  /**
   * The pivot made by the last step, or null: its tableau `row`, the entering `column`, and the `leavingColumn` that
   * left the basis.
   */
  lastPivot: { row: Index; column: Index; leavingColumn: Index } | null
  /** What the step that made this state did. */
  event: SimplexEvent
  /** Whether the run goes on (`running`) or how it ended. */
  status: SimplexStatus
  /** Pivots made so far (steps count the move to phase 2 as well). */
  pivots: Size
  /** Phase 1: the sum of the artificial variables. Phase 2: $\cvec^\top\xvec$ in the original variables. */
  objective: Scalar
  /** The current basic solution in the original variables, length $n$. */
  x: Tensor
  /** The current basic solution in the tableau's columns, length $k$. */
  z: Tensor
  /**
   * True when the last pivot was degenerate: the entering variable entered at zero and $\xvec$ did not move.
   */
  degenerate: boolean
  /** Bases visited in the current phase (sorted column lists joined by commas), to detect cycling. */
  visited: readonly string[]
  /**
   * When unbounded: a direction $\dvec$ in $\xvec$-space along which the objective decreases without bound; null
   * otherwise.
   */
  ray: Tensor | null
  /** The standard form the tableau is built on. */
  standard: StandardForm
  /** True when `status` is `optimal`. */
  converged: boolean
  /** True when `status` is `infeasible`, `unbounded` or `cycling`. */
  terminated: boolean
}

/** The pivot rule and tolerance a run closes over (`tolerance` as in `SimplexOptions`). */
type Config = { rule: SimplexRule; tolerance: Scalar }

// ---------------------------------------------------------------------------------------------------------------------
// Tableau arithmetic on raw arrays; `w` is the row width (columns + 1).

/**
 * Gauss–Jordan pivot on the entry in row $r$ and column $c$, in place: row $r$ is scaled to a unit pivot and column
 * $c$ is cleared in every other row.
 *
 * @param t The tableau, row-major, `rows` rows of `w` values; modified in place.
 * @param rows The number of rows of `t` to update, including the reduced-cost row.
 * @param w The row width (the number of columns plus the right-hand side).
 * @param r The pivot row.
 * @param c The pivot column; its entry in row `r` must be non-zero.
 */
export function pivotTableau(t: Float64Array, rows: number, w: number, r: number, c: number): void {
  const p = t[r * w + c]
  for (let k = 0; k < w; k++) t[r * w + k] /= p
  t[r * w + c] = 1
  for (let i = 0; i < rows; i++) {
    if (i === r) continue
    const f = t[i * w + c]
    if (f === 0) continue
    for (let k = 0; k < w; k++) t[i * w + k] -= f * t[r * w + k]
    t[i * w + c] = 0
  }
}

/**
 * The basic solution of a tableau: basic columns take their row's right-hand side, the others 0.
 *
 * @param t The tableau, row-major with row width `w` (not modified).
 * @param m The number of constraint rows (the reduced-cost row after them is not read).
 * @param w The row width; the solution has $w - 1$ entries and the right-hand side is the last column.
 * @param basis The basic column of each of the $m$ constraint rows.
 * @returns The value of every column, $w - 1$ entries.
 */
export function basicSolution(t: Float64Array, m: number, w: number, basis: ArrayLike<number>): Float64Array {
  const z = new Float64Array(w - 1)
  for (let r = 0; r < m; r++) z[basis[r]] = t[r * w + w - 1]
  return z
}

/**
 * A key for a basis as a set.
 *
 * @param basis The basic columns, in any order.
 * @returns The columns sorted and joined by commas, equal for two bases with the same columns.
 */
const basisKey = (basis: ArrayLike<number>) =>
  Array.from(basis)
    .sort((a, b) => a - b)
    .join(',')

/** The next pivot `choose` picked, or why there is none. */
type Choice = {
  /** The entering column, or $-1$ when no reduced cost is negative. */
  entering: number
  /** The leaving tableau row, or $-1$. */
  leaving: number
  /** The ratio test of the entering column (all `Infinity` when none enters). */
  ratios: Float64Array
  /**
   * `running` while there is a pivot to make, and at the end of a feasible phase 1; `optimal`, `infeasible` or
   * `unbounded` otherwise.
   */
  status: SimplexStatus
  /** When `unbounded`: the ray in the tableau's columns, length $k$. */
  ray: Float64Array | null
}

/**
 * Choose the next pivot of a tableau, or decide the phase is over. The entering column has a reduced cost below
 * $-$`tol` (the first such with `bland`, the most negative with `dantzig`); the leaving row has the smallest ratio,
 * ties broken by the lowest basic column with `bland` and by the first row with `dantzig`. When no column enters,
 * phase 2 is `optimal`, and phase 1 is `infeasible` if the artificials still sum to more than $10 \cdot$ `tol` times
 * the largest right-hand side (or 1). When no row limits the entering column, the phase is `unbounded`.
 *
 * @param t The tableau, row-major with row width `w` (not modified).
 * @param m The number of constraint rows; the reduced costs are row $m$.
 * @param w The row width (columns plus the right-hand side).
 * @param basis The basic column of each constraint row.
 * @param phase The current phase.
 * @param artificialFrom The first artificial column: it and the columns after it never enter.
 * @param rule The pivot rule.
 * @param tol The tolerance on reduced costs and pivot entries.
 * @returns The choice.
 */
function choose(
  t: Float64Array,
  m: number,
  w: number,
  basis: Int32Array,
  phase: 1 | 2,
  artificialFrom: number,
  rule: SimplexRule,
  tol: number,
): Choice {
  const k = w - 1
  const obj = m * w
  let entering = -1
  // Artificial columns never re-enter: once one leaves the basis it is no longer needed.
  for (let j = 0; j < artificialFrom; j++) {
    const d = t[obj + j]
    if (d >= -tol) continue
    if (rule === 'bland') {
      entering = j
      break
    }
    if (entering < 0 || d < t[obj + entering]) entering = j
  }
  const ratios = new Float64Array(m).fill(Infinity)
  if (entering < 0) {
    if (phase === 2) return { entering, leaving: -1, ratios, status: 'optimal', ray: null }
    // Phase 1 is over: infeasible unless the artificial variables sum to zero.
    const w1 = -t[obj + k]
    let scale = 1
    for (let r = 0; r < m; r++) scale = Math.max(scale, Math.abs(t[r * w + k]))
    return { entering, leaving: -1, ratios, status: w1 > tol * scale * 10 ? 'infeasible' : 'running', ray: null }
  }
  let leaving = -1
  let best = Infinity
  for (let r = 0; r < m; r++) {
    const a = t[r * w + entering]
    if (a <= tol) continue
    ratios[r] = Math.max(0, t[r * w + k]) / a
  }
  for (let r = 0; r < m; r++) {
    const q = ratios[r]
    if (q === Infinity) continue
    if (leaving < 0 || q < best - 1e-12 * (1 + best)) {
      leaving = r
      best = q
    } else if (rule === 'bland' && Math.abs(q - best) <= 1e-12 * (1 + best) && basis[r] < basis[leaving]) {
      leaving = r
      best = Math.min(best, q)
    }
  }
  if (leaving < 0) {
    // No row limits the entering column: the objective decreases without bound along the ray.
    const ray = new Float64Array(k)
    ray[entering] = 1
    for (let r = 0; r < m; r++) ray[basis[r]] = -t[r * w + entering]
    return { entering, leaving, ratios, status: 'unbounded', ray }
  }
  return { entering, leaving, ratios, status: 'running', ray: null }
}

/**
 * Assemble a state from raw tableau data: choose the next pivot, read off the basic solution and objective, and mark
 * the run `cycling` when a pivot returns to a basis already visited in this phase.
 *
 * @param cfg The pivot rule and tolerance.
 * @param sf The standard form the tableau is built on.
 * @param t The tableau, row-major, $(m + 1) \times w$; copied into the state.
 * @param m The number of constraint rows.
 * @param w The row width (columns plus the right-hand side).
 * @param basis The basic column of each constraint row.
 * @param rowsKept The standard-form row of each tableau row.
 * @param labels The label of each column.
 * @param artificial The number of artificial columns, the last before the right-hand side.
 * @param phase The phase the tableau is in.
 * @param extra The fields the caller sets: the step count `t`, `lastPivot`, `event`, `pivots`, `degenerate`, and the
 *   bases `visited` before this state.
 * @returns The state.
 */
function makeState(
  cfg: Config,
  sf: StandardForm,
  t: Float64Array,
  m: number,
  w: number,
  basis: Int32Array,
  rowsKept: Int32Array,
  labels: readonly string[],
  artificial: number,
  phase: 1 | 2,
  extra: Pick<SimplexState, 't' | 'lastPivot' | 'event' | 'pivots' | 'degenerate' | 'visited'>,
): SimplexState {
  const choice = choose(t, m, w, basis, phase, w - 1 - artificial, cfg.rule, cfg.tolerance)
  const z = basicSolution(t, m, w, basis)
  const x = toOriginal(sf, z)
  let objective = 0
  if (phase === 1) for (let j = w - 1 - artificial; j < w - 1; j++) objective += z[j]
  else for (let j = 0; j < sf.lp.n; j++) objective += sf.lp.c[j] * x[j]
  let status = choice.status
  let visited = extra.visited
  if (status === 'running' && extra.event === 'pivot') {
    const key = basisKey(basis)
    if (visited.includes(key)) status = 'cycling'
    visited = [...visited, key]
  }
  return {
    ...extra,
    standard: sf,
    visited,
    phase,
    tableau: matrix(t, m + 1, w),
    basis: intTensor(basis),
    labels,
    rowsKept: intTensor(rowsKept),
    artificial,
    entering: status === 'running' ? choice.entering : status === 'unbounded' ? choice.entering : -1,
    leaving: status === 'running' ? choice.leaving : -1,
    ratios: vector(choice.ratios),
    status,
    converged: status === 'optimal',
    terminated: status === 'infeasible' || status === 'unbounded' || status === 'cycling',
    objective,
    x: vector(x),
    z: vector(z),
    ray: choice.ray ? vector(toOriginal(sf, choice.ray.subarray(0, sf.N), false)) : null,
  }
}

/**
 * The initial tableau: slack columns basic where a row has one, artificial columns elsewhere. With artificials the run
 * starts in phase 1, its reduced costs priced out over the artificial rows; without, it starts in phase 2 with the
 * reduced costs $\cvec$.
 *
 * @param cfg The pivot rule and tolerance.
 * @param sf The standard form of the problem.
 * @returns The `start` state.
 */
function initialState(cfg: Config, sf: StandardForm): SimplexState {
  const { m, N } = sf
  const artificialRows: number[] = []
  for (let r = 0; r < m; r++) if (sf.slackOfRow[r] < 0) artificialRows.push(r)
  const k = N + artificialRows.length
  const w = k + 1
  const t = new Float64Array((m + 1) * w)
  const basis = new Int32Array(m)
  for (let r = 0; r < m; r++) {
    for (let j = 0; j < N; j++) t[r * w + j] = sf.A[r * N + j]
    t[r * w + k] = sf.b[r]
    basis[r] = sf.slackOfRow[r]
  }
  artificialRows.forEach((r, a) => {
    t[r * w + N + a] = 1
    basis[r] = N + a
  })
  const labels = [...sf.labels, ...artificialRows.map((r) => `a${r + 1}`)]
  const obj = m * w
  if (artificialRows.length > 0) {
    // Phase 1 minimises the sum of the artificials; pricing out the basis gives d_j = −Σ over artificial rows.
    for (const r of artificialRows) {
      for (let j = 0; j < N; j++) t[obj + j] -= t[r * w + j]
      t[obj + k] -= t[r * w + k]
    }
  } else {
    // All-slack basis: the slacks cost nothing, so the reduced costs are c.
    for (let j = 0; j < N; j++) t[obj + j] = sf.c[j]
  }
  const rowsKept = Int32Array.from({ length: m }, (_, r) => r)
  return makeState(cfg, sf, t, m, w, basis, rowsKept, labels, artificialRows.length, artificialRows.length ? 1 : 2, {
    t: 0,
    lastPivot: null,
    event: 'start',
    pivots: 0,
    degenerate: false,
    visited: [basisKey(basis)],
  })
}

/**
 * Leave phase 1: pivot out artificials left basic at zero (on any original column whose entry exceeds $10^3$ times
 * the tolerance), drop the rows that have no such column as redundant, drop the artificial columns, and price out the
 * phase-2 reduced costs $\cvec - \cvec_B^\top\Bmat^{-1}\Amat$.
 *
 * @param cfg The pivot rule and tolerance.
 * @param s The last phase-1 state, feasible (its artificials sum to zero).
 * @returns The `phase-2` state.
 */
function toPhase2(cfg: Config, s: SimplexState): SimplexState {
  const sf = s.standard
  const m0 = s.basis.shape[0]
  const w0 = s.tableau.shape[1]
  const t0 = Float64Array.from(s.tableau.data)
  const basis0 = Int32Array.from(s.basis.data)
  const N = sf.N
  const drop = new Set<number>()
  for (let r = 0; r < m0; r++) {
    if (basis0[r] < N) continue
    let col = -1
    for (let j = 0; j < N; j++) if (Math.abs(t0[r * w0 + j]) > cfg.tolerance * 1e3) col = col < 0 ? j : col
    if (col < 0)
      drop.add(r) // every original coefficient is zero: the row is a combination of the others
    else {
      pivotTableau(t0, m0 + 1, w0, r, col)
      basis0[r] = col
    }
  }
  const m = m0 - drop.size
  const w = N + 1
  const t = new Float64Array((m + 1) * w)
  const basis = new Int32Array(m)
  const rowsKept = new Int32Array(m)
  let i = 0
  for (let r = 0; r < m0; r++) {
    if (drop.has(r)) continue
    for (let j = 0; j < N; j++) t[i * w + j] = t0[r * w0 + j]
    t[i * w + N] = t0[r * w0 + w0 - 1]
    basis[i] = basis0[r]
    rowsKept[i] = s.rowsKept.data[r]
    i++
  }
  // Phase 2 reduced costs: d = c − c_Bᵀ(B⁻¹A), objective entry −c_Bᵀ(B⁻¹b).
  const obj = m * w
  for (let j = 0; j < N; j++) t[obj + j] = sf.c[j]
  for (let r = 0; r < m; r++) {
    const cb = sf.c[basis[r]]
    if (cb === 0) continue
    for (let j = 0; j <= N; j++) t[obj + j] -= cb * t[r * w + j]
  }
  return makeState(cfg, sf, t, m, w, basis, rowsKept, s.labels.slice(0, N), 0, 2, {
    t: s.t + 1,
    lastPivot: null,
    event: 'phase-2',
    pivots: s.pivots,
    degenerate: false,
    visited: [basisKey(basis)],
  })
}

/**
 * The two-phase primal simplex method (Dantzig, 1951; Chvátal, 1983, ch. 8) on the linear program `problem`, as a
 * traceable algorithm with no start (`trace(simplex(problem), {}, n)`). Each step makes one pivot, or moves from
 * phase 1 to phase 2. The run stops `converged` at an optimum and `terminated` when the problem is `infeasible`,
 * `unbounded` or the method is `cycling` (a basis repeated, possible only with the `dantzig` rule). Use
 * `simplexDuals` for the duals of an optimal state and `linprog` for a one-call solve. Throws as `parseLP` does for an
 * ill-formed problem, when the algorithm is made.
 *
 * @param problem The linear program, in the form of `scipy.optimize.linprog`.
 * @param options The pivot rule and tolerance.
 * @returns The algorithm. Its start is ignored; once finished, a step returns the state unchanged.
 *
 * @example Solve a two-variable LP to its vertex
 * // Maximise x + y subject to x + 2y <= 4 and 3x + y <= 6, x, y >= 0 (so minimise -x - y).
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const s = run(simplex(problem), {}, 20)
 * print('status =', s.status)
 * print('x =', s.x)
 * print('objective =', s.objective)
 * print('pivots =', s.pivots)
 *
 * @example Walk through the pivots
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const tr = trace(simplex(problem), {}, 10)
 * print('entering =', tr.steps.map((s) => s.labels[s.entering] ?? '-'))
 * print('x =', tr.steps.map((s) => Array.from(s.x.data)))
 * print('objective =', tr.steps.map((s) => s.objective))
 *
 * @example An equality row needs phase 1
 * // Minimise x + 2y subject to x + y = 3, x >= 1, y >= 0: phase 1 drives the artificial a1 to zero.
 * const tr = trace(simplex({ c: [1, 2], A_eq: [[1, 1]], b_eq: [3], bounds: [[1, null], [0, null]] }), {}, 10)
 * print('phase =', tr.steps.map((s) => s.phase))
 * print('objective =', tr.steps.map((s) => s.objective))
 * print('x =', tr.steps.at(-1).x)
 */
export function simplex(problem: LinearProgram, options: SimplexOptions = {}): Algorithm<object, SimplexState> {
  const cfg: Config = { rule: options.rule ?? 'bland', tolerance: options.tolerance ?? 1e-9 }
  const sf = standardForm(parseLP(problem))
  return {
    name: 'simplex',
    init: () => initialState(cfg, sf),
    step: (s) => step(cfg, s),
  }
}

/**
 * One simplex step: nothing once the run has finished, the move to phase 2 when no column enters a running phase 1,
 * and otherwise the pivot `s` names.
 *
 * @param cfg The pivot rule and tolerance.
 * @param s The current state (not modified).
 * @returns The next state.
 */
function step(cfg: Config, s: SimplexState): SimplexState {
  if (s.status !== 'running') return s
  if (s.entering < 0) return toPhase2(cfg, s)
  const m = s.basis.shape[0]
  const w = s.tableau.shape[1]
  const t = Float64Array.from(s.tableau.data)
  const basis = Int32Array.from(s.basis.data)
  const degenerate = t[s.leaving * w + w - 1] <= cfg.tolerance
  const leavingColumn = basis[s.leaving]
  pivotTableau(t, m + 1, w, s.leaving, s.entering)
  basis[s.leaving] = s.entering
  return makeState(cfg, s.standard, t, m, w, basis, Int32Array.from(s.rowsKept.data), s.labels, s.artificial, s.phase, {
    t: s.t + 1,
    lastPivot: { row: s.leaving, column: s.entering, leavingColumn },
    event: 'pivot',
    pivots: s.pivots + 1,
    degenerate,
    visited: s.visited,
  })
}

/**
 * The duals of a phase-2 simplex state in scipy's convention: $\yvec$ solves $\Bmat^\top\yvec = \cvec_B$ for the
 * basis matrix $\Bmat$ of the standard form, mapped back to the rows and bounds of the original problem (see
 * `LinearProgramDuals`). They are the optimal duals when the state is `optimal`; a singular basis gives
 * $\yvec = \zeros$.
 *
 * @param s A simplex state, normally the optimal one a run of `simplex` ends on.
 * @returns The duals of the original $\le$ rows, equality rows, lower and upper bounds.
 *
 * @example The duals of a vertex are the prices of its binding rows
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const d = simplexDuals(run(simplex(problem), {}, 20))
 * // Each dual is the change in the optimal objective per unit of its right-hand side.
 * print('row duals =', d.ineq)
 * print('lower-bound duals =', d.lower)
 */
export function simplexDuals(s: SimplexState): LinearProgramDuals {
  const sf = s.standard
  const m = s.basis.shape[0]
  const kept = s.rowsKept.data
  const Bt = new Float64Array(m * m)
  const cb = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    const col = s.basis.data[r]
    cb[r] = col < sf.N ? sf.c[col] : 0
    for (let i = 0; i < m; i++) Bt[r * m + i] = col < sf.N ? sf.A[kept[i] * sf.N + col] : kept[i] === r ? 1 : 0
  }
  const { x: y } = solve(Bt, m, cb)
  return recoverDuals(sf, y, kept)
}

/** Which linear-programming method `linprog` uses. */
export type LinearProgramMethod = 'simplex' | 'interior-point'

/**
 * The outcome of solving a linear program: `optimal`, `infeasible`, `unbounded`, `cycling` (simplex only), `limit`
 * (stopped at `maxSteps`) or `diverged` (interior point only: the iterates became non-finite or stopped improving).
 */
export type LinearProgramStatus = 'optimal' | 'infeasible' | 'unbounded' | 'cycling' | 'limit' | 'diverged'

/** The result of `linprog`. */
export interface LinearProgramResult {
  /** How the solve ended. */
  status: LinearProgramStatus
  /** The solution, length $n$; NaN unless `status` is `optimal`. */
  x: Tensor
  /** $\cvec^\top\xvec$ at the optimum; `-Infinity` when unbounded; NaN otherwise. */
  objective: Scalar
  /** Steps taken: simplex pivots (and the move to phase 2), or interior-point iterations. */
  steps: Size
  /** The method that produced the result. */
  method: LinearProgramMethod
  /** Duals, slacks, reduced costs and the optimality checks, at an optimum; null otherwise. */
  report: DualityReport | null
  /**
   * When unbounded (simplex): a direction in $\xvec$-space along which $\cvec^\top\xvec$ decreases without bound;
   * null otherwise.
   */
  ray: Tensor | null
}

/** Options for `linprog`. `maxSteps` defaults to 10 000 for the simplex method and 200 for the interior point. */
export interface LinprogOptions extends Pick<RunOptions, 'maxSteps'> {
  /** `simplex` (default) or `interior-point` (Mehrotra's predictor–corrector). */
  method?: LinearProgramMethod
  /** Simplex pivot rule (default `bland`). */
  rule?: SimplexRule
  /** Tolerance (default 1e-9). */
  tolerance?: Scalar
}

/**
 * A result for a run that did not reach an optimum: $\xvec$ all NaN, no report, and an objective of `-Infinity` when
 * unbounded and NaN otherwise.
 *
 * @param n The number of variables (the length of $\xvec$).
 * @param status How the run ended.
 * @param steps The steps taken.
 * @param method The method that ran.
 * @param ray The unbounded direction, when there is one.
 * @returns The result.
 */
export function unsolved(
  n: Size,
  status: LinearProgramStatus,
  steps: Size,
  method: LinearProgramMethod,
  ray: Tensor | null = null,
): LinearProgramResult {
  return {
    status,
    x: vector(new Float64Array(n).fill(NaN)),
    objective: status === 'unbounded' ? -Infinity : NaN,
    steps,
    method,
    report: null,
    ray,
  }
}

/**
 * Solve a linear program with the simplex method (a `run` of `simplex` to completion); `linprog`'s simplex path. At an
 * optimum the result carries the duality report of `simplexDuals`; a run still going at `maxSteps` ends as `limit`.
 *
 * @param problem The linear program.
 * @param options The pivot rule, tolerance and `maxSteps` (default 10 000).
 * @returns The result.
 */
export function simplexSolve(
  problem: LinearProgram,
  options: Omit<LinprogOptions, 'method'> = {},
): LinearProgramResult {
  const s = run(simplex(problem, options), {}, options.maxSteps ?? 10_000)
  const n = s.standard.lp.n
  if (s.status === 'optimal') {
    const duals = simplexDuals(s)
    return {
      status: 'optimal',
      x: s.x,
      objective: s.objective,
      steps: s.t,
      method: 'simplex',
      report: dualityReport(s.standard.lp, s.x, duals),
      ray: null,
    }
  }
  return unsolved(n, s.status === 'running' ? 'limit' : s.status, s.t, 'simplex', s.ray)
}
