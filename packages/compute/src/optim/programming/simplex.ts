/**
 * The two-phase primal simplex method on a dense tableau (Dantzig, 1951; the two-phase method as in Chvátal, 1983,
 * "Linear Programming", ch. 8; Bland's rule from Bland, 1977, "New finite pivoting rules for the simplex method",
 * Mathematics of Operations Research 2(2)). Every state exposes the tableau, the basis, the next entering column and
 * leaving row, and the ratio test, so a figure can walk through the pivots.
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

/** What a simplex step did. */
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
  /** 1 while finding a feasible basis (minimising the sum of artificial variables), 2 while optimising cᵀx. */
  phase: 1 | 2
  /**
   * The tableau, (m + 1) × (k + 1): rows 0…m−1 are the constraints B⁻¹A | B⁻¹b, the last row holds the reduced costs
   * and, in its last entry, minus the current phase's objective (in standard form).
   */
  tableau: Tensor
  /** The basic column of each constraint row, int32, length m. */
  basis: Tensor
  /** The label of each of the k columns: `x1`, `x1+`/`x1-` (a free variable's parts), `s1` (slacks), `t1`, `a1`. */
  labels: readonly string[]
  /** The standard-form row of each tableau row (rows found redundant in phase 1 are removed), int32. */
  rowsKept: Tensor
  /** Number of artificial columns (the last columns before the right-hand side); 0 in phase 2. */
  artificial: Size
  /** The column that enters at the next pivot, or −1 when there is none. */
  entering: Index
  /** The tableau row that leaves at the next pivot, or −1. */
  leaving: Index
  /** The ratio test for the entering column, length m: rhsᵢ / aᵢₑ where aᵢₑ > 0, else Infinity. */
  ratios: Tensor
  /** The pivot made by the last step, or null. */
  lastPivot: { row: Index; column: Index; leavingColumn: Index } | null
  event: SimplexEvent
  status: SimplexStatus
  /** Pivots made so far (steps count the move to phase 2 as well). */
  pivots: Size
  /** Phase 1: the sum of the artificial variables. Phase 2: cᵀx. */
  objective: Scalar
  /** The current basic solution in the original variables, length n. */
  x: Tensor
  /** The current basic solution in the tableau's columns, length k. */
  z: Tensor
  /** True when the last pivot was degenerate: the entering variable entered at zero and x did not move. */
  degenerate: boolean
  /** Bases visited in the current phase (sorted column lists), to detect cycling. */
  visited: readonly string[]
  /** When unbounded: a direction d in x-space along which the objective decreases without bound. */
  ray: Tensor | null
  /** The standard form the tableau is built on. */
  standard: StandardForm
  converged: boolean
  terminated: boolean
}

/** The pivot rule and tolerance a run closes over. */
type Config = { rule: SimplexRule; tolerance: Scalar }

// ---------------------------------------------------------------------------------------------------------------------
// Tableau arithmetic on raw arrays; `w` is the row width (columns + 1).

/** Gauss–Jordan pivot on (r, c) in place: row r is scaled to a unit pivot and column c is cleared elsewhere. */
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

/** The basic solution of a tableau: basic columns take their row's right-hand side, the others 0. */
export function basicSolution(t: Float64Array, m: number, w: number, basis: ArrayLike<number>): Float64Array {
  const z = new Float64Array(w - 1)
  for (let r = 0; r < m; r++) z[basis[r]] = t[r * w + w - 1]
  return z
}

/** A key for a basis as a set. */
const basisKey = (basis: ArrayLike<number>) =>
  Array.from(basis)
    .sort((a, b) => a - b)
    .join(',')

type Choice = {
  entering: number
  leaving: number
  ratios: Float64Array
  status: SimplexStatus
  ray: Float64Array | null
}

/** Choose the next pivot of a tableau, or decide the phase is over. */
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

/** Assemble a state from raw tableau data. */
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

/** The initial tableau: slack columns where a row has one, artificial columns elsewhere. */
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

/** Leave phase 1: pivot out artificials left basic at zero, drop redundant rows and the artificial columns. */
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
 * `simplexDuals` for the duals of an optimal state and `linprog` for a one-call solve.
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
 * The duals of a phase-2 simplex state in scipy's convention: y solves Bᵀy = c_B for the basis matrix B of the
 * standard form, mapped back to the rows and bounds of the original problem (see `LinearProgramDuals`).
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

/** The outcome of solving a linear program. */
export type LinearProgramStatus = 'optimal' | 'infeasible' | 'unbounded' | 'cycling' | 'limit' | 'diverged'

/** The result of `linprog`. */
export interface LinearProgramResult {
  status: LinearProgramStatus
  /** The solution, length n; NaN unless `status` is `optimal`. */
  x: Tensor
  /** cᵀx at the optimum; −Infinity when unbounded; NaN otherwise. */
  objective: Scalar
  /** Steps taken: simplex pivots (and the move to phase 2), or interior-point iterations. */
  steps: Size
  method: LinearProgramMethod
  /** Duals, slacks, reduced costs and the optimality checks, at an optimum; null otherwise. */
  report: DualityReport | null
  /** When unbounded (simplex): a direction in x-space along which cᵀx decreases without bound. */
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

/** A result for a run that did not reach an optimum. */
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

/** Solve a linear program with the simplex method (a `run` of `simplex` to completion); `linprog`'s simplex path. */
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
