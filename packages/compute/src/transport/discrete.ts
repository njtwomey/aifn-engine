/**
 * Discrete optimal transport between weighted point sets: cost matrices, the exact plan, and entropic transport by
 * Sinkhorn iterations.
 *
 * Every function here solves Kantorovich's problem for weights $\avec \in \reals^n$ and $\bvec \in \reals^m$ with
 * equal totals and a cost matrix $\Cmat$ ($n \times m$): minimise $\inner{\Cmat}{\Pmat} = \sum_{ij} C_{ij} P_{ij}$
 * over couplings $\Pmat \ge 0$ with $\Pmat\ones = \avec$ and $\Pmat^\top\ones = \bvec$, where $P_{ij}$ is the mass
 * moved from $\xvec_i$ to $\yvec_j$. The exact plan uses the Hungarian algorithm for equal numbers of equally weighted
 * points and the simplex method otherwise, both from `aifn-compute/optim/programming`. Entropic transport adds
 * $\varepsilon \KL(\Pmat \,\Vert\, \avec\bvec^\top)$ and is solved by log-domain Sinkhorn iterations (Cuturi, 2013,
 * NeurIPS; Peyré and Cuturi, 2019, "Computational Optimal Transport", §4.4), as a traceable algorithm. Matrices are
 * read and returned row-major; the reading helpers here are shared with the rest of the module.
 */

import { hungarian, linprog } from 'aifn-compute/optim/programming'
import { copy, dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Scalar, Size, Status, VectorLike, VectorLike as WeightsInput } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { VectorLike as WeightsInput } from 'aifn-compute/foundation/contracts'

/** Points: an $n \times d$ tensor, rows of coordinates, or a 1-D array or tensor ($n$ points on a line, $d = 1$). */
export type PointsInput = Tensor | readonly (readonly number[])[] | readonly number[]
/** A cost matrix: a rank-2 tensor or an array of rows of equal length. */
export type CostInput = Tensor | readonly (readonly number[])[]

/**
 * Points as a fresh row-major float64 array with their count $n$ and dimension $d$. A rank-1 tensor or a flat array of
 * numbers is $n$ points on a line; an empty array is no points with $d = 1$. Rows are not checked for equal length.
 * Throws `ShapeError` for a tensor of rank other than 1 or 2.
 *
 * @param x The points: an $n \times d$ tensor, an array of $n$ rows of $d$ coordinates, or $n$ numbers.
 * @param what The caller's name, for error messages.
 * @returns `v`, the coordinates (row `i` occupies entries `i * d` to `i * d + d - 1`), with the count `n` and the
 *   dimension `d`.
 */
export function readPoints(x: PointsInput, what: string): { v: Float64Array; n: number; d: number } {
  if (isTensor(x)) {
    const v = copy(x, 'float64').data as Float64Array
    if (x.shape.length === 1) return { v, n: x.shape[0], d: 1 }
    if (x.shape.length === 2) return { v, n: x.shape[0], d: x.shape[1] }
    throw new ShapeError(what, `${what}: points must be rank 1 or 2`)
  }
  if (x.length === 0) return { v: new Float64Array(0), n: 0, d: 1 }
  if (typeof x[0] === 'number') return { v: Float64Array.from(x as readonly number[]), n: x.length, d: 1 }
  const rows = x as readonly (readonly number[])[]
  const d = rows[0].length
  const v = new Float64Array(rows.length * d)
  rows.forEach((r, i) => v.set(r, i * d))
  return { v, n: rows.length, d }
}

/**
 * A weight vector as a fresh float64 array (the input is not modified). Values are not checked for sign or total.
 *
 * @param x The vector: a rank-0 or rank-1 tensor, or an array of numbers.
 * @param what The caller's name, for error messages.
 * @returns A copy of its entries.
 */
export function readVector(x: WeightsInput, what: string): Float64Array {
  return dense.toF64(x, what)
}

/**
 * An $n \times m$ cost matrix as a fresh row-major float64 array, with its shape checked (a wrong shape or ragged rows
 * throw).
 *
 * @param c The cost matrix: a rank-2 tensor or an array of rows.
 * @param n The number of rows required (the number of source points).
 * @param m The number of columns required (the number of target points).
 * @param what The caller's name, for error messages.
 * @returns The entries, row-major: $C_{ij}$ at index `i * m + j`.
 */
export function readCost(c: CostInput, n: Size, m: Size, what: string): Float64Array {
  return dense.toMatrixF64(c, what, n, m).data
}

/**
 * Uniform weights: $n$ entries of $1/n$, a probability vector, as POT's `ot.unif`.
 *
 * @param n The number of points.
 * @returns A float64 vector of length $n$, each entry $1/n$.
 *
 * @example Four equally weighted points
 * print('weights =', uniformWeights(4))
 */
export function uniformWeights(n: Size): Tensor {
  return fromData(new Float64Array(n).fill(1 / n))
}

/**
 * The cost matrix $C_{ij} = \norm{\xvec_i - \yvec_j}^p$ between two point sets: the Euclidean norm by default, and
 * with the default $p = 2$ the squared Euclidean distance (as POT's `ot.dist`); `metric: 'cityblock'` uses the
 * $\ell_1$ norm $\sum_k \abs{x_{ik} - y_{jk}}$, raised to $p$. Throws `ShapeError` when the points differ in
 * dimension.
 *
 * @param x The $n$ source points, of dimension $d$.
 * @param y The $m$ target points, of the same dimension $d$.
 * @param options The exponent and the norm.
 * @param options.p The power $p$ the distance is raised to (2, the squared distance, by default; 1 for the distance
 *   itself).
 * @param options.metric `'euclidean'` (default) for $\norm{\xvec - \yvec}_2$, `'cityblock'` for
 *   $\norm{\xvec - \yvec}_1$.
 * @returns The $n \times m$ cost matrix $\Cmat$.
 *
 * @example Squared, plain and city-block distances to the origin
 * const x = [[0, 0], [3, 4]]
 * const y = [[0, 0]]
 * print('squared Euclidean:', costMatrix(x, y))
 * print('Euclidean:', costMatrix(x, y, { p: 1 }))
 * print('city block:', costMatrix(x, y, { p: 1, metric: 'cityblock' }))
 */
export function costMatrix(
  x: PointsInput,
  y: PointsInput,
  { p = 2, metric = 'euclidean' }: { p?: number; metric?: 'euclidean' | 'cityblock' } = {},
): Tensor {
  const X = readPoints(x, 'costMatrix')
  const Y = readPoints(y, 'costMatrix')
  if (X.d !== Y.d) throw new ShapeError('costMatrix', `costMatrix: points of dimension ${X.d} and ${Y.d}`)
  const out = new Float64Array(X.n * Y.n)
  for (let i = 0; i < X.n; i++)
    for (let j = 0; j < Y.n; j++) {
      let s = 0
      for (let k = 0; k < X.d; k++) {
        const diff = X.v[i * X.d + k] - Y.v[j * Y.d + k]
        s += metric === 'euclidean' ? diff * diff : Math.abs(diff)
      }
      const dist = metric === 'euclidean' ? Math.sqrt(s) : s
      out[i * Y.n + j] = p === 2 && metric === 'euclidean' ? s : dist ** p
    }
  return fromData(out, [X.n, Y.n])
}

/** An exact optimal transport plan: the result of `exactTransport`. */
export interface TransportPlan {
  /** The plan $\Pmat$ ($n \times m$): $P_{ij}$ is the mass moved from $\xvec_i$ to $\yvec_j$. */
  plan: Tensor
  /** $\inner{\Cmat}{\Pmat}$, the optimal cost. */
  cost: Scalar
  /**
   * The dual potential $\fvec$ of the rows ($n$), with $f_i + g_j \le C_{ij}$ for all $i, j$ and equality on the
   * support of the plan; null for the Hungarian path.
   */
  f: Tensor | null
  /** The dual potential $\gvec$ of the columns ($m$), paired with `f`; null for the Hungarian path. */
  g: Tensor | null
  /** Which solver found the plan: the Hungarian algorithm or the simplex method. */
  method: 'hungarian' | 'simplex'
  /** For `hungarian`, the column matched to each row (int32); null for the simplex path. */
  assignment: Tensor | null
}

/**
 * Whether every weight is $1/n$, to within $10^{-12}$.
 *
 * @param w The weights.
 * @param n The count that $1/n$ is taken from (the length of `w`).
 * @returns True when the weights are uniform.
 */
function isUniform(w: Float64Array, n: number): boolean {
  return w.every((v) => Math.abs(v - 1 / n) < 1e-12)
}

/**
 * The exact optimal transport plan minimising $\inner{\Cmat}{\Pmat}$ over couplings $\Pmat \ge 0$ with row sums
 * $\avec$ and column sums $\bvec$ (Kantorovich's problem), as POT's `ot.emd`. With $n = m$ and uniform weights an
 * optimal plan is a permutation scaled by $1/n$ (Birkhoff–von Neumann), found by the Hungarian algorithm; otherwise the
 * linear program in the $nm$ entries of $\Pmat$ is solved by the simplex method, which also gives dual potentials.
 * The program is dense, so this is for small problems. Totals that differ by more than $10^{-9} \max(1, \sum_i a_i)$
 * throw `ShapeError`; a linear program that does not end optimal throws `Error`.
 *
 * @param a The source weights $\avec$ ($n$ non-negative values).
 * @param b The target weights $\bvec$ ($m$ non-negative values), with the same total as $\avec$.
 * @param cost The cost matrix $\Cmat$, $n \times m$: $C_{ij}$ is the cost of moving a unit of mass from source $i$ to
 *   target $j$.
 * @returns The plan, its cost, the method used, and the dual potentials (simplex) or the assignment (Hungarian).
 *
 * @example Three points each move to the nearest shifted point
 * // Targets 0.5 to the right of the sources, listed out of order: source i goes to target (i + 1) mod 3.
 * const C = costMatrix([0, 1, 2], [2.5, 0.5, 1.5])
 * const r = exactTransport(uniformWeights(3), uniformWeights(3), C)
 * print('method:', r.method)
 * print('assignment =', r.assignment)
 * print('plan =', r.plan)
 * print('cost =', r.cost)
 *
 * @example Unequal weights split mass, and the duals certify the cost
 * // Half the mass at 0 and half at 1, moved to a quarter at 0, a quarter at 0.5 and a half at 1.
 * const a = [0.5, 0.5]
 * const b = [0.25, 0.25, 0.5]
 * const r = exactTransport(a, b, costMatrix([0, 1], [0, 0.5, 1]))
 * print('method:', r.method)
 * print('plan =', r.plan)
 * print('cost =', r.cost)
 * print('f =', r.f, ' g =', r.g)
 */
export function exactTransport(a: WeightsInput, b: WeightsInput, cost: CostInput): TransportPlan {
  const av = readVector(a, 'exactTransport a')
  const bv = readVector(b, 'exactTransport b')
  const n = av.length
  const m = bv.length
  const C = readCost(cost, n, m, 'exactTransport')
  const ta = av.reduce((s, v) => s + v, 0)
  const tb = bv.reduce((s, v) => s + v, 0)
  if (Math.abs(ta - tb) > 1e-9 * Math.max(1, ta))
    throw new ShapeError('exactTransport', `exactTransport: masses ${ta} and ${tb} differ`)
  if (n === m && isUniform(av, n) && isUniform(bv, m)) {
    const r = hungarian(fromData(C, [n, m]))
    const plan = new Float64Array(n * m)
    const assign = r.assignment.data
    for (let i = 0; i < n; i++) plan[i * m + assign[i]] = 1 / n
    return {
      plan: fromData(plan, [n, m]),
      cost: r.cost / n,
      f: null,
      g: null,
      method: 'hungarian',
      assignment: r.assignment,
    }
  }
  // Variables P_ij in row-major order; one equality per row sum and per column sum.
  const Aeq: number[][] = []
  for (let i = 0; i < n; i++) Aeq.push(Array.from({ length: n * m }, (_, k) => (Math.floor(k / m) === i ? 1 : 0)))
  for (let j = 0; j < m; j++) Aeq.push(Array.from({ length: n * m }, (_, k) => (k % m === j ? 1 : 0)))
  const r = linprog({ c: Array.from(C), A_eq: Aeq, b_eq: [...av, ...bv] })
  if (r.status !== 'optimal') throw new Error(`exactTransport: the linear program ended ${r.status}`)
  const eq = r.report ? (copy(r.report.duals.eq, 'float64').data as Float64Array) : null
  return {
    plan: fromData(copy(r.x, 'float64').data as Float64Array, [n, m]),
    cost: r.objective,
    f: eq ? fromData(eq.slice(0, n)) : null,
    g: eq ? fromData(eq.slice(n)) : null,
    method: 'simplex',
    assignment: null,
  }
}

// ── Sinkhorn ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `sinkhornSteps` and `sinkhorn`. */
export interface SinkhornOptions {
  /**
   * Entropic regularisation $\varepsilon > 0$, in cost units: the weight of $\KL(\Pmat \,\Vert\, \avec\bvec^\top)$,
   * as POT's `reg`.
   */
  epsilon: Scalar
  /** Stop when the $\ell_1$ error of the row marginal, $\norm{\Pmat\ones - \avec}_1$, is below this. Default 1e-9. */
  tolerance?: Scalar
}

/** The start of `sinkhornSteps`: an optional initial column potential $\gvec$ (default zeros), e.g. to warm-start. */
export type SinkhornStart = { g?: VectorLike }

/** One state of log-domain Sinkhorn. */
export interface SinkhornState extends Status {
  /** The dual potential $\fvec$ of the rows ($n$). */
  f: Tensor
  /** The dual potential $\gvec$ of the columns ($m$). */
  g: Tensor
  /** The plan $P_{ij} = a_i b_j \exp((f_i + g_j - C_{ij})/\varepsilon)$, $n \times m$. */
  plan: Tensor
  /**
   * $\ell_1$ error of the row sums, $\norm{\Pmat\ones - \avec}_1$, after the column update (the column sums are
   * exact after a step).
   */
  marginalError: Scalar
  /** $\inner{\Cmat}{\Pmat}$. */
  transportCost: Scalar
  /**
   * The entropic objective $\inner{\Cmat}{\Pmat} + \varepsilon \KL(\Pmat \,\Vert\, \avec\bvec^\top)$, with the
   * generalised $\KL(\Pmat \,\Vert\, \Qmat) = \sum_{ij} (P_{ij} \log(P_{ij}/Q_{ij}) - P_{ij} + Q_{ij})$.
   */
  value: Scalar
  /**
   * $\avec^\top\fvec + \bvec^\top\gvec$. For weights that each sum to 1 this is the entropic dual objective after
   * every step, and it increases monotonically.
   */
  dual: Scalar
  /** The row-marginal error is below `tolerance`. */
  converged: boolean
  /** A potential became non-finite ($\varepsilon$ too small for the cost scale is the usual cause). */
  diverged: boolean
}

/**
 * The soft minimum $-\varepsilon \log \sum_k \exp((u_k - c_k)/\varepsilon + \log w_k)$, computed stably by shifting
 * by the maximum exponent: one entry of a soft $c$-transform. When no exponent is finite (every weight zero, say) it
 * returns $-\varepsilon$ times that exponent ($+\infty$ for all $-\infty$).
 *
 * @param pot The potential $\uvec$ of the other side, one value per term.
 * @param cost The cost $c_k$ of term `k` (a row or column of $\Cmat$).
 * @param logw The log-weights $\log w_k$ of the other side, one per term.
 * @param eps The regularisation $\varepsilon$.
 * @param buf Scratch space of at least `pot.length` values, overwritten with the exponents.
 * @returns The soft minimum.
 */
function softMin(
  pot: Float64Array,
  cost: (k: number) => number,
  logw: Float64Array,
  eps: number,
  buf: Float64Array,
): number {
  let mx = -Infinity
  for (let k = 0; k < pot.length; k++) {
    const v = (pot[k] - cost(k)) / eps + logw[k]
    buf[k] = v
    if (v > mx) mx = v
  }
  if (!Number.isFinite(mx)) return -eps * mx
  let s = 0
  for (let k = 0; k < pot.length; k++) s += Math.exp(buf[k] - mx)
  return -eps * (mx + Math.log(s))
}

/**
 * A Sinkhorn problem as working arrays: the weights `a` ($n$) and `b` ($m$), the cost `C` (row-major, $n \times m$),
 * the regularisation `epsilon` and the stopping `tolerance` on the row-marginal error.
 */
type SinkhornProblem = { a: Float64Array; b: Float64Array; C: Float64Array; epsilon: Scalar; tolerance: Scalar }

/**
 * The Sinkhorn state for given potentials: the plan $P_{ij} = a_i b_j \exp((f_i + g_j - C_{ij})/\varepsilon)$, its
 * row-marginal error, transport cost, entropic objective and dual objective, and the convergence and divergence flags.
 *
 * @param options The problem's working arrays and settings.
 * @param options.a The source weights $\avec$ ($n$).
 * @param options.b The target weights $\bvec$ ($m$).
 * @param options.C The cost matrix $\Cmat$, row-major, $n \times m$.
 * @param options.epsilon The regularisation $\varepsilon$.
 * @param options.tolerance The row-marginal error below which the state is `converged`.
 * @param t The step number to record.
 * @param f The row potential $\fvec$ ($n$ values); kept by the state, not copied.
 * @param g The column potential $\gvec$ ($m$ values); kept by the state, not copied.
 * @returns The state.
 */
function summarise(
  { a, b, C, epsilon: eps, tolerance }: SinkhornProblem,
  t: Size,
  f: Float64Array,
  g: Float64Array,
): SinkhornState {
  const n = a.length
  const m = b.length
  const P = new Float64Array(n * m)
  let cost = 0
  let kl = 0
  let err = 0
  for (let i = 0; i < n; i++) {
    let row = 0
    for (let j = 0; j < m; j++) {
      const ab = a[i] * b[j]
      const p = ab * Math.exp((f[i] + g[j] - C[i * m + j]) / eps)
      P[i * m + j] = p
      row += p
      cost += p * C[i * m + j]
      // Generalised KL(P ‖ a⊗b) = Σ p log(p/ab) − p + ab: the p log p terms vanish at p = 0, the + ab term does not.
      kl += (p > 0 ? p * Math.log(p / ab) - p : 0) + ab
    }
    err += Math.abs(row - a[i])
  }
  let dual = 0
  for (let i = 0; i < n; i++) dual += a[i] * f[i]
  for (let j = 0; j < m; j++) dual += b[j] * g[j]
  return {
    t,
    f: fromData(f, [n]),
    g: fromData(g, [m]),
    plan: fromData(P, [n, m]),
    marginalError: err,
    transportCost: cost,
    value: cost + eps * kl,
    dual,
    converged: err < tolerance,
    diverged: !Number.isFinite(dual) || !Number.isFinite(err),
  }
}

/**
 * Sinkhorn's algorithm in the log domain as a traceable algorithm (Schmitzer, 2019, SIAM J. Sci. Comput. 41(3)): each
 * step sets $\fvec$ to the soft $c$-transform of $\gvec$,
 * $f_i = -\varepsilon \log \sum_j b_j \exp((g_j - C_{ij})/\varepsilon)$, then $\gvec$ to that of $\fvec$, so the
 * column marginal is exact and the row marginal carries the error. The plan is relative to the product measure
 * $\avec\bvec^\top$, so $\varepsilon$ is the weight of $\KL(\Pmat \,\Vert\, \avec\bvec^\top)$; the converged plan is
 * that of POT's `ot.sinkhorn(a, b, C, reg)` with `reg` $= \varepsilon$. As $\varepsilon \to 0$ the plan approaches an
 * exact optimal plan; large $\varepsilon$ blurs it towards $\avec\bvec^\top$. Convergence slows as $\varepsilon$
 * shrinks relative to the spread of the costs. `init` takes `{ g? }`, the start of the column potential (zeros by
 * default). Throws `DomainError` unless $\varepsilon > 0$.
 *
 * @param a The source weights $\avec$ ($n$ positive values).
 * @param b The target weights $\bvec$ ($m$ positive values), with the same total as $\avec$.
 * @param cost The cost matrix $\Cmat$, $n \times m$.
 * @param options The regularisation and the stopping tolerance.
 * @param options.epsilon The entropic regularisation $\varepsilon > 0$, in cost units.
 * @param options.tolerance The $\ell_1$ row-marginal error below which a state is `converged`.
 * @returns The algorithm: its state holds the potentials, the plan and the objectives.
 *
 * @example The row-marginal error falls and the dual rises, step by step
 * const alg = sinkhornSteps([0.5, 0.5], [0.5, 0.5], costMatrix([0, 1], [0.1, 1.1]), { epsilon: 0.5 })
 * for (const steps of [1, 2, 5, 20]) {
 *   const s = run(alg, {}, steps)
 *   print(`after ${steps} steps: marginal error =`, s.marginalError, ' dual =', s.dual)
 * }
 *
 * @example Warm-start from a converged column potential
 * const alg = sinkhornSteps([0.5, 0.5], [0.5, 0.5], costMatrix([0, 1], [0.1, 1.1]), { epsilon: 0.5 })
 * const cold = run(alg, {}, 1000)
 * print('steps from zeros:', cold.t)
 * print('steps from the converged g:', run(alg, { g: cold.g }, 1000).t)
 */
export function sinkhornSteps(
  a: WeightsInput,
  b: WeightsInput,
  cost: CostInput,
  { epsilon, tolerance = 1e-9 }: SinkhornOptions,
): Algorithm<SinkhornStart, SinkhornState> {
  if (!(epsilon > 0)) throw new DomainError('sinkhorn', 'sinkhorn: epsilon must be positive')
  const av = readVector(a, 'sinkhorn a')
  const bv = readVector(b, 'sinkhorn b')
  const n = av.length
  const m = bv.length
  const problem: SinkhornProblem = { a: av, b: bv, C: readCost(cost, n, m, 'sinkhorn'), epsilon, tolerance }
  const la = av.map(Math.log)
  const lb = bv.map(Math.log)
  return {
    name: 'sinkhorn',
    init: ({ g } = {}) => {
      const g0 = g === undefined ? new Float64Array(m) : readVector(g, 'sinkhorn g')
      if (g0.length !== m) throw new ShapeError('sinkhorn', `sinkhorn: g must have length ${m}`)
      return summarise(problem, 0, new Float64Array(n), g0)
    },
    step: (s) => {
      const C = problem.C
      const f = new Float64Array(n)
      const g = new Float64Array(m)
      const gPrev = dense.data(s.g)
      const bufN = new Float64Array(n)
      const bufM = new Float64Array(m)
      for (let i = 0; i < n; i++) f[i] = softMin(gPrev, (j) => C[i * m + j], lb, epsilon, bufM)
      for (let j = 0; j < m; j++) g[j] = softMin(f, (i) => C[i * m + j], la, epsilon, bufN)
      return summarise(problem, s.t + 1, f, g)
    },
  }
}

/**
 * Entropic optimal transport, as POT's `ot.sinkhorn`: `sinkhornSteps` run until the row-marginal error is below
 * `tolerance`, a potential diverges, or `maxSteps` steps (default 1000) have been taken. Check `converged` on the
 * result: a small $\varepsilon$ can need many more steps.
 *
 * @param a The source weights $\avec$ ($n$ positive values).
 * @param b The target weights $\bvec$ ($m$ positive values), with the same total as $\avec$.
 * @param cost The cost matrix $\Cmat$, $n \times m$.
 * @param options The regularisation `epsilon` and `tolerance` of `SinkhornOptions`, and the step limit.
 * @param options.maxSteps The most Sinkhorn steps to take.
 * @param options.options The remaining fields, `epsilon` (required) and `tolerance`, passed to `sinkhornSteps`.
 * @returns The last state: the plan, the potentials, the objectives and whether it converged.
 *
 * @example The plan approaches the exact one as epsilon shrinks
 * // Two points matched to themselves: the exact plan is diagonal, and the off-diagonal mass is the blur of epsilon.
 * const C = [[0, 1], [1, 0]]
 * for (const epsilon of [1, 0.3, 0.1]) {
 *   const { plan } = sinkhorn([0.5, 0.5], [0.5, 0.5], C, { epsilon })
 *   print(`epsilon = ${epsilon}: plan =`, plan)
 * }
 * print('exact plan =', exactTransport([0.5, 0.5], [0.5, 0.5], C).plan)
 *
 * @example The entropic objective meets the dual at convergence
 * const s = sinkhorn([0.5, 0.5], [0.5, 0.5], costMatrix([0, 1], [0.1, 1.1]), { epsilon: 1 })
 * print('converged:', s.converged, ' steps:', s.t)
 * print('transport cost =', s.transportCost)
 * print('entropic objective =', s.value, ' dual =', s.dual)
 */
export function sinkhorn(
  a: WeightsInput,
  b: WeightsInput,
  cost: CostInput,
  { maxSteps = 1000, ...options }: SinkhornOptions & { maxSteps?: Size },
): SinkhornState {
  return run(sinkhornSteps(a, b, cost, options), {}, maxSteps)
}
