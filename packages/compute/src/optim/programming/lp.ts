/**
 * Linear programs: the problem type, its conversion to standard form, and duality reporting.
 *
 * The problem follows `scipy.optimize.linprog`: minimise $\cvec^\top\xvec$ subject to
 * $\Amat_{\text{ub}}\xvec \le \bvec_{\text{ub}}$, $\Amat_{\text{eq}}\xvec = \bvec_{\text{eq}}$ and
 * $l_i \le x_i \le u_i$, with default bounds $0 \le x_i < \infty$. Dual values follow scipy's HiGHS convention: each is
 * the sensitivity of the optimal objective to its right-hand side ($\partial f^\star / \partial b$), so the duals of
 * $\le$ rows and of upper bounds are $\le 0$, and the duals of lower bounds are $\ge 0$ (Vanderbei, 2020, "Linear
 * Programming", ch. 5 and ch. 7).
 *
 * Every solver of the module works on the standard form built here (minimise $\cvec^\top\zvec$ subject to
 * $\Amat\zvec = \bvec$, $\zvec \ge \zeros$, $\bvec \ge \zeros$), and maps its solution and duals back to the original
 * variables, rows and bounds with `toOriginal` and `recoverDuals`.
 */

import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Index, MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { checkFinite, intTensor, matTVec, matVec, readMatrix, readVector, vector, type Mat } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A bound on one variable: `[lower, upper]`. `null`, or `-Infinity` for the lower and `Infinity` for the upper, means
 * no bound on that side.
 */
export type Bound = readonly [Scalar | null, Scalar | null]

/**
 * A linear program in the form of `scipy.optimize.linprog`: minimise $\cvec^\top\xvec$ subject to
 * $\Amat_{\text{ub}}\xvec \le \bvec_{\text{ub}}$, $\Amat_{\text{eq}}\xvec = \bvec_{\text{eq}}$ and `bounds`. To
 * maximise, negate `c` (and the optimal objective). A $\ge$ row is a $\le$ row with both sides negated. Absent
 * constraints are empty; every entry of the matrices and vectors must be finite.
 */
export interface LinearProgram {
  /** Objective coefficients $\cvec$, length $n$ (the number of variables). */
  c: VectorLike
  /** Inequality constraint matrix $\Amat_{\text{ub}}$, $m_{\text{ub}} \times n$. */
  A_ub?: MatrixLike
  /** Inequality right-hand side $\bvec_{\text{ub}}$, length $m_{\text{ub}}$. */
  b_ub?: VectorLike
  /** Equality constraint matrix $\Amat_{\text{eq}}$, $m_{\text{eq}} \times n$. */
  A_eq?: MatrixLike
  /** Equality right-hand side $\bvec_{\text{eq}}$, length $m_{\text{eq}}$. */
  b_eq?: VectorLike
  /**
   * One bound for every variable (a single `[lower, upper]` pair), or a list of $n$ bounds. Default `[0, null]`
   * (non-negative), as in scipy.
   */
  bounds?: Bound | readonly Bound[]
}

/** A linear program read into dense arrays; an absent bound is `-Infinity` (lower) or `Infinity` (upper). */
export interface ParsedLP {
  /** The number of variables. */
  n: Size
  /** Objective coefficients, length $n$. */
  c: Float64Array
  /** Inequality constraint matrix, $m_{\text{ub}} \times n$ ($0 \times n$ when there is none). */
  Aub: Mat
  /** Inequality right-hand side, length $m_{\text{ub}}$. */
  bub: Float64Array
  /** Equality constraint matrix, $m_{\text{eq}} \times n$ ($0 \times n$ when there is none). */
  Aeq: Mat
  /** Equality right-hand side, length $m_{\text{eq}}$. */
  beq: Float64Array
  /** Lower bound of each variable, length $n$. */
  lower: Float64Array
  /** Upper bound of each variable, length $n$. */
  upper: Float64Array
}

/**
 * Read and validate a linear program. Throws `ShapeError` when the sizes disagree with the length of `c`, and
 * `DomainError` for a non-finite entry in the constraints or objective, or a bound that is NaN, a lower bound of
 * `Infinity` or an upper bound of `-Infinity`. A lower bound above the upper one is not checked here.
 *
 * @param problem The linear program, with its matrices and vectors as tensors or plain arrays.
 * @returns The program as dense arrays, with the default bounds filled in.
 */
export function parseLP(problem: LinearProgram): ParsedLP {
  const c = readVector(problem.c, 'linear program: c')
  const n = c.length
  const Aub = readMatrix(problem.A_ub, 'linear program: A_ub', n)
  const bub = readVector(problem.b_ub, 'linear program: b_ub', Aub.m)
  const Aeq = readMatrix(problem.A_eq, 'linear program: A_eq', n)
  const beq = readVector(problem.b_eq, 'linear program: b_eq', Aeq.m)
  for (const [a, name] of [
    [c, 'c'],
    [Aub.a, 'A_ub'],
    [bub, 'b_ub'],
    [Aeq.a, 'A_eq'],
    [beq, 'b_eq'],
  ] as const)
    checkFinite(a, `linear program: ${name}`)
  const lower = new Float64Array(n)
  const upper = new Float64Array(n).fill(Infinity)
  const b = problem.bounds
  if (b !== undefined) {
    const single = b.length === 2 && !Array.isArray(b[0]) && (typeof b[0] === 'number' || b[0] === null)
    const list = single ? Array.from({ length: n }, () => b as Bound) : (b as readonly Bound[])
    if (list.length !== n)
      throw new ShapeError('linear program', `linear program: expected ${n} bounds, got ${list.length}`)
    for (let j = 0; j < n; j++) {
      const [lo, hi] = list[j]
      lower[j] = lo === null ? -Infinity : lo
      upper[j] = hi === null ? Infinity : hi
      if (Number.isNaN(lower[j]) || Number.isNaN(upper[j]) || lower[j] === Infinity || upper[j] === -Infinity)
        throw new DomainError('linear program', `linear program: invalid bound on x${j + 1}`)
    }
  }
  return { n, c, Aub, bub, Aeq, beq, lower, upper }
}

/** How one original variable $x$ is written in standard-form columns $\zvec$. */
export type VariableMap =
  /**
   * $x = l + z_c$ ($l$ the lower bound, $c$ = `column`). With a finite upper bound $u$, the row `boundRow` holds
   * $z_c + t = u - l$ for a slack $t$; otherwise `boundRow` is $-1$.
   */
  | { kind: 'lower'; column: Index; boundRow: Index }
  /** $x = u - z_c$ ($u$ the upper bound, $c$ = `column`), for a variable with no lower bound. */
  | { kind: 'upper'; column: Index }
  /** $x = z_p - z_m$ for a free variable, with $p$ = `plus` and $m$ = `minus`. */
  | { kind: 'free'; plus: Index; minus: Index }

/**
 * Where a standard-form row came from, and whether it was negated to make its right-hand side non-negative. `kind` is
 * `ub` (a $\le$ row), `eq` (an equality row) or `bound` (a variable's upper bound); `source` is the index of that row,
 * or of the variable; `sign` is $-1$ when the row was negated, else 1.
 */
export type RowSource = { kind: 'ub' | 'eq' | 'bound'; source: Index; sign: 1 | -1 }

/**
 * A linear program in standard form: minimise $\cvec^\top\zvec + \text{offset}$ subject to $\Amat\zvec = \bvec$,
 * $\zvec \ge \zeros$, with $\bvec \ge \zeros$.
 *
 * Variables are shifted and split so that every column is non-negative ($x = l + z$, $x = u - z$, or
 * $x = z^+ - z^-$), every $\le$ row gets a slack column, so does the finite upper bound of a variable that also has a
 * finite lower bound, and rows with a negative right-hand side are negated. The rows are the $\le$ rows, then the
 * equality rows, then the bound rows.
 */
export interface StandardForm {
  /** The number of rows, $m$. */
  m: Size
  /** The number of columns, $N$. */
  N: Size
  /** Constraint matrix $\Amat$, $m \times N$, row-major. */
  A: Float64Array
  /** Right-hand side $\bvec$, length $m$, every entry $\ge 0$. */
  b: Float64Array
  /** Objective coefficients $\cvec$ of the columns, length $N$ (0 on slack columns). */
  c: Float64Array
  /** The constant dropped from the objective: $\sum_j c_j s_j$, with $s_j$ the bound $x_j$ is shifted by (or 0). */
  offset: Scalar
  /**
   * A readable label for each column: `x1`, `x1+`, `x1-`, `s1` (slack of $\le$ row 1), `t1` (slack of `x1`'s upper
   * bound).
   */
  labels: string[]
  /** For each row, the column of a slack with coefficient $+1$ in it (a ready basic column), or $-1$. */
  slackOfRow: Int32Array
  /** Where each row came from, and its sign. */
  rows: RowSource[]
  /** How each original variable is written in the columns, length $n$. */
  variables: VariableMap[]
  /** The original problem. */
  lp: ParsedLP
}

/**
 * Convert a parsed linear program to standard form (see `StandardForm`).
 *
 * @param lp The parsed linear program, as `parseLP` returns it (not modified).
 * @returns The standard form, which keeps `lp` for mapping back.
 */
export function standardForm(lp: ParsedLP): StandardForm {
  const { n } = lp
  const labels: string[] = []
  const variables: VariableMap[] = []
  const shift = new Float64Array(n)
  // Columns for the original variables.
  for (let j = 0; j < n; j++) {
    const lo = lp.lower[j]
    const hi = lp.upper[j]
    if (Number.isFinite(lo)) {
      variables.push({ kind: 'lower', column: labels.length, boundRow: -1 })
      labels.push(`x${j + 1}`)
      shift[j] = lo
    } else if (Number.isFinite(hi)) {
      variables.push({ kind: 'upper', column: labels.length })
      labels.push(`x${j + 1}`)
      shift[j] = hi
    } else {
      variables.push({ kind: 'free', plus: labels.length, minus: labels.length + 1 })
      labels.push(`x${j + 1}+`, `x${j + 1}-`)
    }
  }
  const boundVars = variables.flatMap((v, j) => (v.kind === 'lower' && Number.isFinite(lp.upper[j]) ? [j] : []))
  const m = lp.Aub.m + lp.Aeq.m + boundVars.length
  const firstSlack = labels.length
  for (let i = 0; i < lp.Aub.m; i++) labels.push(`s${i + 1}`)
  for (const j of boundVars) labels.push(`t${j + 1}`)
  const N = labels.length
  const A = new Float64Array(m * N)
  const b = new Float64Array(m)
  const rows: RowSource[] = []
  const slackOfRow = new Int32Array(m).fill(-1)

  // Write original row (coefficients a over x, right-hand side r) into standard-form row i: substitute x.
  const writeRow = (i: number, a: Float64Array, offset: number, r: number) => {
    let rhs = r
    for (let j = 0; j < n; j++) {
      const aj = a[offset + j]
      if (aj === 0) continue
      const v = variables[j]
      rhs -= aj * shift[j]
      if (v.kind === 'lower') A[i * N + v.column] = aj
      else if (v.kind === 'upper') A[i * N + v.column] = -aj
      else {
        A[i * N + v.plus] = aj
        A[i * N + v.minus] = -aj
      }
    }
    b[i] = rhs
  }
  let i = 0
  for (let r = 0; r < lp.Aub.m; r++, i++) {
    writeRow(i, lp.Aub.a, r * n, lp.bub[r])
    A[i * N + firstSlack + r] = 1
    slackOfRow[i] = firstSlack + r
    rows.push({ kind: 'ub', source: r, sign: 1 })
  }
  for (let r = 0; r < lp.Aeq.m; r++, i++) {
    writeRow(i, lp.Aeq.a, r * n, lp.beq[r])
    rows.push({ kind: 'eq', source: r, sign: 1 })
  }
  boundVars.forEach((j, k) => {
    const v = variables[j] as { kind: 'lower'; column: number; boundRow: number }
    v.boundRow = i
    A[i * N + v.column] = 1
    A[i * N + firstSlack + lp.Aub.m + k] = 1
    b[i] = lp.upper[j] - lp.lower[j]
    slackOfRow[i] = firstSlack + lp.Aub.m + k
    rows.push({ kind: 'bound', source: j, sign: 1 })
    i++
  })
  // Make b ≥ 0: negate rows with a negative right-hand side; their slack then has coefficient −1.
  for (let r = 0; r < m; r++) {
    if (b[r] < 0) {
      b[r] = -b[r]
      for (let k = 0; k < N; k++) A[r * N + k] = -A[r * N + k]
      rows[r] = { ...rows[r], sign: -1 }
      slackOfRow[r] = -1
    }
  }
  const c = new Float64Array(N)
  let offset = 0
  for (let j = 0; j < n; j++) {
    const v = variables[j]
    offset += lp.c[j] * shift[j]
    if (v.kind === 'lower') c[v.column] = lp.c[j]
    else if (v.kind === 'upper') c[v.column] = -lp.c[j]
    else {
      c[v.plus] = lp.c[j]
      c[v.minus] = -lp.c[j]
    }
  }
  return { m, N, A, b, c, offset, labels, slackOfRow, rows, variables, lp }
}

/**
 * The original variables $\xvec$ from standard-form values $\zvec$, by the `variables` map.
 *
 * @param sf The standard form $\zvec$ belongs to.
 * @param z The values of the standard-form columns; only the first $N$ entries are read, so extra (artificial) columns
 *   are ignored.
 * @param shift Whether to add the bound each variable is shifted by. `false` maps only the linear part, for a direction
 *   such as an unbounded ray rather than a point.
 * @returns $\xvec$, length $n$.
 */
export function toOriginal(sf: StandardForm, z: ArrayLike<number>, shift = true): Float64Array {
  const x = new Float64Array(sf.lp.n)
  sf.variables.forEach((v, j) => {
    if (v.kind === 'lower') x[j] = (shift ? sf.lp.lower[j] : 0) + z[v.column]
    else if (v.kind === 'upper') x[j] = (shift ? sf.lp.upper[j] : 0) - z[v.column]
    else x[j] = z[v.plus] - z[v.minus]
  })
  return x
}

/**
 * Duals of a linear program in scipy's convention (sensitivities $\partial f^\star / \partial b$ of the optimal
 * objective).
 */
export interface LinearProgramDuals {
  /** Duals of the $\le$ rows, length $m_{\text{ub}}$; $\le 0$ at an optimum. */
  ineq: Tensor
  /** Duals of the equality rows, length $m_{\text{eq}}$; any sign. */
  eq: Tensor
  /** Duals of the lower bounds, length $n$; $\ge 0$ at an optimum, 0 where there is no lower bound. */
  lower: Tensor
  /** Duals of the upper bounds, length $n$; $\le 0$ at an optimum, 0 where there is no upper bound. */
  upper: Tensor
}

/**
 * Map standard-form row duals $\yvec$ to the original problem's duals. A row's dual is $\yvec$'s entry times the row's
 * sign; a bound's dual is the reduced cost $d = \cvec - \Amat^\top\yvec$ of its column (negated for an upper bound
 * written as $x = u - z$), or the dual of its bound row.
 *
 * @param sf The standard form the duals belong to.
 * @param y The row duals: $m$ values, one per standard-form row, or one per entry of `rowsKept`.
 * @param rowsKept The standard-form rows `y` covers, in order. Rows not listed (removed as redundant) take dual 0. Left
 *   out, `y` covers every row.
 * @returns The duals of the original rows and bounds.
 */
export function recoverDuals(sf: StandardForm, y: ArrayLike<number>, rowsKept?: ArrayLike<number>): LinearProgramDuals {
  const full = new Float64Array(sf.m)
  if (rowsKept) for (let k = 0; k < rowsKept.length; k++) full[rowsKept[k]] = y[k]
  else full.set(Array.from(y))
  const { lp } = sf
  // Reduced costs of the standard-form columns, d = c − Aᵀy.
  const d = Float64Array.from(sf.c)
  for (let i = 0; i < sf.m; i++) {
    const yi = full[i]
    if (yi !== 0) for (let k = 0; k < sf.N; k++) d[k] -= sf.A[i * sf.N + k] * yi
  }
  const ineq = new Float64Array(lp.Aub.m)
  const eq = new Float64Array(lp.Aeq.m)
  const lower = new Float64Array(lp.n)
  const upper = new Float64Array(lp.n)
  sf.rows.forEach((r, i) => {
    // A negated row has its dual negated too, so the original row's dual is sign · y.
    const value = r.sign * full[i]
    if (r.kind === 'ub') ineq[r.source] = value
    else if (r.kind === 'eq') eq[r.source] = value
    else upper[r.source] = value
  })
  sf.variables.forEach((v, j) => {
    if (v.kind === 'lower') lower[j] = d[v.column]
    else if (v.kind === 'upper') upper[j] = -d[v.column]
  })
  return { ineq: vector(ineq), eq: vector(eq), lower: vector(lower), upper: vector(upper) }
}

/** Primal and dual quantities at a point of a linear program, and the checks that certify optimality. */
export interface DualityReport {
  /** The point $\xvec$, length $n$. */
  x: Tensor
  /** $\cvec^\top\xvec$. */
  objective: Scalar
  /** The duals the report was made with, as given. */
  duals: LinearProgramDuals
  /**
   * Reduced costs $\cvec - \Amat_{\text{ub}}^\top\yvec_{\text{ub}} - \Amat_{\text{eq}}^\top\yvec_{\text{eq}}$, length
   * $n$ (the sum of the lower- and upper-bound duals at an optimum).
   */
  reducedCosts: Tensor
  /**
   * Slacks $\bvec_{\text{ub}} - \Amat_{\text{ub}}\xvec$ of the $\le$ rows, length $m_{\text{ub}}$ ($\ge 0$ when
   * feasible).
   */
  slack: Tensor
  /** Residuals $\bvec_{\text{eq}} - \Amat_{\text{eq}}\xvec$, length $m_{\text{eq}}$. */
  residualEq: Tensor
  /**
   * The dual objective $\bvec_{\text{ub}}^\top\yvec_{\text{ub}} + \bvec_{\text{eq}}^\top\yvec_{\text{eq}}$
   * $+ \sum_j l_j \lambda_j + \sum_j u_j \mu_j$, the sums over finite bounds ($\lambda_j$ and $\mu_j$ the lower- and
   * upper-bound duals).
   */
  dualObjective: Scalar
  /** `objective` minus `dualObjective`; zero at an optimum (strong duality). */
  dualityGap: Scalar
  /** Largest violation of a primal constraint or bound (0 when feasible). */
  primalInfeasibility: Scalar
  /**
   * Largest violation of a dual sign condition, or of stationarity: each reduced cost must equal the sum of its
   * variable's bound duals over finite bounds (so a free variable's must be zero). 0 when dual feasible.
   */
  dualInfeasibility: Scalar
  /**
   * Largest complementary-slackness product: $\lvert y_i s_i \rvert$ over $\le$ rows ($s_i$ the slack), and
   * $\lvert \lambda_j (x_j - l_j) \rvert$, $\lvert \mu_j (u_j - x_j) \rvert$ over finite bounds. Zero at an optimum.
   */
  complementarity: Scalar
  /**
   * 1 for each $\le$ row whose slack is at most `tolerance` in absolute value (the active, or binding, constraints),
   * else 0; int32.
   */
  active: Tensor
}

/**
 * The duality report of a point `x` and duals of a linear program: slacks, reduced costs, the dual objective, the
 * duality gap and the complementary-slackness, primal- and dual-feasibility residuals that certify optimality
 * (Vanderbei, 2020, Theorem 5.3). Throws `ShapeError` when a length disagrees with the problem.
 *
 * @param problem The linear program, or its parsed form (as `parseLP` returns it, which is not parsed again).
 * @param x The point $\xvec$, $n$ values.
 * @param duals The duals to check, in scipy's convention (as in `LinearProgramResult.report` or from `simplexDuals`).
 * @param tolerance The largest absolute slack at which a $\le$ row is reported `active`. It decides nothing else.
 * @returns The report: every residual is zero at an optimum with its duals.
 *
 * @example Certify the optimum of a two-variable LP
 * // Maximise x + y subject to x + 2y <= 4 and 3x + y <= 6, x, y >= 0. The optimum is the vertex (1.6, 1.2), where
 * // both rows bind; its duals solve A^T y = c.
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const duals = { ineq: tensor([-0.4, -0.2]), eq: tensor([]), lower: tensor([0, 0]), upper: tensor([0, 0]) }
 * const r = dualityReport(problem, [1.6, 1.2], duals)
 * print('objective =', r.objective, ' dual objective =', r.dualObjective)
 * print('duality gap =', r.dualityGap)
 * print('active rows =', r.active)
 *
 * @example A feasible point that is not optimal leaves a gap
 * const problem = { c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }
 * const duals = { ineq: tensor([-0.4, -0.2]), eq: tensor([]), lower: tensor([0, 0]), upper: tensor([0, 0]) }
 * const r = dualityReport(problem, [1, 1], duals)
 * print('slack =', r.slack)
 * print('duality gap =', r.dualityGap)
 * print('complementarity =', r.complementarity)
 */
export function dualityReport(
  problem: LinearProgram | ParsedLP,
  x: VectorLike,
  duals: LinearProgramDuals,
  tolerance = 1e-7,
): DualityReport {
  const lp = 'Aub' in problem ? problem : parseLP(problem)
  const xs = readVector(x, 'dualityReport: x', lp.n)
  const yu = readVector(duals.ineq, 'dualityReport: duals.ineq', lp.Aub.m)
  const ye = readVector(duals.eq, 'dualityReport: duals.eq', lp.Aeq.m)
  const lam = readVector(duals.lower, 'dualityReport: duals.lower', lp.n)
  const mu = readVector(duals.upper, 'dualityReport: duals.upper', lp.n)
  const Ax = matVec(lp.Aub, xs)
  const slack = lp.bub.map((b, i) => b - Ax[i])
  const Ex = matVec(lp.Aeq, xs)
  const residualEq = lp.beq.map((b, i) => b - Ex[i])
  const reduced = Float64Array.from(lp.c)
  const tu = matTVec(lp.Aub, yu)
  const te = matTVec(lp.Aeq, ye)
  for (let j = 0; j < lp.n; j++) reduced[j] -= tu[j] + te[j]
  let dualObjective = dense.dot(lp.bub, yu) + dense.dot(lp.beq, ye)
  let primal = 0
  let dual = 0
  let comp = 0
  for (let i = 0; i < lp.Aub.m; i++) {
    primal = Math.max(primal, -slack[i])
    dual = Math.max(dual, yu[i])
    comp = Math.max(comp, Math.abs(yu[i] * slack[i]))
  }
  for (let i = 0; i < lp.Aeq.m; i++) primal = Math.max(primal, Math.abs(residualEq[i]))
  for (let j = 0; j < lp.n; j++) {
    const lo = lp.lower[j]
    const hi = lp.upper[j]
    if (Number.isFinite(lo)) {
      primal = Math.max(primal, lo - xs[j])
      dualObjective += lo * lam[j]
      dual = Math.max(dual, -lam[j])
      comp = Math.max(comp, Math.abs(lam[j] * (xs[j] - lo)))
    }
    if (Number.isFinite(hi)) {
      primal = Math.max(primal, xs[j] - hi)
      dualObjective += hi * mu[j]
      dual = Math.max(dual, mu[j])
      comp = Math.max(comp, Math.abs(mu[j] * (hi - xs[j])))
    }
    // Stationarity: the reduced cost must be carried by the bound duals (a free variable's must be zero).
    dual = Math.max(dual, Math.abs(reduced[j] - (Number.isFinite(lo) ? lam[j] : 0) - (Number.isFinite(hi) ? mu[j] : 0)))
  }
  const objective = dense.dot(lp.c, xs)
  return {
    x: vector(xs),
    objective,
    duals,
    reducedCosts: vector(reduced),
    slack: vector(slack),
    residualEq: vector(residualEq),
    dualObjective,
    dualityGap: objective - dualObjective,
    primalInfeasibility: primal,
    dualInfeasibility: dual,
    complementarity: comp,
    active: intTensor(Array.from(slack, (v) => (Math.abs(v) <= tolerance ? 1 : 0))),
  }
}
