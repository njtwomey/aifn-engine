/**
 * Linear programs: the problem type, its conversion to standard form, and duality reporting.
 *
 * The problem follows `scipy.optimize.linprog`: minimise cᵀx subject to A_ub x ≤ b_ub, A_eq x = b_eq and
 * lᵢ ≤ xᵢ ≤ uᵢ, with default bounds 0 ≤ x < ∞. Dual values follow scipy's HiGHS convention: each is the sensitivity of
 * the optimal objective to its right-hand side (∂f⋆/∂b), so the duals of ≤ rows and of upper bounds are ≤ 0, and the
 * duals of lower bounds are ≥ 0 (Vanderbei, 2020, "Linear Programming", ch. 5 and ch. 7).
 */

import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Index, MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { checkFinite, intTensor, matTVec, matVec, readMatrix, readVector, vector, type Mat } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A bound on one variable: `[lower, upper]`; `null` or ±`Infinity` means no bound on that side. */
export type Bound = readonly [Scalar | null, Scalar | null]

/**
 * A linear program in the form of `scipy.optimize.linprog`: minimise cᵀx subject to A_ub x ≤ b_ub, A_eq x = b_eq and
 * `bounds`. To maximise, negate `c` (and the optimal objective). A ≥ row is a ≤ row with both sides negated.
 */
export interface LinearProgram {
  /** Objective coefficients, length n. */
  c: VectorLike
  /** Inequality constraint matrix, m_ub × n (rows of A_ub x ≤ b_ub). */
  A_ub?: MatrixLike
  /** Inequality right-hand side, length m_ub. */
  b_ub?: VectorLike
  /** Equality constraint matrix, m_eq × n. */
  A_eq?: MatrixLike
  /** Equality right-hand side, length m_eq. */
  b_eq?: VectorLike
  /** One bound for every variable, or a list of n bounds. Default `[0, null]` (non-negative), as in scipy. */
  bounds?: Bound | readonly Bound[]
}

/** A linear program read into dense arrays; infinite bounds are ±Infinity. */
export interface ParsedLP {
  n: Size
  c: Float64Array
  Aub: Mat
  bub: Float64Array
  Aeq: Mat
  beq: Float64Array
  lower: Float64Array
  upper: Float64Array
}

/** Read and validate a linear program. */
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

/** How one original variable is written in standard-form columns. */
export type VariableMap =
  /** x = lower + z[column]; with a finite upper bound, the row `boundRow` holds z + t = upper − lower. */
  | { kind: 'lower'; column: Index; boundRow: Index }
  /** x = upper − z[column] (no lower bound). */
  | { kind: 'upper'; column: Index }
  /** x = z[plus] − z[minus] (free). */
  | { kind: 'free'; plus: Index; minus: Index }

/** Where a standard-form row came from, and whether it was negated to make its right-hand side non-negative. */
export type RowSource = { kind: 'ub' | 'eq' | 'bound'; source: Index; sign: 1 | -1 }

/**
 * A linear program in standard form: minimise cᵀz + offset subject to A z = b, z ≥ 0, with b ≥ 0.
 *
 * Variables are shifted and split so that every column is non-negative (x = l + z, x = u − z, or x = z⁺ − z⁻), every ≤
 * row and every finite upper bound gets a slack column, and rows with a negative right-hand side are negated.
 */
export interface StandardForm {
  /** Rows. */
  m: Size
  /** Columns. */
  N: Size
  /** Constraint matrix, m × N, row-major. */
  A: Float64Array
  b: Float64Array
  c: Float64Array
  /** The constant cᵀ(shift) dropped from the objective. */
  offset: Scalar
  /** A readable label for each column: `x1`, `x1+`, `x1-`, `s1` (slack of ≤ row 1), `t1` (slack of x1's upper bound). */
  labels: string[]
  /** For each row, the column of a slack with coefficient +1 in it (a ready basic column), or −1. */
  slackOfRow: Int32Array
  rows: RowSource[]
  variables: VariableMap[]
  /** The original problem. */
  lp: ParsedLP
}

/** Convert a parsed linear program to standard form. */
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

/** The original variables x from standard-form values z (with `shift: false`, only the linear part, for directions). */
export function toOriginal(sf: StandardForm, z: ArrayLike<number>, shift = true): Float64Array {
  const x = new Float64Array(sf.lp.n)
  sf.variables.forEach((v, j) => {
    if (v.kind === 'lower') x[j] = (shift ? sf.lp.lower[j] : 0) + z[v.column]
    else if (v.kind === 'upper') x[j] = (shift ? sf.lp.upper[j] : 0) - z[v.column]
    else x[j] = z[v.plus] - z[v.minus]
  })
  return x
}

/** Duals of a linear program in scipy's convention (sensitivities ∂f⋆/∂b of the optimal objective). */
export interface LinearProgramDuals {
  /** Duals of the ≤ rows, length m_ub; ≤ 0 at an optimum. */
  ineq: Tensor
  /** Duals of the equality rows, length m_eq; any sign. */
  eq: Tensor
  /** Duals of the lower bounds, length n; ≥ 0 at an optimum, 0 where there is no lower bound. */
  lower: Tensor
  /** Duals of the upper bounds, length n; ≤ 0 at an optimum, 0 where there is no upper bound. */
  upper: Tensor
}

/**
 * Map standard-form row duals y (length m, for the rows as stored) to the original problem's duals. Rows absent from
 * `y` (e.g. removed as redundant) take 0 when `rowsKept` lists which rows y covers.
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
  /** The point, length n. */
  x: Tensor
  /** cᵀx. */
  objective: Scalar
  duals: LinearProgramDuals
  /** Reduced costs c − A_ubᵀy_ub − A_eqᵀy_eq, length n (the sum of the lower- and upper-bound duals at an optimum). */
  reducedCosts: Tensor
  /** Slacks b_ub − A_ub x of the ≤ rows, length m_ub (≥ 0 when feasible). */
  slack: Tensor
  /** Residuals b_eq − A_eq x, length m_eq. */
  residualEq: Tensor
  /** The dual objective b_ubᵀy_ub + b_eqᵀy_eq + Σ lᵢ λᵢ + Σ uᵢ μᵢ over finite bounds. */
  dualObjective: Scalar
  /** objective − dualObjective; zero at an optimum (strong duality). */
  dualityGap: Scalar
  /** Largest violation of a primal constraint or bound (0 when feasible). */
  primalInfeasibility: Scalar
  /** Largest violation of a dual sign condition, or of dual stationarity for a free variable (0 when dual feasible). */
  dualInfeasibility: Scalar
  /**
   * Largest complementary-slackness product: |yᵢ · slackᵢ| over ≤ rows, and |λⱼ (xⱼ − lⱼ)|, |μⱼ (uⱼ − xⱼ)| over
   * finite bounds. Zero at an optimum.
   */
  complementarity: Scalar
  /** 1 for each ≤ row whose slack is at most `tolerance` (the active, or binding, constraints), else 0; int32. */
  active: Tensor
}

/**
 * The duality report of a point `x` and duals of a linear program: slacks, reduced costs, the dual objective, the
 * duality gap and the complementary-slackness, primal- and dual-feasibility residuals that certify optimality
 * (Vanderbei, 2020, Theorem 5.3). `tolerance` decides which rows are reported `active`.
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
