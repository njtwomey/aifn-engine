/**
 * Linear model predictive control (MPC): at each sample, solve a finite-horizon quadratic program for the input
 * sequence from the measured state, apply its first input, and repeat (the receding horizon). Rawlings, Mayne & Diehl
 * (2017), "Model Predictive Control: Theory, Computation, and Design", 2nd ed., §1.3 (the linear-quadratic problem as
 * a QP) and §2.4 (the terminal cost from the DARE, which makes unconstrained MPC equal to LQR).
 */

import { boxQuadprog, quadprog } from 'aifn-compute/optim/programming'
import { dense, fromData, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dlqr, type StateFeedbackPlant } from './lqr'

type F64 = dense.F64

/** A linear MPC problem on the discrete plant x_{k+1} = Ax_k + Bu_k. */
export type MpcProblem = StateFeedbackPlant & {
  /** State weight Q (n × n, symmetric positive semi-definite). */
  Q: MatrixLike
  /** Input weight R (m × m, symmetric positive definite). */
  R: MatrixLike
  /** Terminal weight P; default the stabilising DARE solution, the LQR cost-to-go. */
  P?: MatrixLike
  /** Prediction horizon N ≥ 1 (steps). */
  horizon: Size
  /** Input bounds, per input or one number for all (±Infinity for none). */
  uMin?: VectorLike | Scalar
  uMax?: VectorLike | Scalar
  /** State bounds on x₁ … x_N, per state or one number (±Infinity for none). */
  xMin?: VectorLike | Scalar
  xMax?: VectorLike | Scalar
  /**
   * Make the state bounds soft: a slack s ≥ 0 per state and step widens them, xMin − s ≤ x ≤ xMax + s, at a cost
   * `quadratic`·‖s‖² + `linear`·1ᵀs added to the objective (defaults 1e3 and 1e2). The plan is then always feasible.
   * With `linear` above the hard problem's largest bound multiplier the penalty is exact: the soft plan equals the hard
   * one whenever the hard one exists (Kerrigan & Maciejowski, 2000).
   */
  soft?: { quadratic?: Scalar; linear?: Scalar }
}

/** The solution of one MPC problem from a state x₀. */
export type MpcPlan = {
  /** The optimal inputs u₀ … u_{N−1}, N × m. */
  u: Matrix
  /** The predicted states x₀ … x_N, (N + 1) × n. */
  x: Matrix
  /** The optimal cost Σ (x−r)ᵀQ(x−r) + uᵀRu + terminal term. */
  cost: Scalar
  /** The QP's status. */
  status: string
  /** Box constraints active at the solution (inputs at a bound), counted. */
  active: Size
  /** With soft state bounds: the slacks s₁ … s_N, N × n (how far each predicted state leaves its bounds). */
  slack?: Matrix
  /** With soft state bounds: the penalty paid for the slacks (not included in `cost`). */
  penalty?: Scalar
}

/** A linear MPC controller: the condensed QP's matrices, built once, and `plan(x0, reference?)`. */
export type MpcController = {
  readonly n: Size
  readonly m: Size
  readonly horizon: Size
  /** The terminal weight used. */
  readonly P: Matrix
  /** The model the controller predicts with (B as n × m), the default plant of `recedingHorizon`. */
  readonly model: { A: Matrix; B: Matrix }
  /** Solve from x₀, regulating to the state `reference` (default 0; it should be an equilibrium with u = 0). */
  plan(x0: VectorLike, reference?: VectorLike): MpcPlan
}

function bound(v: VectorLike | Scalar | undefined, k: number, fill: number, where: string): F64 {
  if (v === undefined) return new Float64Array(k).fill(fill)
  if (typeof v === 'number') return new Float64Array(k).fill(v)
  const a = dense.toF64(v, where)
  if (a.length !== k) throw new ShapeError('mpc', `${where} needs ${k} values`)
  return a
}

/**
 * A linear MPC controller (Rawlings, Mayne & Diehl, 2017, §1.3). The predicted states are affine in the stacked
 * inputs U = (u₀, …, u_{N−1}): X = Φx₀ + ΓU with Φ = (A, A², …, Aᴺ) and Γ the block lower-triangular matrix of
 * A^{i−j−1}B. Substituting gives the condensed QP min ½UᵀHU + fᵀU with H = 2(ΓᵀQ̄Γ + R̄) and f = 2ΓᵀQ̄(Φx₀ − r̄),
 * Q̄ = diag(Q, …, Q, P). Input bounds alone are a box QP (`boxQuadprog`); state bounds add the rows ±ΓU ≤ … of a
 * general QP (`quadprog`); when the state bounds cannot be met from x₀, the plan keeps only the input bounds and its
 * status says so. With `soft`, the state bounds take slacks s (one per state and step) and the QP is over (U, s) with
 * the penalty ρ‖s‖² + λ1ᵀs; it always has a solution. With the DARE terminal weight and no active constraints the first
 * input equals −Kx₀ of LQR for every horizon.
 */
export function mpcController(problem: MpcProblem): MpcController {
  const a = dense.toMatrixF64(problem.A, 'mpc A')
  const n = a.m
  if (a.n !== n) throw new ShapeError('mpc', 'mpc: A must be square')
  const bShape = (problem.B as { shape?: readonly number[] }).shape
  const bNested = !bShape && typeof (problem.B as ArrayLike<unknown>)[0] === 'object'
  const b =
    (bShape && bShape.length === 2) || bNested
      ? dense.toMatrixF64(problem.B, 'mpc B', n)
      : { data: dense.toF64(problem.B as unknown as VectorLike, 'mpc B'), m: n, n: 1 }
  const m = b.n
  const N = problem.horizon
  if (!(Number.isInteger(N) && N >= 1)) throw new DomainError('mpc', 'mpc: the horizon must be a positive integer')
  const Q = dense.toMatrixF64(problem.Q, 'mpc Q', n, n).data
  const R = dense.toMatrixF64(problem.R, 'mpc R', m, m).data
  const plant = { A: fromData(a.data, [n, n]), B: fromData(b.data, [n, m]) }
  const P = problem.P
    ? dense.toMatrixF64(problem.P, 'mpc P', n, n).data
    : dense.data(dlqr(plant, fromData(Q, [n, n]), fromData(R, [m, m])).P)
  // Φ ((N·n) × n) and Γ ((N·n) × (N·m)): row block i is x_{i+1}.
  const Nn = N * n
  const Nm = N * m
  const Phi = new Float64Array(Nn * n)
  const Gam = new Float64Array(Nn * Nm)
  const powers: F64[] = [dense.identity(n)]
  for (let i = 1; i <= N; i++) powers.push(dense.matMul(a.data, powers[i - 1], n, n, n))
  for (let i = 0; i < N; i++) {
    Phi.set(powers[i + 1], i * n * n)
    for (let j = 0; j <= i; j++) {
      const blk = dense.matMul(powers[i - j], b.data, n, n, m)
      for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) Gam[(i * n + r) * Nm + j * m + c] = blk[r * m + c]
    }
  }
  // Q̄ Γ, with Q̄ = diag(Q, …, Q, P).
  const QbarGam = new Float64Array(Nn * Nm)
  for (let i = 0; i < N; i++) {
    const W = i === N - 1 ? P : Q
    const rows = Gam.subarray(i * n * Nm, (i + 1) * n * Nm)
    QbarGam.set(dense.matMul(W, rows, n, n, Nm), i * n * Nm)
  }
  const H = dense.matMul(dense.transpose(Gam, Nn, Nm), QbarGam, Nm, Nn, Nm)
  for (let i = 0; i < N; i++)
    for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) H[(i * m + r) * Nm + i * m + c] += R[r * m + c]
  for (let k = 0; k < H.length; k++) H[k] *= 2
  const Hs = dense.symmetrise(H, Nm)
  const uLo = bound(problem.uMin, m, -Infinity, 'mpc uMin')
  const uHi = bound(problem.uMax, m, Infinity, 'mpc uMax')
  const xLo = bound(problem.xMin, n, -Infinity, 'mpc xMin')
  const xHi = bound(problem.xMax, n, Infinity, 'mpc xMax')
  const stateBounded = [...xLo, ...xHi].some((v) => Number.isFinite(v))
  const soft = stateBounded && problem.soft ? problem.soft : null
  const rho = soft?.quadratic ?? 1e3
  const lambda = soft?.linear ?? 1e2
  if (soft && !(rho > 0 && lambda >= 0))
    throw new DomainError('mpc', 'mpc: soft needs a positive quadratic and a non-negative linear weight')
  const lower = Float64Array.from({ length: Nm }, (_, k) => uLo[k % m])
  const upper = Float64Array.from({ length: Nm }, (_, k) => uHi[k % m])
  const QbarGamT = dense.transpose(QbarGam, Nn, Nm)

  const plan = (x0v: VectorLike, reference?: VectorLike): MpcPlan => {
    const x0 = dense.toF64(x0v, 'mpc x0')
    if (x0.length !== n) throw new ShapeError('mpc', `mpc: x0 needs ${n} values`)
    const r = reference === undefined ? new Float64Array(n) : dense.toF64(reference, 'mpc reference')
    const free = dense.matVec(Phi, x0, Nn, n)
    const dev = Float64Array.from(free, (v, k) => v - r[k % n])
    const f = dense.scale(2, dense.matVec(QbarGamT, dev, Nm, Nn))
    let U: F64
    let status: string
    let slack: F64 | null = null
    let penalty = 0
    const boxed = () => {
      const s = boxQuadprog({ Q: fromData(Hs, [Nm, Nm]), c: f, lower, upper }, { tolerance: 1e-10, maxSteps: 2000 })
      return { U: dense.data(s.x), status: s.converged ? 'optimal' : 'limit' }
    }
    if (!stateBounded) {
      ;({ U, status } = boxed())
    } else {
      // Variables z = (U, s), s empty unless soft. Rows: u ≤ uHi, −u ≤ −uLo, Γu − s ≤ xHi − Φx₀,
      // −Γu − s ≤ Φx₀ − xLo (finite bounds only), and −s ≤ 0.
      const Ns = soft ? Nn : 0
      const Nz = Nm + Ns
      const rowsA: number[][] = []
      const rhs: number[] = []
      for (let k = 0; k < Nm; k++) {
        if (Number.isFinite(upper[k])) {
          const row = new Array<number>(Nz).fill(0)
          row[k] = 1
          rowsA.push(row)
          rhs.push(upper[k])
        }
        if (Number.isFinite(lower[k])) {
          const row = new Array<number>(Nz).fill(0)
          row[k] = -1
          rowsA.push(row)
          rhs.push(-lower[k])
        }
      }
      for (let k = 0; k < Nn; k++) {
        const g = Array.from(Gam.subarray(k * Nm, (k + 1) * Nm))
        const pad = (row: number[]) => {
          const out = row.concat(new Array<number>(Ns).fill(0))
          if (soft) out[Nm + k] = -1
          return out
        }
        if (Number.isFinite(xHi[k % n])) {
          rowsA.push(pad(g))
          rhs.push(xHi[k % n] - free[k])
        }
        if (Number.isFinite(xLo[k % n])) {
          rowsA.push(pad(g.map((v) => -v)))
          rhs.push(free[k] - xLo[k % n])
        }
        if (soft) {
          const row = new Array<number>(Nz).fill(0)
          row[Nm + k] = -1
          rowsA.push(row)
          rhs.push(0)
        }
      }
      let Hz = Hs
      let fz = f
      if (soft) {
        Hz = new Float64Array(Nz * Nz)
        for (let i = 0; i < Nm; i++) Hz.set(Hs.subarray(i * Nm, (i + 1) * Nm), i * Nz)
        for (let k = 0; k < Ns; k++) Hz[(Nm + k) * Nz + Nm + k] = 2 * rho
        fz = new Float64Array(Nz)
        fz.set(f)
        fz.fill(lambda, Nm)
      }
      const s = quadprog(
        { Q: fromData(Hz, [Nz, Nz]), c: fz, A: rowsA, b: rhs },
        { method: 'interior-point', tolerance: 1e-9 },
      )
      const z = dense.data(s.x)
      U = z.slice(0, Nm)
      if (soft) {
        slack = Float64Array.from(z.subarray(Nm), (v) => Math.max(0, v))
        penalty = slack.reduce((acc, v) => acc + rho * v * v + lambda * v, 0)
      }
      status = s.status
      if (s.status !== 'optimal') {
        // No input sequence keeps the states in bounds from here: fall back to the input bounds alone and say so.
        ;({ U } = boxed())
        status = `${s.status}: state bounds dropped`
        slack = null
      }
    }
    const X = dense.add(free, dense.matVec(Gam, U, Nn, Nm))
    const states = new Float64Array((N + 1) * n)
    states.set(x0, 0)
    states.set(X, n)
    let cost = 0
    for (let i = 0; i < N; i++) {
      const W = i === N - 1 ? P : Q
      const e = Float64Array.from(X.subarray(i * n, (i + 1) * n), (v, k) => v - r[k])
      cost += dense.dot(e, dense.matVec(W, e, n, n))
      const ui = U.subarray(i * m, (i + 1) * m)
      cost += dense.dot(ui, dense.matVec(R, ui, m, m))
    }
    const e0 = Float64Array.from(x0, (v, k) => v - r[k])
    cost += dense.dot(e0, dense.matVec(Q, e0, n, n))
    let active = 0
    for (let k = 0; k < Nm; k++) if (Math.abs(U[k] - upper[k]) < 1e-7 || Math.abs(U[k] - lower[k]) < 1e-7) active++
    const out: MpcPlan = { u: fromData(U, [N, m]), x: fromData(states, [N + 1, n]), cost, status, active }
    return slack ? { ...out, slack: fromData(slack, [N, n]), penalty } : out
  }
  return { n, m, horizon: N, P: fromData(P, [n, n]), model: plant, plan }
}

/** The state of `recedingHorizon`. */
export interface RecedingHorizonState extends Status {
  /** Samples applied so far. */
  t: Size
  /** The plant state now. */
  x: Vector
  /** The input applied from now to the next sample: the plan's first input. */
  u: Vector
  /** The plan made now: predicted states (N + 1) × n and inputs N × m. */
  predictedX: Matrix
  predictedU: Matrix
  /** The plan's cost. */
  cost: Scalar
  /** The QP's status at this sample. */
  status: string
  terminated: boolean
  diverged: boolean
}

/** Options for `recedingHorizon`. */
export type RecedingHorizonOptions = {
  /** The true plant, if it differs from the controller's model (model mismatch). Default `controller.model`. */
  plant?: StateFeedbackPlant
  /** The reference state to regulate to. Default 0. */
  reference?: VectorLike
  /** An additive disturbance on the state at each step, w_k (a function of the step). */
  disturbance?: (t: Size) => VectorLike
  /** Stop after this many samples. */
  steps?: Size
}

/**
 * Receding-horizon control as a traceable algorithm: at each sample the controller plans N steps ahead from the
 * measured state, the plant moves under the plan's first input, and the rest of the plan is discarded. `init` takes
 * `{ x0 }`. The state records both the plan (predicted) and what happened (applied), so a view can draw them together.
 */
export function recedingHorizon(
  controller: MpcController,
  options: RecedingHorizonOptions = {},
): Algorithm<{ x0: VectorLike }, RecedingHorizonState> {
  const { n, m } = controller
  const plant = options.plant ?? controller.model
  const A = dense.toMatrixF64(plant.A, 'recedingHorizon A', n, n).data
  const bt = (plant.B as { shape?: readonly number[] }).shape
  const B =
    (bt && bt.length === 2) || (!bt && typeof (plant.B as ArrayLike<unknown>)[0] === 'object')
      ? dense.toMatrixF64(plant.B, 'recedingHorizon B', n, m).data
      : dense.toF64(plant.B as unknown as VectorLike, 'recedingHorizon B')
  const make = (t: number, x: F64): RecedingHorizonState => {
    const p = controller.plan(x, options.reference)
    const u = toFlat(p.u).slice(0, m)
    return {
      t,
      x: fromData(Float64Array.from(x), [n]),
      u: fromData(Float64Array.from(u), [m]),
      predictedX: p.x,
      predictedU: p.u,
      cost: p.cost,
      status: p.status,
      terminated: options.steps !== undefined && t >= options.steps,
      diverged: !dense.allFinite(x),
    }
  }
  return {
    name: 'receding-horizon',
    init: ({ x0 }) => make(0, dense.toF64(x0, 'recedingHorizon x0')),
    step: (s) => {
      const x = dense.add(dense.matVec(A, dense.data(s.x), n, n), dense.matVec(B, dense.data(s.u), n, m))
      if (options.disturbance) {
        const w = dense.toF64(options.disturbance(s.t), 'recedingHorizon disturbance')
        for (let k = 0; k < n; k++) x[k] += w[k]
      }
      return make(s.t + 1, x)
    },
  }
}
