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

/**
 * A linear MPC problem on the discrete plant $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k$ (`A` $n \times n$; `B`
 * $n \times m$, or a vector of $n$ for a single input).
 */
export type MpcProblem = StateFeedbackPlant & {
  /** State weight $\Qmat$ ($n \times n$, symmetric positive semi-definite). */
  Q: MatrixLike
  /** Input weight $\Rmat$ ($m \times m$, symmetric positive definite). */
  R: MatrixLike
  /** Terminal weight $\Pmat$ ($n \times n$); default the stabilising DARE solution, the LQR cost-to-go. */
  P?: MatrixLike
  /** Prediction horizon $N \ge 1$ (steps). */
  horizon: Size
  /** Lower input bounds, per input or one number for all (`-Infinity` for none, the default). */
  uMin?: VectorLike | Scalar
  /** Upper input bounds, per input or one number for all (`Infinity` for none, the default). */
  uMax?: VectorLike | Scalar
  /** Lower state bounds on $\xvec_1, \dots, \xvec_N$, per state or one number (`-Infinity` for none, the default). */
  xMin?: VectorLike | Scalar
  /** Upper state bounds on $\xvec_1, \dots, \xvec_N$, per state or one number (`Infinity` for none, the default). */
  xMax?: VectorLike | Scalar
  /**
   * Make the state bounds soft: a slack $\svec \ge 0$ per state and step widens them,
   * $\xvec_{\min} - \svec \le \xvec \le \xvec_{\max} + \svec$, at a cost
   * $\rho \lVert \svec \rVert^2 + \lambda \ones^\top \svec$ added to the objective, with $\rho$ = `quadratic` and
   * $\lambda$ = `linear` (defaults $10^3$ and $10^2$). The plan is then always feasible. With $\lambda$ above the hard
   * problem's largest bound multiplier the penalty is exact: the soft plan equals the hard one whenever the hard one
   * exists (Kerrigan & Maciejowski, 2000). Ignored when no state bound is finite.
   */
  soft?: { quadratic?: Scalar; linear?: Scalar }
}

/** The solution of one MPC problem from a state $\xvec_0$. */
export type MpcPlan = {
  /** The optimal inputs $\uvec_0, \dots, \uvec_{N-1}$, $N \times m$. */
  u: Matrix
  /** The predicted states $\xvec_0, \dots, \xvec_N$, $(N + 1) \times n$. */
  x: Matrix
  /**
   * The optimal cost
   * $\sum_{k=0}^{N-1} (\evec_k^\top\Qmat\evec_k + \uvec_k^\top\Rmat\uvec_k) + \evec_N^\top\Pmat\evec_N$, with
   * $\evec_k = \xvec_k - \rvec$ the deviation from the reference.
   */
  cost: Scalar
  /**
   * The QP's status: `'optimal'`, `'limit'` (input bounds only, step limit reached), or the general QP's status, with
   * ": state bounds dropped" appended when the state bounds could not be met and only the input bounds were kept.
   */
  status: string
  /** How many entries of the planned inputs sit at one of their bounds. */
  active: Size
  /**
   * With soft state bounds: the slacks $\svec_1, \dots, \svec_N$, $N \times n$ (how far each predicted state leaves its
   * bounds).
   */
  slack?: Matrix
  /** With soft state bounds: the penalty paid for the slacks (not included in `cost`). */
  penalty?: Scalar
}

/** A linear MPC controller: the condensed QP's matrices, built once, and `plan(x0, reference?)`. */
export type MpcController = {
  /** The number of states $n$. */
  readonly n: Size
  /** The number of inputs $m$. */
  readonly m: Size
  /** The prediction horizon $N$. */
  readonly horizon: Size
  /** The terminal weight $\Pmat$ used ($n \times n$). */
  readonly P: Matrix
  /** The model the controller predicts with (`B` as $n \times m$), the default plant of `recedingHorizon`. */
  readonly model: { A: Matrix; B: Matrix }
  /**
   * Solve from $\xvec_0$, regulating to the state `reference` (default $\zeros$; it should be an equilibrium with
   * $\uvec = \zeros$).
   */
  plan(x0: VectorLike, reference?: VectorLike): MpcPlan
}

/**
 * Expand an optional bound to one value per entry.
 *
 * @param v The bound as given: undefined, one number for every entry, or one value per entry.
 * @param k The number of entries (states or inputs).
 * @param fill The value used when `v` is undefined (`-Infinity` or `Infinity`, no bound).
 * @param where The caller's name for the bound, used in error messages.
 * @returns A new array of `k` bounds. A vector of the wrong length throws `ShapeError`.
 */
function bound(v: VectorLike | Scalar | undefined, k: number, fill: number, where: string): F64 {
  if (v === undefined) return new Float64Array(k).fill(fill)
  if (typeof v === 'number') return new Float64Array(k).fill(v)
  const a = dense.toF64(v, where)
  if (a.length !== k) throw new ShapeError('mpc', `${where} needs ${k} values`)
  return a
}

/**
 * A linear MPC controller (Rawlings, Mayne & Diehl, 2017, §1.3). The predicted states are affine in the stacked
 * inputs $\bar{\uvec} = (\uvec_0, \dots, \uvec_{N-1})$: $\bar{\xvec} = \Phimat\xvec_0 + \Gammamat\bar{\uvec}$ with
 * $\Phimat = (\Amat, \Amat^2, \dots, \Amat^N)$ and $\Gammamat$ the block lower-triangular matrix of blocks
 * $\Amat^{i-j-1}\Bmat$. Substituting gives the condensed QP
 * $\min \tfrac{1}{2}\bar{\uvec}^\top\Hmat\bar{\uvec} + \fvec^\top\bar{\uvec}$ with
 * $\Hmat = 2(\Gammamat^\top\bar{\Qmat}\Gammamat + \bar{\Rmat})$ and
 * $\fvec = 2\Gammamat^\top\bar{\Qmat}(\Phimat\xvec_0 - \bar{\rvec})$, where
 * $\bar{\Qmat} = \diag(\Qmat, \dots, \Qmat, \Pmat)$ and $\bar{\Rmat} = \diag(\Rmat, \dots, \Rmat)$. Input bounds alone
 * are a box QP (`boxQuadprog`); state bounds add the rows $\pm\Gammamat\bar{\uvec} \le \dots$ of a general QP
 * (`quadprog`); when the state bounds cannot be met from $\xvec_0$, the plan keeps only the input bounds and its status
 * says so. With `soft`, the state bounds take slacks $\svec$ (one per state and step) and the QP is over
 * $(\bar{\uvec}, \svec)$ with the penalty $\rho\lVert \svec \rVert^2 + \lambda\ones^\top\svec$; it always has a
 * solution. With the DARE terminal weight and no active constraints the first input equals $-\Kmat\xvec_0$ of LQR for
 * every horizon.
 *
 * @param problem The plant, weights, horizon and bounds. A non-square `A`, weights of the wrong shape or a vector
 *   bound of the wrong length throw `ShapeError`; a horizon that is not a positive integer, or `soft` weights that are
 *   not positive (quadratic) and non-negative (linear), throw `DomainError`.
 * @returns The controller: its dimensions, terminal weight and model, and `plan`, which solves the QP from a state.
 *   Planning from an $\xvec_0$ of the wrong length throws `ShapeError`.
 *
 * @example Unconstrained MPC is LQR
 * // x_{k+1} = x_k + u_k, Q = R = 1: the LQR gain is 0.618 (dlqr's example), so u₀ = −0.618 · 5.
 * const mpc = mpcController({ A: [[1]], B: [[1]], Q: [[1]], R: [[1]], horizon: 3 })
 * const plan = mpc.plan([5])
 * print('inputs =', plan.u)
 * print('predicted states =', plan.x)
 * print('cost =', plan.cost, ' status:', plan.status)
 *
 * @example An input bound saturates the plan
 * const mpc = mpcController({ A: [[1]], B: [[1]], Q: [[1]], R: [[1]], horizon: 3, uMin: -1, uMax: 1 })
 * const plan = mpc.plan([5])
 * print('inputs =', plan.u)
 * print('predicted states =', plan.x)
 * print('inputs at a bound:', plan.active)
 *
 * @example A state bound that cannot be met, kept soft
 * // From x₀ = 5 with |u| ≤ 1 the state cannot be brought below 3 in one step: the slack says by how much it misses.
 * const mpc = mpcController({
 *   A: [[1]], B: [[1]], Q: [[1]], R: [[1]], horizon: 2, uMin: -1, uMax: 1, xMax: 3, soft: {},
 * })
 * const plan = mpc.plan([5])
 * print('predicted states =', plan.x)
 * print('slack =', plan.slack)
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
  /** The plan made now: its predicted states, $(N + 1) \times n$. */
  predictedX: Matrix
  /** The plan made now: its inputs, $N \times m$. */
  predictedU: Matrix
  /** The plan's cost. */
  cost: Scalar
  /** The QP's status at this sample. */
  status: string
  /** True once `steps` samples have been applied (never when `steps` is not given). */
  terminated: boolean
  /** True when the plant state is not finite. */
  diverged: boolean
}

/** Options for `recedingHorizon`. */
export type RecedingHorizonOptions = {
  /** The true plant, if it differs from the controller's model (model mismatch). Default `controller.model`. */
  plant?: StateFeedbackPlant
  /** The reference state to regulate to. Default $\zeros$. */
  reference?: VectorLike
  /** An additive disturbance $\wvec_k$ on the state at each step, as a function of the step $k$. */
  disturbance?: (t: Size) => VectorLike
  /** Stop after this many samples. */
  steps?: Size
}

/**
 * Receding-horizon control as a traceable algorithm: at each sample the controller plans $N$ steps ahead from the
 * measured state, the plant moves under the plan's first input, $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k + \wvec_k$,
 * and the rest of the plan is discarded. `init` takes `{ x0 }`. The state records both the plan (predicted) and what
 * happened (applied), so a view can draw them together.
 *
 * @param controller The controller that plans at every sample, as `mpcController` returns it.
 * @param options The true plant (default the controller's model), the reference, a disturbance and the number of
 *   samples after which the run stops.
 * @returns The algorithm: `init` plans from `x0`, and each `step` applies the first input and plans again.
 *
 * @example Five samples of bounded control
 * const mpc = mpcController({ A: [[1]], B: [[1]], Q: [[1]], R: [[1]], horizon: 3, uMin: -1, uMax: 1 })
 * const loop = recedingHorizon(mpc, { steps: 5 })
 * const end = run(loop, { x0: [5] }, 10)
 * print('samples applied =', end.t)
 * print('state now =', end.x)
 * print('next input =', end.u)
 *
 * @example A model that is wrong
 * // The controller believes x_{k+1} = x_k + u_k, the plant is x_{k+1} = 1.2 x_k + u_k: feedback still regulates it.
 * const mpc = mpcController({ A: [[1]], B: [[1]], Q: [[1]], R: [[1]], horizon: 5 })
 * const loop = recedingHorizon(mpc, { plant: { A: [[1.2]], B: [[1]] } })
 * print('after 10 samples x =', run(loop, { x0: [5] }, 10).x)
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
