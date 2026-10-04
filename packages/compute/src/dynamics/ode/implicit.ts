/**
 * Implicit solvers for stiff systems: implicit (backward) Euler, the implicit trapezoid rule and the backward differentiation
 * formulae BDF1–3 (Hairer & Wanner, 1996, "Solving Ordinary Differential Equations II", §IV.1 and §III.1; Ascher &
 * Petzold, 1998, §5.1). Each step solves its nonlinear equation by Newton's method (`aifn-compute/numerics/roots`), with the Jacobian
 * ∂f/∂x from `aifn-compute/foundation/autodiff` (or forward differences, or a given function).
 */

import { jacobian as autodiffJacobian } from 'aifn-compute/foundation/autodiff'
import { newtonSystem } from 'aifn-compute/numerics/roots'
import { dense, fromData, SQRT_EPS, type Tensor, toFlat, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { Algorithm, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { checkDirection, evaluate, initialState, nextStep, reached } from './explicit'
import type { FixedStepOptions, InitialValue, JacobianOption, OdeState, Rhs } from './types'
import { DomainError } from 'aifn-compute/foundation/errors'

const { allFinite, toF64, toMatrixF64 } = dense
type F64 = dense.F64

/**
 * ∂f/∂x at (t, x) as a row-major n×n array. `'autodiff'` differentiates f with a traced x; `'finite-difference'`
 * uses forward differences with h = √ε·max(1, |x_j|) (n extra evaluations, counted in `evaluations`). Internal to
 * the implicit solvers.
 */
export function jacobianOf(
  f: Rhs,
  option: JacobianOption,
  t: Scalar,
  x: F64,
  where: string,
): { J: F64; evaluations: Size } {
  const n = x.length
  if (typeof option === 'function')
    return { J: toMatrixF64(option(t, fromData(x, [n])), where, n, n).data, evaluations: 0 }
  if (option === 'autodiff') {
    const J = autodiffJacobian((y: Value) => f(t, y as Tensor) as Value)(fromData(Float64Array.from(x), [n]))
    return { J: Float64Array.from(toFlat(J as Tensor)), evaluations: 0 }
  }
  const fx = evaluate(f, t, x, where)
  const J = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const h = SQRT_EPS * Math.max(1, Math.abs(x[j]))
    const xh = Float64Array.from(x)
    xh[j] += h
    const fh = evaluate(f, t, xh, where)
    for (let i = 0; i < n; i++) J[i * n + j] = (fh[i] - fx[i]) / h
  }
  return { J, evaluations: n + 1 }
}

/** Options for the implicit solvers. */
export type ImplicitOptions = FixedStepOptions & {
  /** How to compute ∂f/∂x. Default `'autodiff'`. */
  jacobian?: JacobianOption
  /** Newton stops when ‖G(y)‖₂ ≤ this. Default 1e-10. */
  newtonTolerance?: Scalar
  /** Most Newton iterations per step. Default 20. */
  maxNewtonSteps?: Size
}

/** The state of an implicit solver. */
export interface ImplicitState extends OdeState {
  /** Newton iterations used by the last step. */
  newtonSteps: Size
  /** Whether the last step's Newton iteration converged. */
  newtonConverged: boolean
  /** Earlier states x_{n−1}, x_{n−2} (newest first), kept by the multistep BDF methods. */
  history: Vector[]
}

/**
 * Solves the step equation y = c + γh f(t₁, y) (the form shared by every method here, with c collecting the known
 * terms) by Newton's method on G(y) = y − c − γh f(t₁, y), whose Jacobian is I − γh ∂f/∂x, from the guess y₀.
 */
function solveStep(
  f: Rhs,
  jac: JacobianOption,
  t1: number,
  c: F64,
  gammaH: number,
  guess: F64,
  tol: number,
  maxNewton: number,
  where: string,
) {
  const n = c.length
  let evaluations = 0
  let jacobians = 0
  const G = (y: Vector) => {
    const yv = dense.data(y)
    const fy = evaluate(f, t1, yv, where)
    const { J, evaluations: extra } = jacobianOf(f, jac, t1, yv, where)
    evaluations += 1 + extra
    jacobians++
    const value = new Float64Array(n)
    const jacobian = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      value[i] = yv[i] - c[i] - gammaH * fy[i]
      for (let j = 0; j < n; j++) jacobian[i * n + j] = (i === j ? 1 : 0) - gammaH * J[i * n + j]
    }
    return { value: fromData(value, [n]), jacobian: fromData(jacobian, [n, n]) }
  }
  const s = run(newtonSystem(G, { ftol: tol, xtol: 1e-15 }), { x0: guess }, maxNewton)
  return {
    y: Float64Array.from(toFlat(s.x)),
    iterations: s.t,
    converged: s.converged && s.failure === null,
    evaluations,
    jacobians,
  }
}

type Scheme = {
  name: string
  /** The known part c and the weight γ of the step equation y = c + γh f(t + h, y), given the history. */
  equation: (x: F64, history: F64[], fx: F64, h: number) => { c: F64; gamma: number }
  /** Whether the equation uses f(t, x) (the implicit trapezoid rule). */
  needsFx: boolean | ((historyLength: number) => boolean)
  /** How many past states to keep. */
  keep: number
  /** Whether a step must be a one-step start-up step (TR-BDF2) because the history is too short. */
  startup?: (historyLength: number) => boolean
}

// BDF coefficients: Σ_j α_j x_{n+1−j} = β h f(x_{n+1}), written as x_{n+1} = Σ_j a_j x_{n+1−j} + β h f(x_{n+1}).
// Orders 1–3 (Hairer & Wanner, 1996, §III.1, Table 1.1); the first steps use the lower orders their history allows.
const BDF: { a: number[]; beta: number }[] = [
  { a: [1], beta: 1 },
  { a: [4 / 3, -1 / 3], beta: 2 / 3 },
  { a: [18 / 11, -9 / 11, 2 / 11], beta: 6 / 11 },
]

function bdfScheme(order: 1 | 2 | 3): Scheme {
  return {
    name: `bdf${order}`,
    needsFx: false,
    keep: order - 1,
    startup: (historyLength) => historyLength < order - 1,
    equation: (x, history) => {
      const { a, beta } = BDF[order - 1]
      const k = order
      const past = [x, ...history]
      const c = new Float64Array(x.length)
      for (let j = 0; j < k; j++) for (let i = 0; i < x.length; i++) c[i] += a[j] * past[j][i]
      return { c, gamma: beta }
    },
  }
}

const SCHEMES = {
  'implicit-euler': { name: 'implicit-euler', needsFx: false, keep: 0, equation: (x: F64) => ({ c: x, gamma: 1 }) },
  'implicit-trapezoid': {
    name: 'implicit-trapezoid',
    needsFx: true,
    keep: 0,
    equation: (x: F64, _h: F64[], fx: F64, h: number) => ({
      c: Float64Array.from(x, (v, i) => v + 0.5 * h * fx[i]),
      gamma: 0.5,
    }),
  },
} satisfies Record<string, Scheme>

function implicitSolver(f: Rhs, scheme: Scheme, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  const {
    stepSize: h,
    tEnd,
    jacobian = 'autodiff',
    newtonTolerance: newtonTol = 1e-10,
    maxNewtonSteps: maxNewton = 20,
  } = options
  if (!(h !== 0 && Number.isFinite(h)))
    throw new DomainError(scheme.name, `${scheme.name}: the step size must be finite and non-zero`)
  const name = scheme.name
  return {
    name,
    init: ({ x0, t0 = 0 }) => {
      checkDirection(t0, h, tEnd, name)
      return { ...initialState(toF64(x0, name), t0), newtonSteps: 0, newtonConverged: true, history: [] }
    },
    step: (s) => {
      const x = dense.data(s.x)
      const hk = nextStep(s.time, h, tEnd)
      // The multistep formulae assume equal steps, so a shortened last step restarts them at order 1.
      const history = hk === h ? s.history.map((v) => dense.data(v)) : []
      const needsFx = typeof scheme.needsFx === 'function' ? scheme.needsFx(history.length) : scheme.needsFx
      let r: ReturnType<typeof solveStep>
      let extra = 0
      if (scheme.startup?.(history.length)) {
        // TR-BDF2 (Bank et al., 1985; Hosea & Shampine, 1996): a trapezoid stage to t + γh, then BDF2 through
        // (t, x), (t + γh, x_γ) to t + h, with γ = 2 − √2. It is L-stable (unlike the trapezoid rule, stiff modes are
        // damped) and its O(h³) local error keeps the starting values accurate enough for BDF2's and BDF3's order.
        const g = 2 - Math.SQRT2
        const fx0 = evaluate(f, s.time, x, name)
        const c1 = Float64Array.from(x, (v, i) => v + 0.5 * g * hk * fx0[i])
        const r1 = solveStep(f, jacobian, s.time + g * hk, c1, 0.5 * g * hk, x, newtonTol, maxNewton, name)
        const w = 1 / (g * (2 - g))
        const c2 = Float64Array.from(x, (v, i) => w * r1.y[i] - w * (1 - g) ** 2 * v)
        const r2 = solveStep(f, jacobian, s.time + hk, c2, ((1 - g) / (2 - g)) * hk, r1.y, newtonTol, maxNewton, name)
        r = {
          y: r2.y,
          iterations: r1.iterations + r2.iterations,
          converged: r1.converged && r2.converged,
          evaluations: r1.evaluations + r2.evaluations,
          jacobians: r1.jacobians + r2.jacobians,
        }
        extra = 1
      } else {
        const fx = needsFx ? evaluate(f, s.time, x, name) : x
        const { c, gamma } = scheme.equation(x, history, fx, hk)
        r = solveStep(f, jacobian, s.time + hk, c, gamma * hk, x, newtonTol, maxNewton, name)
        extra = needsFx ? 1 : 0
      }
      const finite = allFinite(r.y)
      const ok = finite && r.converged
      const kept = [s.x, ...s.history].slice(0, scheme.keep)
      return {
        ...s,
        t: s.t + 1,
        time: s.time + hk,
        x: fromData(r.y, [r.y.length]),
        stepSize: hk,
        evaluations: s.evaluations + r.evaluations + extra,
        jacobianEvaluations: s.jacobianEvaluations + r.jacobians,
        newtonSteps: r.iterations,
        newtonConverged: r.converged,
        history: kept,
        diverged: !ok,
        failure: ok ? null : finite ? 'newton failed' : 'not finite',
      }
    },
    done: (s) => reached(s.time, h, tEnd),
  }
}

/**
 * Implicit (backward) Euler: x_{n+1} = x_n + h f(t_{n+1}, x_{n+1}). Order 1 and L-stable: its stability region
 * contains the whole left half-plane and |R(z)| → 0 as z → −∞, so stiff components are damped at any step size.
 * `init` takes `{ x0, t0 }`.
 */
export function implicitEuler(f: Rhs, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  return implicitSolver(f, SCHEMES['implicit-euler'], options)
}

/**
 * The implicit trapezoid rule (Crank–Nicolson for PDEs): x_{n+1} = x_n + (h/2)(f(t_n, x_n) + f(t_{n+1}, x_{n+1})). Order 2 and
 * A-stable but not L-stable: |R(z)| → 1 as z → −∞, so very stiff components oscillate in sign instead of decaying.
 */
export function implicitTrapezoid(f: Rhs, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  return implicitSolver(f, SCHEMES['implicit-trapezoid'], options)
}

/**
 * The backward differentiation formula of order 1, 2 or 3 with a fixed step: x_{n+1} = Σ_j a_j x_{n+1−j} + β h
 * f(t_{n+1}, x_{n+1}) (BDF1 is implicit Euler). The first order − 1 steps, before the history exists, are L-stable
 * TR-BDF2 steps, accurate enough not to lower the global order. BDF1
 * and BDF2 are A-stable; BDF3 is A(α)-stable with α ≈ 86°.
 */
export function bdf(f: Rhs, order: 1 | 2 | 3, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  if (![1, 2, 3].includes(order)) throw new DomainError('bdf', 'bdf: the order must be 1, 2 or 3')
  return implicitSolver(f, bdfScheme(order), options)
}
