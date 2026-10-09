/**
 * Fixed-step implicit solvers for stiff systems $\xvec' = f(t, \xvec)$: implicit (backward) Euler, the implicit
 * trapezoid rule and the backward differentiation formulae BDF1 to BDF3 (Hairer & Wanner, 1996, "Solving Ordinary
 * Differential Equations II", §IV.1 and §III.1; Ascher & Petzold, 1998, §5.1).
 *
 * Every method here reduces a step to the equation $\yvec = \cvec + \gamma h \, f(t_{n+1}, \yvec)$, with $\cvec$
 * collecting the known terms, and solves it by Newton's method (`newtonSystem` of `aifn-compute/numerics/roots`) with
 * the Jacobian $\partial f/\partial\xvec$ from `aifn-compute/foundation/autodiff` (or forward differences, or a
 * given function). A step whose Newton iteration fails or whose state is not finite is reported in the state
 * (`diverged`, `failure`), not thrown. BDF2 and BDF3 start with TR-BDF2 steps until they have the history they need.
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
 * The Jacobian $\partial f/\partial\xvec$ at $(t, \xvec)$ as a row-major $n \times n$ array. `'autodiff'`
 * differentiates $f$ with a traced $\xvec$; `'finite-difference'` uses forward differences with step
 * $h_j = \sqrt{\varepsilon} \max(1, \lvert x_j \rvert)$, at a cost of $n + 1$ evaluations of $f$; a function is
 * called and its result read as the matrix. Internal to the implicit solvers.
 *
 * @param f The right-hand side $f$.
 * @param option How to obtain the Jacobian: `'autodiff'`, `'finite-difference'`, or a function of `(t, x)` returning
 *   the $n \times n$ matrix.
 * @param t The time at which the Jacobian is taken.
 * @param x The state $\xvec$, $n$ values; not modified.
 * @param where The caller's name, for error messages.
 * @returns `J`, the Jacobian row-major ($J_{ij} = \partial f_i / \partial x_j$ at entry `i * n + j`), and
 *   `evaluations`, the evaluations of $f$ it cost ($n + 1$ for finite differences, 0 otherwise).
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
  /** How to compute $\partial f/\partial\xvec$. Default `'autodiff'`. */
  jacobian?: JacobianOption
  /**
   * Newton stops when $\lVert G(\yvec) \rVert_2 \le$ this (or when its step is below $10^{-15}$ relative to
   * $\yvec$). Default 1e-10.
   */
  newtonTolerance?: Scalar
  /** Most Newton iterations per Newton solve (a TR-BDF2 start-up step makes two solves). Default 20. */
  maxNewtonSteps?: Size
}

/** The state of an implicit solver. */
export interface ImplicitState extends OdeState {
  /** Newton iterations used by the last step. */
  newtonSteps: Size
  /** Whether the last step's Newton iteration converged. */
  newtonConverged: boolean
  /**
   * Earlier states $\xvec_{n-1}, \xvec_{n-2}$ before the current $\xvec_n$ (newest first), kept by BDF2 (one) and
   * BDF3 (two); empty for the one-step methods.
   */
  history: Vector[]
}

/**
 * Solves the step equation $\yvec = \cvec + \gamma h \, f(t_1, \yvec)$ (the form shared by every method here,
 * with $\cvec$ collecting the known terms) by Newton's method on
 * $G(\yvec) = \yvec - \cvec - \gamma h \, f(t_1, \yvec)$, whose Jacobian is
 * $\Imat - \gamma h \, \partial f/\partial\xvec$, from the guess $\yvec_0$. The Jacobian of $f$ is taken afresh
 * at every iterate (full Newton).
 *
 * @param f The right-hand side $f$.
 * @param jac How to obtain $\partial f/\partial\xvec$ (see `jacobianOf`).
 * @param t1 The time $t_1$ at which $f$ is evaluated: the end of the step or stage.
 * @param c The known part $\cvec$ of the equation, $n$ values; not modified.
 * @param gammaH The weight $\gamma h$ of $f$ in the equation (the step size already multiplied in).
 * @param guess Newton's starting point $\yvec_0$, $n$ values (the solvers pass the current state).
 * @param tol Newton's residual tolerance on $\lVert G(\yvec) \rVert_2$.
 * @param maxNewton The most Newton iterations.
 * @param where The caller's name, for error messages.
 * @returns `y`, the last Newton iterate; `iterations`, the Newton iterations taken; `converged`, whether Newton
 *   converged without a failure; `evaluations`, the evaluations of $f$ (including those of finite-difference
 *   Jacobians); and `jacobians`, the Jacobians taken.
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

/** One implicit method as `implicitSolver` steps it: how it forms its step equation and what history it keeps. */
type Scheme = {
  /** The method's name: the algorithm's `name` and the prefix of its error messages. */
  name: string
  /**
   * The known part $\cvec$ and the weight $\gamma$ of the step equation
   * $\yvec = \cvec + \gamma h \, f(t + h, \yvec)$, from the current state `x`, the `history` of earlier states
   * (newest first), `fx` $= f(t, \xvec)$ (only meaningful when `needsFx`) and the step `h`.
   */
  equation: (x: F64, history: F64[], fx: F64, h: number) => { c: F64; gamma: number }
  /** Whether the equation uses $f(t, \xvec)$ (the implicit trapezoid rule). */
  needsFx: boolean | ((historyLength: number) => boolean)
  /** How many past states to keep. */
  keep: number
  /** Whether a step must be a one-step start-up step (TR-BDF2) because the history is too short. */
  startup?: (historyLength: number) => boolean
}

/**
 * BDF coefficients: $\sum_{j=0}^{k} \alpha_j \xvec_{n+1-j} = \beta h \, f(t_{n+1}, \xvec_{n+1})$, written as
 * $\xvec_{n+1} = \sum_{j=1}^{k} a_j \xvec_{n+1-j} + \beta h \, f(t_{n+1}, \xvec_{n+1})$, with `a[j - 1]` $= a_j$.
 * Orders 1 to 3, entry $k - 1$ for order $k$ (Hairer & Wanner, 1996, §III.1, Table 1.1). Until a multistep method
 * has its history, `implicitSolver` takes TR-BDF2 steps instead.
 */
const BDF: { a: number[]; beta: number }[] = [
  { a: [1], beta: 1 },
  { a: [4 / 3, -1 / 3], beta: 2 / 3 },
  { a: [18 / 11, -9 / 11, 2 / 11], beta: 6 / 11 },
]

/**
 * The BDF method of the given order as a `Scheme`: it keeps $k - 1$ earlier states and takes TR-BDF2 start-up steps
 * while it has fewer.
 *
 * @param order The order $k$ of the formula, 1 to 3.
 * @returns The scheme named `bdf1`, `bdf2` or `bdf3`.
 */
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

/**
 * The one-step schemes: implicit Euler, $\cvec = \xvec_n$ and $\gamma = 1$; the implicit trapezoid rule,
 * $\cvec = \xvec_n + \tfrac{h}{2} f(t_n, \xvec_n)$ and $\gamma = \tfrac{1}{2}$.
 */
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

/**
 * The fixed-step algorithm shared by every method of this file: each step forms the scheme's equation and solves it
 * by Newton's method (or takes a TR-BDF2 start-up step while the scheme's history is too short). The last step is
 * shortened to land on `tEnd`, and a shortened step drops the history. A failed Newton solve or a non-finite state
 * sets `diverged` and `failure` (`'newton failed'` or `'not finite'`).
 *
 * @param f The right-hand side $f$.
 * @param scheme The method: its step equation, the history it keeps and when it needs a start-up step.
 * @param options The step size (finite and non-zero, or `DomainError` is thrown), end time, Jacobian option and
 *   Newton settings.
 * @returns The algorithm; `init` takes `{ x0, t0 }` and throws when the step points away from `tEnd`.
 */
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
 * Implicit (backward) Euler with a fixed step: $\xvec_{n+1} = \xvec_n + h f(t_{n+1}, \xvec_{n+1})$, solved for
 * $\xvec_{n+1}$ by Newton's method. Order 1 and L-stable: its stability region contains the whole left half-plane and
 * $\lvert R(z) \rvert \to 0$ as $z \to -\infty$, so stiff components are damped at any step size. `init` takes
 * `{ x0, t0 }`; the run stops at `tEnd` when given. A failed Newton solve or a non-finite state sets `diverged` and
 * `failure` instead of throwing. A zero or non-finite step, or (at `init`) one pointing away from `tEnd`, throws
 * `DomainError`.
 *
 * @param f The right-hand side $f(t, \xvec)$. With the default `jacobian: 'autodiff'` it is called with a traced
 *   $\xvec$, so it must be written with tensor primitives.
 * @param options The step size $h$ (negative integrates backwards), the optional end time, how to obtain the Jacobian
 *   and Newton's tolerance and iteration limit.
 * @returns The step-through algorithm: each step is one implicit Euler step.
 *
 * @example A stiff decay stays stable at a large step
 * // x′ = −50x with h = 0.1: explicit Euler would multiply x by 1 − 5 = −4 each step.
 * const s = run(implicitEuler((t, x) => mul(-50, x), { stepSize: 0.1, tEnd: 1 }), { x0: [1] }, 100)
 * print('x(1) =', s.x)
 * print('(1 / (1 + 5))^10 =', (1 / 6) ** 10)
 * print('steps =', s.t)
 * print('Newton iterations in the last step =', s.newtonSteps)
 *
 * @example Halving the step halves the error (order 1)
 * for (const h of [0.1, 0.05, 0.025]) {
 *   const s = run(implicitEuler((t, x) => neg(x), { stepSize: h, tEnd: 1 }), { x0: [1] }, 1000)
 *   print(`h = ${h}: error =`, s.x.data[0] - Math.exp(-1))
 * }
 */
export function implicitEuler(f: Rhs, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  return implicitSolver(f, SCHEMES['implicit-euler'], options)
}

/**
 * The implicit trapezoid rule (Crank–Nicolson for PDEs) with a fixed step:
 * $\xvec_{n+1} = \xvec_n + \tfrac{h}{2} \big(f(t_n, \xvec_n) + f(t_{n+1}, \xvec_{n+1})\big)$, solved for
 * $\xvec_{n+1}$ by Newton's method. Order 2 and A-stable but not L-stable: $\lvert R(z) \rvert \to 1$ as
 * $z \to -\infty$, so very stiff components oscillate in sign instead of decaying. Each step costs one more
 * evaluation of $f$ than implicit Euler, for $f(t_n, \xvec_n)$. `init` takes `{ x0, t0 }`; failures are reported and
 * the step is checked as for `implicitEuler`.
 *
 * @param f The right-hand side $f(t, \xvec)$, written with tensor primitives when the Jacobian is by autodiff.
 * @param options The step size $h$, the optional end time, how to obtain the Jacobian and Newton's settings.
 * @returns The step-through algorithm: each step is one trapezoid step.
 *
 * @example Quartering the error by halving the step (order 2)
 * for (const h of [0.1, 0.05]) {
 *   const s = run(implicitTrapezoid((t, x) => neg(x), { stepSize: h, tEnd: 1 }), { x0: [1] }, 100)
 *   print(`h = ${h}: error =`, s.x.data[0] - Math.exp(-1))
 * }
 *
 * @example A very stiff mode flips sign instead of decaying
 * // x′ = −1000x with h = 0.1: each step multiplies x by (1 − 50) / (1 + 50); implicit Euler by 1 / 101.
 * const f = (t, x) => mul(-1000, x)
 * for (const n of [1, 2, 3]) {
 *   print(`trapezoid, ${n} steps:`, run(implicitTrapezoid(f, { stepSize: 0.1 }), { x0: [1] }, n).x)
 *   print(`implicit Euler, ${n} steps:`, run(implicitEuler(f, { stepSize: 0.1 }), { x0: [1] }, n).x)
 * }
 */
export function implicitTrapezoid(f: Rhs, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  return implicitSolver(f, SCHEMES['implicit-trapezoid'], options)
}

/**
 * The backward differentiation formula of order $k$ = 1, 2 or 3 with a fixed step:
 * $\xvec_{n+1} = \sum_{j=1}^{k} a_j \xvec_{n+1-j} + \beta h \, f(t_{n+1}, \xvec_{n+1})$, solved for $\xvec_{n+1}$
 * by Newton's method (BDF1 is implicit Euler). The first $k - 1$ steps, before the history exists, are L-stable
 * TR-BDF2 steps, accurate enough not to lower the global order; a last step shortened to land on `tEnd` is one too.
 * BDF1 and BDF2 are A-stable; BDF3 is $A(\alpha)$-stable with $\alpha \approx 86^\circ$. An order other than 1, 2
 * or 3 throws `DomainError`; failures are reported and the step is checked as for `implicitEuler`.
 *
 * @param f The right-hand side $f(t, \xvec)$, written with tensor primitives when the Jacobian is by autodiff.
 * @param order The order $k$ of the formula: 1, 2 or 3.
 * @param options The step size $h$, the optional end time, how to obtain the Jacobian and Newton's settings.
 * @returns The step-through algorithm; its state keeps the $k - 1$ earlier states in `history`.
 *
 * @example Higher order, smaller error
 * for (const order of [1, 2, 3]) {
 *   const s = run(bdf((t, x) => neg(x), order, { stepSize: 0.1, tEnd: 1 }), { x0: [1] }, 100)
 *   print(`BDF${order}: error at t = 1 =`, s.x.data[0] - Math.exp(-1))
 * }
 *
 * @example A stiff system with a slow and a fast mode
 * // x′ = diag(−1, −1000) x: the fast mode decays at once, the slow one is followed accurately.
 * const f = (t, x) => mul(tensor([-1, -1000]), x)
 * const s = run(bdf(f, 2, { stepSize: 0.1, tEnd: 1 }), { x0: [1, 1] }, 100)
 * print('x(1) =', s.x)
 * print('exact =', [Math.exp(-1), Math.exp(-1000)])
 * print('earlier states kept =', s.history.length)
 */
export function bdf(f: Rhs, order: 1 | 2 | 3, options: ImplicitOptions): Algorithm<InitialValue, ImplicitState> {
  if (![1, 2, 3].includes(order)) throw new DomainError('bdf', 'bdf: the order must be 1, 2 or 3')
  return implicitSolver(f, bdfScheme(order), options)
}
