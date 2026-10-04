/**
 * Explicit Runge–Kutta methods from their Butcher tableaux: explicit Euler, Heun, the midpoint rule and the classical
 * fourth-order method (Butcher, 2016, "Numerical Methods for Ordinary Differential Equations", 3rd ed., §23; Hairer,
 * Nørsett & Wanner, 1993, "Solving Ordinary Differential Equations I", §II.1).
 */

import {
  dense,
  fromData,
  isTensor,
  isTraced,
  linearCombination,
  shapeOfValue,
  unwrap,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, NotDifferentiableError, ShapeError } from 'aifn-compute/foundation/errors'
import type { FixedStepOptions, InitialValue, OdeState, Rhs } from './types'

const { allFinite, toF64 } = dense
type F64 = dense.F64

/**
 * A Butcher tableau: stage times c, stage coefficients a (strictly lower triangular for an explicit method, rows of
 * length s), weights b, and optionally the weights `bHat` of an embedded lower-order solution for error estimates.
 */
export type ButcherTableau = {
  name: string
  /** The classical order of the method. */
  order: Size
  c: readonly number[]
  a: readonly (readonly number[])[]
  b: readonly number[]
  bHat?: readonly number[]
}

/** Explicit Euler: x ← x + h f(t, x). Order 1. */
export const EULER: ButcherTableau = { name: 'euler', order: 1, c: [0], a: [[]], b: [1] }

/** Heun's method (the explicit trapezoid rule): average the slopes at both ends of an Euler step. Order 2. */
export const HEUN: ButcherTableau = { name: 'heun', order: 2, c: [0, 1], a: [[], [1]], b: [0.5, 0.5] }

/** The explicit midpoint rule: the slope at the midpoint of a half Euler step. Order 2. */
export const MIDPOINT: ButcherTableau = { name: 'midpoint', order: 2, c: [0, 0.5], a: [[], [0.5]], b: [0, 1] }

/** The classical fourth-order Runge–Kutta method (Kutta, 1901). Order 4. */
export const RK4: ButcherTableau = {
  name: 'rk4',
  order: 4,
  c: [0, 0.5, 0.5, 1],
  a: [[], [0.5], [0, 0.5], [0, 0, 1]],
  b: [1 / 6, 1 / 3, 1 / 3, 1 / 6],
}

/** The explicit tableaux by name. */
export const TABLEAUX = { euler: EULER, heun: HEUN, midpoint: MIDPOINT, rk4: RK4 } as const

const lengthOf = (v: Value) =>
  isTensor(v) && v.shape.length === 1 ? v.shape[0] : shapeOfValue(v).reduce((a, b) => a * b, 1)

/**
 * f(t, x) as a value, its length checked against x's (internal to the ode solvers). x may be traced: f written with
 * primitives then returns a traced derivative, which is what lets `unrolled` differentiate through a solver.
 */
export function evaluateValue(f: Rhs, t: Scalar, x: Value, where: string): Value {
  const out = f(t, x as Tensor)
  // A plain array for a traced state was computed on raw values: its dependence on x is lost, and a gradient through
  // the solver would silently be wrong.
  if (isTraced(x) && !isTensor(out) && !isTraced(out) && typeof out !== 'number')
    throw new NotDifferentiableError(
      where,
      `${where}: f returned a plain array for a traced state; write it with aifn primitives to differentiate the solution`,
    )
  const k: Value =
    typeof out === 'number' || isTensor(out) || isTraced(out)
      ? (out as Value)
      : fromData(Float64Array.from(out as ArrayLike<number>), [(out as ArrayLike<number>).length])
  const n = lengthOf(x)
  if (lengthOf(k) !== n)
    throw new ShapeError(where, `${where}: f returned ${lengthOf(k)} values for a state of length ${n}`)
  return k
}

/** f(t, x) on a working array (internal to the ode solvers that compute on raw arrays). */
export function evaluate(f: Rhs, t: Scalar, x: F64, where: string): F64 {
  return toF64(unwrap(evaluateValue(f, t, fromData(x, [x.length]), where)) as Tensor, where)
}

/**
 * The stages k_i = f(t + c_i h, x + h Σ_j a_ij k_j) of an explicit tableau; `k0` reuses a known f(t, x). Written with
 * primitives, so a traced x (or an f closing over traced parameters) gives traced stages.
 */
export function stages(
  f: Rhs,
  tab: ButcherTableau,
  t: number,
  x: Value,
  h: number,
  where: string,
  k0?: Value,
): Value[] {
  const k: Value[] = []
  for (let i = 0; i < tab.b.length; i++) {
    if (i === 0 && k0 !== undefined) {
      k.push(k0)
      continue
    }
    k.push(evaluateValue(f, t + tab.c[i] * h, combine(x, h, tab.a[i], k), where))
  }
  return k
}

/** x + h Σ w_i k_i (terms with w_i = 0 skipped), as one primitive. */
export function combine(x: Value, h: number, w: readonly number[], k: readonly Value[]): Value {
  const terms: Value[] = [x]
  const coefficients = [1]
  for (let i = 0; i < w.length; i++) {
    if (w[i] === 0) continue
    terms.push(k[i])
    coefficients.push(h * w[i])
  }
  return terms.length === 1 ? x : linearCombination(terms, coefficients)
}

/** The initial state every solver starts from (internal to the ode solvers). */
export function initialState(x0: F64, t0: Scalar): OdeState {
  return {
    t: 0,
    time: t0,
    x: fromData(x0, [x0.length]),
    stepSize: 0,
    error: NaN,
    evaluations: 0,
    jacobianEvaluations: 0,
    rejected: 0,
    diverged: !allFinite(x0),
    failure: allFinite(x0) ? null : 'not finite',
  }
}

/**
 * The initial state of a solver that steps on values (internal): a traced x₀ (differentiating the solution with
 * respect to the initial state) is kept as the state, the rest of the state read from its primal value.
 */
export function initialValue(x0: VectorLike, t0: Scalar, where: string): OdeState {
  if (!isTraced(x0)) return initialState(toF64(x0, where), t0)
  return { ...initialState(toF64(unwrap(x0 as unknown as Value) as Tensor, where), t0), x: x0 as unknown as Vector }
}

/** The step to take from time t (internal): h, shortened so as not to pass `tEnd`. */
export function nextStep(t: number, h: number, tEnd: number | undefined): number {
  if (tEnd === undefined) return h
  const left = tEnd - t
  return Math.abs(left) < Math.abs(h) ? left : h
}

/**
 * Throw unless a fixed step h points from t₀ towards `tEnd` (internal). Otherwise `reached` holds at t₀ and the run
 * would report `done` without taking a step.
 */
export function checkDirection(t0: number, h: number, tEnd: number | undefined, where: string): void {
  if (tEnd !== undefined && tEnd !== t0 && Math.sign(tEnd - t0) !== Math.sign(h))
    throw new DomainError(where, `${where}: the step size ${h} points away from tEnd = ${tEnd} (t₀ = ${t0})`)
}

/** True when a solver has reached `tEnd` (to rounding; internal). */
export function reached(t: number, h: number, tEnd: number | undefined): boolean {
  if (tEnd === undefined) return false
  return Math.sign(h) * (tEnd - t) <= 1e-12 * Math.max(1, Math.abs(tEnd))
}

/**
 * A fixed-step explicit Runge–Kutta solver for x′ = f(t, x) from a Butcher tableau or the name of one (`'euler'`,
 * `'heun'`, `'midpoint'`, `'rk4'`). Each step costs one evaluation of f per stage; the global error is O(h^order).
 * `init` takes `{ x0, t0 }`; the run stops at `tEnd` when given. A non-finite state sets `diverged`. The steps are
 * written with primitives: with f written with primitives too, `unrolled` differentiates the solution with respect to
 * x₀ and to the parameters f closes over (discretise-then-differentiate, exact for the discrete solution).
 *
 * @example run(rungeKutta((t, x) => neg(x), 'rk4', { stepSize: 0.1, tEnd: 1 }), { x0: [1] }, 100).x // ≈ e⁻¹
 */
export function rungeKutta(
  f: Rhs,
  method: keyof typeof TABLEAUX | ButcherTableau,
  { stepSize: h, tEnd }: FixedStepOptions,
): Algorithm<InitialValue, OdeState> {
  const tab = typeof method === 'string' ? TABLEAUX[method] : method
  if (!tab) throw new DomainError('rungeKutta', `rungeKutta: unknown method ${String(method)}`)
  if (!(h !== 0 && Number.isFinite(h)))
    throw new DomainError('rungeKutta', 'rungeKutta: the step size must be finite and non-zero')
  const name = tab.name
  return {
    name,
    init: ({ x0, t0 = 0 }) => {
      checkDirection(t0, h, tEnd, name)
      return initialValue(x0, t0, name)
    },
    step: (s) => {
      const hk = nextStep(s.time, h, tEnd)
      const k = stages(f, tab, s.time, s.x, hk, name)
      const y = combine(s.x, hk, tab.b, k)
      const finite = allFinite(dense.data(unwrap(y) as Tensor))
      return {
        ...s,
        t: s.t + 1,
        time: s.time + hk,
        x: y as Vector,
        stepSize: hk,
        evaluations: s.evaluations + tab.b.length,
        diverged: !finite,
        failure: finite ? null : 'not finite',
      }
    },
    done: (s) => reached(s.time, h, tEnd),
  }
}
