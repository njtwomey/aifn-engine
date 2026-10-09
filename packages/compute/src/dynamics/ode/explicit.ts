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
 * A Butcher tableau of an $s$-stage explicit method: stage times $c_i$, stage coefficients $a_{ij}$ (strictly lower
 * triangular), weights $b_i$, and optionally the weights $\hat{b}_i$ of an embedded lower-order solution for error
 * estimates. A step from $(t, \xvec)$ with size $h$ is $\xvec + h \sum_i b_i \kvec_i$, with
 * $\kvec_i = f(t + c_i h, \xvec + h \sum_{j<i} a_{ij} \kvec_j)$.
 */
export type ButcherTableau = {
  /** The method's name, which becomes the solver's `name`. */
  name: string
  /** The classical order of the method. */
  order: Size
  /** The stage times $c_i$, as fractions of the step (length $s$). */
  c: readonly number[]
  /** The stage coefficients: row $i$ holds $a_{i0}, \dots, a_{i,i-1}$ (so row 0 is empty). */
  a: readonly (readonly number[])[]
  /** The weights $b_i$ of the solution (length $s$). */
  b: readonly number[]
  /** The weights $\hat{b}_i$ of the embedded solution (length $s$), when the method estimates its error. */
  bHat?: readonly number[]
}

/** Explicit Euler: $\xvec \leftarrow \xvec + h f(t, \xvec)$. Order 1. */
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

/**
 * The number of entries of a value: a vector's length, or the product of any other value's shape (1 for a number).
 *
 * @param v The value to measure: a tensor, a traced value or a number.
 * @returns How many entries it holds.
 */
const lengthOf = (v: Value) =>
  isTensor(v) && v.shape.length === 1 ? v.shape[0] : shapeOfValue(v).reduce((a, b) => a * b, 1)

/**
 * $f(t, \xvec)$ as a value, its length checked against that of $\xvec$ (internal to the ode solvers). $\xvec$ may be
 * traced: $f$ written with primitives then returns a traced derivative, which is what lets `unrolled` differentiate
 * through a solver. A plain array returned for a traced $\xvec$ throws `NotDifferentiableError`, and a result of the
 * wrong length throws `ShapeError`.
 *
 * @param f The right-hand side.
 * @param t The time at which to evaluate it.
 * @param x The state $\xvec$, a vector of length $n$ (possibly traced).
 * @param where The caller's name for error messages.
 * @returns The derivative: a tensor or traced value as $f$ returned it, or a plain array converted to a vector.
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

/**
 * $f(t, \xvec)$ on a working array (internal to the ode solvers that compute on raw arrays), with the checks of
 * `evaluateValue`.
 *
 * @param f The right-hand side.
 * @param t The time at which to evaluate it.
 * @param x The state $\xvec$ as a plain array of $n$ values; not modified.
 * @param where The caller's name for error messages.
 * @returns A new array of the $n$ values of $f(t, \xvec)$.
 */
export function evaluate(f: Rhs, t: Scalar, x: F64, where: string): F64 {
  return toF64(unwrap(evaluateValue(f, t, fromData(x, [x.length]), where)) as Tensor, where)
}

/**
 * The stages $\kvec_i = f(t + c_i h, \xvec + h \sum_{j<i} a_{ij} \kvec_j)$ of an explicit tableau. Written with
 * primitives, so a traced $\xvec$ (or an $f$ closing over traced parameters) gives traced stages.
 *
 * @param f The right-hand side.
 * @param tab The tableau whose `c` and `a` define the stages; one stage per entry of its `b`.
 * @param t The time $t$ at the start of the step.
 * @param x The state $\xvec$ at the start of the step (possibly traced).
 * @param h The step size $h$.
 * @param where The caller's name for error messages.
 * @param k0 A known $f(t, \xvec)$ to use as the first stage instead of evaluating it (first same as last); when left
 *   out, every stage is evaluated.
 * @returns The stages $\kvec_i$, one per stage of the tableau.
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

/**
 * $\xvec + h \sum_i w_i \kvec_i$ (terms with $w_i = 0$ skipped), as one primitive.
 *
 * @param x The base value $\xvec$ (0 to combine the stages alone, as an error estimate does).
 * @param h The step size $h$ that scales every weight.
 * @param w The weights $w_i$: a tableau row of `a`, its `b`, or a difference of weights; may be shorter than `k`.
 * @param k The stages $\kvec_i$.
 * @returns The combination, or `x` itself when every weight is 0.
 */
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

/**
 * The initial state every solver starts from (internal to the ode solvers): no steps or evaluations yet, an error of
 * NaN, and `diverged` (with `failure` `'not finite'`) when $\xvec_0$ has a non-finite entry.
 *
 * @param x0 The initial state $\xvec_0$ as a plain array of $n$ values; the state's vector is built on it, not copied.
 * @param t0 The initial time $t_0$.
 * @returns The state at step 0.
 */
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
 * The initial state of a solver that steps on values (internal): a traced $\xvec_0$ (differentiating the solution with
 * respect to the initial state) is kept as the state, the rest of the state read from its primal value.
 *
 * @param x0 The initial state $\xvec_0$: an array, a vector or a traced value.
 * @param t0 The initial time $t_0$.
 * @param where The caller's name for error messages.
 * @returns The state at step 0, as `initialState` makes it, with `x` the traced $\xvec_0$ when it is traced.
 */
export function initialValue(x0: VectorLike, t0: Scalar, where: string): OdeState {
  if (!isTraced(x0)) return initialState(toF64(x0, where), t0)
  return { ...initialState(toF64(unwrap(x0 as unknown as Value) as Tensor, where), t0), x: x0 as unknown as Vector }
}

/**
 * The step to take from time $t$ (internal): $h$, shortened so as not to pass `tEnd`.
 *
 * @param t The current time.
 * @param h The fixed step size (negative for a backward run).
 * @param tEnd The end time, or undefined for none (then $h$ is returned as it is).
 * @returns $h$, or $t_\text{end} - t$ when that is smaller in magnitude.
 */
export function nextStep(t: number, h: number, tEnd: number | undefined): number {
  if (tEnd === undefined) return h
  const left = tEnd - t
  return Math.abs(left) < Math.abs(h) ? left : h
}

/**
 * Throw `DomainError` unless a fixed step $h$ points from $t_0$ towards `tEnd` (internal). Otherwise `reached` holds at
 * $t_0$ and the run would report `done` without taking a step.
 *
 * @param t0 The initial time $t_0$.
 * @param h The fixed step size.
 * @param tEnd The end time, or undefined for none (nothing is checked, nor when it equals $t_0$).
 * @param where The caller's name for error messages.
 */
export function checkDirection(t0: number, h: number, tEnd: number | undefined, where: string): void {
  if (tEnd !== undefined && tEnd !== t0 && Math.sign(tEnd - t0) !== Math.sign(h))
    throw new DomainError(where, `${where}: the step size ${h} points away from tEnd = ${tEnd} (t₀ = ${t0})`)
}

/**
 * True when a solver has reached `tEnd` (to rounding; internal): the time left in the direction of $h$ is at most
 * $10^{-12} \max(1, \lvert t_\text{end} \rvert)$, so a time past `tEnd` also counts.
 *
 * @param t The current time.
 * @param h The step size, whose sign gives the direction of the run.
 * @param tEnd The end time, or undefined, in which case the end is never reached.
 * @returns Whether the run is at (or past) `tEnd`.
 */
export function reached(t: number, h: number, tEnd: number | undefined): boolean {
  if (tEnd === undefined) return false
  return Math.sign(h) * (tEnd - t) <= 1e-12 * Math.max(1, Math.abs(tEnd))
}

/**
 * A fixed-step explicit Runge–Kutta solver for $\xvec' = f(t, \xvec)$ from a Butcher tableau or the name of one
 * (`'euler'`, `'heun'`, `'midpoint'`, `'rk4'`). Each step costs one evaluation of $f$ per stage; the global error is
 * $O(h^p)$ for a method of order $p$. `init` takes `{ x0, t0 }`; the run stops at `tEnd` when given. A non-finite state
 * sets `diverged`. The state's `error` stays NaN: a fixed-step method makes no estimate, even from a tableau with
 * `bHat`. The steps are written with primitives: with $f$ written with primitives too, `unrolled` differentiates the
 * solution with respect to $\xvec_0$ and to the parameters $f$ closes over (discretise-then-differentiate, exact for
 * the discrete solution). An unknown method name, a zero or non-finite step, or (at `init`) a step pointing away from
 * `tEnd` throws `DomainError`.
 *
 * @param f The right-hand side $f(t, \xvec)$.
 * @param method The tableau, or the name of one of `TABLEAUX`.
 * @param options The step size and the optional end time.
 * @param options.stepSize The step size $h$; negative integrates backwards in time.
 * @param options.tEnd The time to stop at: the last step is shortened to land on it and the run is then `done`. When
 *   left out, the run takes as many steps as the runner asks for.
 * @returns The solver, an `Algorithm` to run with `run(alg, { x0, t0 }, steps)` or trace with `trace`.
 *
 * @example The classical method on exponential decay
 * // x′ = −x from x(0) = 1, so x(1) = e^{−1}: ten steps of 0.1.
 * const s = run(rungeKutta((t, x) => neg(x), 'rk4', { stepSize: 0.1, tEnd: 1 }), { x0: [1] }, 100)
 * print('x(1) =', s.x)
 * print('e^{-1} =', Math.exp(-1))
 * print('steps =', s.t)
 * print('evaluations of f =', s.evaluations)
 *
 * @example The error at t = 1 falls with the order of the method
 * const decay = (t, x) => neg(x)
 * for (const m of ['euler', 'heun', 'midpoint', 'rk4']) {
 *   const s = run(rungeKutta(decay, m, { stepSize: 0.1, tEnd: 1 }), { x0: [1] }, 100)
 *   print(m, 'error =', Math.abs(toFlat(s.x)[0] - Math.exp(-1)))
 * }
 *
 * @example A harmonic oscillator over one period, from a tableau of your own
 * // Ralston's second-order method; q′ = p, p′ = −q returns to (1, 0) after 2π.
 * const ralston = { name: 'ralston', order: 2, c: [0, 2 / 3], a: [[], [2 / 3]], b: [1 / 4, 3 / 4] }
 * const f = (t, x) => stack([get(x, 1), neg(get(x, 0))])
 * const s = run(rungeKutta(f, ralston, { stepSize: (2 * Math.PI) / 200, tEnd: 2 * Math.PI }), { x0: [1, 0] }, 1000)
 * print('time =', s.time)
 * print('x =', s.x)
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
