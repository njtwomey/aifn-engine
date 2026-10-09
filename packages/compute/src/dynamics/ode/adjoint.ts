/**
 * The adjoint sensitivity method for ODE solutions (Pontryagin et al., 1962, "The Mathematical Theory of Optimal
 * Processes"; Chen, Rubanova, Bettencourt & Duvenaud, 2018, "Neural ordinary differential equations", NeurIPS,
 * §2 and appendix B): the gradient of a loss of $\xvec(t_1)$ with respect to $\xvec(t_0)$ and the parameters
 * $\thetavec$ of $f$, from one backward solve of the augmented system instead of a record of every step.
 *
 * With $\xvec' = f(t, \xvec, \thetavec)$ and $\avec(t) = \partial L/\partial\xvec(t)$, the adjoint obeys
 * $\avec' = -\avec^\top \partial f/\partial\xvec$ with $\avec(t_1) = \partial L/\partial\xvec(t_1)$, and
 * $\partial L/\partial\thetavec = \int_{t_0}^{t_1} \avec^\top \partial f/\partial\thetavec \, dt$. Integrating
 * $[\xvec, \avec, \gvec]$ backwards from $t_1$ with $\xvec' = f$, $\avec' = -\avec^\top \partial f/\partial\xvec$
 * and $\gvec' = -\avec^\top \partial f/\partial\thetavec$, $\gvec(t_1) = \zeros$, gives
 * $\avec(t_0) = \partial L/\partial\xvec_0$ and $\gvec(t_0) = \partial L/\partial\thetavec$. The vector–Jacobian
 * products come from one `vjp` of $f$ per evaluation. Memory is $O(1)$ in the number of steps (the state is recomputed
 * backwards); with `checkpoints` the forward pass keeps $\xvec$ at the segment boundaries and the backward pass
 * restarts $\xvec$ from them, which bounds the error of reconstructing $\xvec$ by integrating an unstable direction
 * backwards.
 */

import { defineCustomVjp, vjp } from 'aifn-compute/foundation/autodiff'
import { DomainError, NumericalError } from 'aifn-compute/foundation/errors'
import {
  concat,
  fromData,
  get,
  neg,
  reshape,
  shapeOfValue,
  slice,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { solverFor, type OdeMethod } from './solve'
import type { InitialValue, OdeState, Rhs } from './types'

/**
 * A right-hand side $\xvec' = f(t, \xvec, \thetavec)$ with parameters $\thetavec$ (a number or a tensor), written
 * with `aifn-compute/foundation/tensor` primitives so that its vector–Jacobian products exist.
 */
export type ParametricRhs = (t: Scalar, x: Value, params: Value) => Value

/** Options of {@link odeAdjoint}. */
export type OdeAdjointOptions = {
  /**
   * The solver of both the forward and the backward (augmented) solve: an explicit tableau or Dormand–Prince.
   * Default `'rk4'`.
   */
  method?: Extract<OdeMethod, 'euler' | 'heun' | 'midpoint' | 'rk4' | 'dormand-prince'>
  /**
   * The step size of a fixed-step method (its magnitude; the sign follows the direction of each solve). Default
   * $(t_1 - t_0)/100$, whatever the number of checkpoints.
   */
  stepSize?: Scalar
  /** The relative tolerance of Dormand–Prince (default 1e-3, as `dormandPrince`). */
  rtol?: Scalar
  /** The absolute tolerance of Dormand–Prince (default 1e-6, as `dormandPrince`). */
  atol?: Scalar
  /** The most steps per solve (each segment, forward and backward); more throws `NumericalError`. Default 100 000. */
  maxSteps?: Size
  /**
   * The number of equal segments of $[t_0, t_1]$ whose start states the forward pass keeps (memory
   * $O(\text{checkpoints})$), a positive integer. The backward pass restarts $\xvec$ from each. Default 1: only
   * $\xvec(t_0)$ and $\xvec(t_1)$ are kept, and $\xvec$ is integrated backwards throughout.
   */
  checkpoints?: Size
  /**
   * Called after every solve with its work and end state: the forward segments, then the backward (augmented) segments
   * from $t_1$ to $t_0$. Comparing a backward segment's reconstructed $\xvec$ with the forward state kept at its end
   * time measures the reconstruction error that `checkpoints` bounds.
   */
  onSolve?: (info: OdeSolveInfo) => void
}

/** The record of one solve of {@link odeAdjoint} (or of `odeFlow`), as passed to `onSolve`. */
export type OdeSolveInfo = {
  /**
   * `'forward'` for $\xvec' = f$; `'backward'` for the augmented system $[\xvec, \avec, \gvec]$, integrated from
   * `from` back to `to`.
   */
  phase: 'forward' | 'backward'
  /** The time the solve started from. */
  from: Scalar
  /** The time the solve ended at. */
  to: Scalar
  /** Evaluations of the right-hand side (of f forward; of the augmented system, one vjp of f each, backward). */
  evaluations: Size
  /** Accepted steps. */
  steps: Size
  /** Rejected step attempts (adaptive methods). */
  rejected: Size
  /** The primal $\xvec$ at `to` (the $\xvec$ part of the augmented state backward), flattened. */
  x: Float64Array
}

/**
 * The values of a number or a (possibly traced) tensor as a flat array, read from the primal value.
 *
 * @param v The number or tensor.
 * @returns Its entries in row-major order (one entry for a number).
 */
const flat = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))

/**
 * The solution map $(\xvec_0, \thetavec) \mapsto \xvec(t_1)$ of $\xvec' = f(t, \xvec, \thetavec)$ on
 * $[t_0, t_1]$, differentiated by the adjoint method: the forward pass runs the solver on values and keeps only
 * $\xvec(t_1)$ (and the checkpoints); the reverse rule integrates the augmented system $[\xvec, \avec, \gvec]$
 * backwards with the same solver and returns $\partial L/\partial\xvec_0 = \avec(t_0)$ and
 * $\partial L/\partial\thetavec = \gvec(t_0)$. This is optimise-then-discretise: the gradient is that of the exact
 * flow to the solver's accuracy, not the exact gradient of the discrete solution (which `unrolled` over `rungeKutta`
 * gives, at memory linear in the steps). Forward mode obtains $\Jmat \tvec$ from the same rule by the transpose
 * trick. A solve that fails or does not reach its end within `maxSteps` throws `NumericalError` 'not-converged'; an
 * empty or non-finite interval, or `checkpoints` that is not a positive integer, throws `DomainError`.
 *
 * @param f The right-hand side $f(t, \xvec, \thetavec)$, written with tensor primitives: the backward pass takes
 *   its vector–Jacobian products with respect to $\xvec$ and $\thetavec$. It returns the derivative with $n$ values.
 * @param interval The times $[t_0, t_1]$ (finite and distinct; $t_1 < t_0$ integrates backwards).
 * @param options The solver, its step size or tolerances, the step limit, the checkpoints and a callback reporting
 *   the work of each solve.
 * @returns The differentiable map from $\xvec_0$ (a vector of $n$ values) and $\thetavec$ (a number or a tensor) to
 *   $\xvec(t_1)$; under `grad` the gradients have the shapes of $\xvec_0$ and $\thetavec$.
 *
 * @example Gradients of the decay x′ = −kx with respect to k and to the initial value
 * const flow = odeAdjoint((t, x, k) => mul(neg(k), x), [0, 1], { stepSize: 0.01 })
 * print('x(1) =', flow(tensor([1]), 2), ' e^-2 =', Math.exp(-2))
 * print('d x(1) / dk =', grad((k) => sum(flow(tensor([1]), k)))(2), ' -e^-2 =', -Math.exp(-2))
 * print('d x(1) / d x0 =', grad((x0) => sum(flow(x0, 2)))(tensor([1])))
 *
 * @example The solves behind one gradient, with two checkpoints
 * const solves = []
 * const flow = odeAdjoint((t, x, k) => mul(neg(k), x), [0, 1], {
 *   method: 'dormand-prince',
 *   checkpoints: 2,
 *   onSolve: (s) => solves.push(`${s.phase} ${s.from} to ${s.to}: ${s.steps} steps, ${s.evaluations} evaluations`),
 * })
 * print('d x(1) / dk =', grad((k) => sum(flow(tensor([1]), k)))(2))
 * for (const s of solves) print(s)
 */
export function odeAdjoint(
  f: ParametricRhs,
  [t0, t1]: readonly [Scalar, Scalar],
  options: OdeAdjointOptions = {},
): (x0: Value, params: Value) => Value {
  const {
    method = 'rk4',
    stepSize = (t1 - t0) / 100,
    rtol,
    atol,
    maxSteps = 100_000,
    checkpoints = 1,
    onSolve,
  } = options
  if (!(Number.isInteger(checkpoints) && checkpoints >= 1))
    throw new DomainError('odeAdjoint', 'odeAdjoint: checkpoints must be a positive integer')
  if (!(t1 !== t0 && Number.isFinite(t0) && Number.isFinite(t1)))
    throw new DomainError('odeAdjoint', 'odeAdjoint: the interval must be finite and non-empty')
  const name = 'odeAdjoint'
  const bounds = Array.from({ length: checkpoints + 1 }, (_, j) =>
    j === checkpoints ? t1 : t0 + ((t1 - t0) * j) / checkpoints,
  )

  /** x(to) from x(from) with the chosen solver, on values (raw or traced). */
  const solve = (rhs: Rhs, x: Value, from: Scalar, to: Scalar, phase: 'forward' | 'backward', n = 0): Value => {
    const h = Math.sign(to - from) * Math.abs(stepSize)
    const alg = solverFor(rhs, to, from, { method, stepSize: h, rtol, atol })
    const s: OdeState = run(alg, { x0: x, t0: from } as unknown as InitialValue, maxSteps)
    if (s.failure !== null || Math.abs(s.time - to) > 1e-9 * Math.max(1, Math.abs(to)))
      throw new NumericalError(
        name,
        `${name}: the solve from t = ${from} stopped at t = ${s.time} (${s.failure ?? 'maxSteps reached'})`,
        'not-converged',
      )
    if (onSolve) {
      const end = flat(s.x as Value)
      onSolve({
        phase,
        from,
        to,
        evaluations: s.evaluations,
        steps: s.t,
        rejected: s.rejected,
        x: phase === 'forward' ? end : end.slice(0, n),
      })
    }
    return s.x as Value
  }

  /** The states at every segment boundary, t₀ first. */
  const forward = (x0: Value, params: Value): Value[] => {
    const rhs: Rhs = (t, x) => f(t, x, params)
    const xs: Value[] = [x0]
    for (let j = 1; j < bounds.length; j++) xs.push(solve(rhs, xs[j - 1], bounds[j - 1], bounds[j], 'forward'))
    return xs
  }

  type Residuals = { states: Value[]; params: Value }

  return defineCustomVjp<[Value, Value], Value, Residuals>({
    name,
    f: (x0, params) => forward(x0, params).at(-1)!,
    fwd: (x0, params) => {
      const states = forward(x0, params)
      return { out: states.at(-1)!, residuals: { states, params } }
    },
    bwd: ({ states, params }, ct) => {
      const n = flat(states[0]).length
      const pShape = shapeOfValue(params)
      const p = pShape.reduce((a, b) => a * b, 1)
      const asVector = (v: Value, size: number): Value =>
        typeof v === 'number' || shapeOfValue(v).length !== 1 ? reshape(v, [size]) : v
      // [x, a, g]′ = [f, −aᵀ∂f/∂x, −aᵀ∂f/∂θ], θ held at its value.
      const augmented: Rhs = (t, z) => {
        const x = slice(z, [0, n])
        const a = slice(z, [n, 2 * n])
        const { value, pullback } = vjp(([xx, th]: [Value, Value]) => f(t, xx, th), [x, params] as [Value, Value])
        const [ax, ath] = pullback(a) as [Value, Value]
        return concat([asVector(value as Value, n), neg(asVector(ax, n)), neg(asVector(ath, p))])
      }
      let a: Value = asVector(ct, n)
      let g: Value = fromData(new Float64Array(p), [p])
      for (let j = bounds.length - 1; j >= 1; j--) {
        const z = solve(augmented, concat([asVector(states[j], n), a, g]), bounds[j], bounds[j - 1], 'backward', n)
        a = slice(z, [n, 2 * n])
        g = slice(z, [2 * n, 2 * n + p])
      }
      const gParams = typeof params === 'number' ? get(g, 0) : reshape(g, pShape)
      return [reshape(a, shapeOfValue(states[0])), gParams]
    },
  })
}
