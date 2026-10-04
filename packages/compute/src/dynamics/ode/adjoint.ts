/**
 * The adjoint sensitivity method for ODE solutions (Pontryagin et al., 1962, "The Mathematical Theory of Optimal
 * Processes"; Chen, Rubanova, Bettencourt & Duvenaud, 2018, "Neural ordinary differential equations", NeurIPS,
 * §2 and appendix B): the gradient of a loss of x(t₁) with respect to x(t₀) and the parameters θ of f, from one
 * backward solve of the augmented system instead of a record of every step.
 *
 * With x′ = f(t, x, θ) and a(t) = ∂L/∂x(t), the adjoint obeys a′ = −aᵀ ∂f/∂x with a(t₁) = ∂L/∂x(t₁), and
 * ∂L/∂θ = ∫_{t₀}^{t₁} aᵀ ∂f/∂θ dt. Integrating [x, a, g] backwards from t₁ with x′ = f, a′ = −aᵀ∂f/∂x and
 * g′ = −aᵀ∂f/∂θ, g(t₁) = 0, gives a(t₀) = ∂L/∂x₀ and g(t₀) = ∂L/∂θ. The vector–Jacobian products come from one `vjp`
 * of f per evaluation. Memory is O(1) in the number of steps (the state is recomputed backwards); with `checkpoints`
 * the forward pass keeps x at the segment boundaries and the backward pass restarts x from them, which bounds the error
 * of reconstructing x by integrating an unstable direction backwards.
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
 * A right-hand side x′ = f(t, x, θ) with parameters θ (a number or a tensor), written with `aifn-compute/foundation/tensor`
 * primitives so that its vector–Jacobian products exist.
 */
export type ParametricRhs = (t: Scalar, x: Value, params: Value) => Value

/** Options of {@link odeAdjoint}. */
export type OdeAdjointOptions = {
  /**
   * The solver of both the forward and the backward (augmented) solve: an explicit tableau or Dormand–Prince.
   * Default `'rk4'`.
   */
  method?: Extract<OdeMethod, 'euler' | 'heun' | 'midpoint' | 'rk4' | 'dormand-prince'>
  /** The step size of a fixed-step method. Default (t₁ − t₀)/100. */
  stepSize?: Scalar
  /** Tolerances of Dormand–Prince. */
  rtol?: Scalar
  atol?: Scalar
  /** The most steps per solve. Default 100 000. */
  maxSteps?: Size
  /**
   * The number of equal segments of [t₀, t₁] whose start states the forward pass keeps (memory O(checkpoints)). The
   * backward pass restarts x from each. Default 1: only x(t₁) is kept, and x is integrated backwards throughout.
   */
  checkpoints?: Size
  /**
   * Called after every solve with its work and end state: the forward segments, then the backward (augmented) segments
   * from t₁ to t₀. Comparing a backward segment's reconstructed x with the forward state kept at its end time measures
   * the reconstruction error that `checkpoints` bounds.
   */
  onSolve?: (info: OdeSolveInfo) => void
}

/** The record of one solve of {@link odeAdjoint} (or of `odeFlow`), as passed to `onSolve`. */
export type OdeSolveInfo = {
  /** `'forward'` for x′ = f; `'backward'` for the augmented system [x, a, g] integrated from `from` to `to` < `from`. */
  phase: 'forward' | 'backward'
  from: Scalar
  to: Scalar
  /** Evaluations of the right-hand side (of f forward; of the augmented system, one vjp of f each, backward). */
  evaluations: Size
  /** Accepted steps. */
  steps: Size
  /** Rejected step attempts (adaptive methods). */
  rejected: Size
  /** The primal x at `to` (the x part of the augmented state backward), flattened. */
  x: Float64Array
}

const flat = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))

/**
 * The solution map (x₀, θ) ↦ x(t₁) of x′ = f(t, x, θ) on [t₀, t₁], differentiated by the adjoint method: the
 * forward pass runs the solver on values and keeps only x(t₁) (and the checkpoints); the reverse rule integrates the
 * augmented system [x, a, g] backwards with the same solver and returns ∂L/∂x₀ = a(t₀) and ∂L/∂θ = g(t₀). This is
 * optimise-then-discretise: the gradient is that of the exact flow to the solver's accuracy, not the exact gradient
 * of the discrete solution (which `unrolled` over `rungeKutta` gives, at memory linear in the steps). Forward mode
 * obtains J·t from the same rule by the transpose trick.
 *
 * @example
 * const flow = odeAdjoint((t, x, k) => mul(neg(k), x), [0, 1], { stepSize: 0.01 })
 * grad((k: Value) => sum(flow(tensor([1]), k)))(2) // ≈ −e⁻²
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
