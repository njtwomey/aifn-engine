/**
 * Differentiable ODE solves for neural ODEs and continuous normalising flows (Chen, Rubanova, Bettencourt & Duvenaud,
 * 2018, "Neural ordinary differential equations", NeurIPS; Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud, 2019,
 * "FFJORD", ICLR; Finlay, Jacobsen, Nurbekyan & Oberman, 2020, "How to train your neural ODE", ICML).
 *
 * - `odeFlow`: the states x(t₁), …, x(t_m) of x′ = f(t, x, θ) from x(t₀), differentiable with respect to x(t₀) and θ
 *   either by backpropagation through the solver's steps (discretise-then-optimise: the exact gradient of the discrete
 *   solution, memory linear in the steps) or by the adjoint method (optimise-then-discretise: one backward solve,
 *   memory constant in the steps, the gradient of the exact flow to the solver's accuracy). Work is reported per solve.
 * - `jacobianTrace`: tr(∂f/∂x) for each row of a batch, exactly (one jvp per dimension) or by Hutchinson's estimator
 *   εᵀ(∂f/∂x)ε with a Rademacher or Gaussian probe ε (Hutchinson, 1989), from one vjp.
 * - `augmentedDynamics`: f with the integrals a CNF and its regularisers need appended to the state: the change in log
 *   density d(Δ)/dt = −tr(∂f/∂x) (the instantaneous change of variables), the kinetic energy ‖f‖² and the Jacobian's
 *   Frobenius norm ‖εᵀ∂f/∂x‖² (RNODE).
 */

import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { normals, units, type Stream } from 'aifn-compute/foundation/random'
import { vjp, jvp } from 'aifn-compute/foundation/autodiff'
import {
  add,
  concat,
  fromData,
  mul,
  neg,
  reshape,
  shapeOfValue,
  slice,
  square,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { unrolled } from 'aifn-compute/foundation/trace'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { odeAdjoint, type OdeSolveInfo } from './adjoint'
import { solverFor } from './solve'
import type { OdeState, Rhs } from './types'

/** How the gradient of an ODE solve is computed. */
export type OdeGradient = 'backprop' | 'adjoint'

/** The solvers `odeFlow` can differentiate: the explicit tableaux and Dormand–Prince. */
export type OdeFlowMethod = 'euler' | 'heun' | 'midpoint' | 'rk4' | 'dormand-prince'

/** Options of {@link odeFlow}. */
export type OdeFlowOptions = {
  /** Backpropagation through the solver's steps, or the adjoint method. Default `'backprop'`. */
  gradient?: OdeGradient
  /** The solver. Default `'rk4'`. */
  method?: OdeFlowMethod
  /** The step size of a fixed-step method (its magnitude; the sign follows each interval). Default 0.1. */
  stepSize?: Scalar
  /** Tolerances of Dormand–Prince (default 1e-3 and 1e-6). */
  rtol?: Scalar
  atol?: Scalar
  /** The most steps per solve (default 10 000); a solve that needs more throws `NumericalError`. */
  maxSteps?: Size
  /** Segments per interval whose start states the adjoint keeps (its `checkpoints`). Default 1. */
  checkpoints?: Size
  /**
   * Called after every solve with its work. Backprop reports the forward solves only: its backward pass replays the
   * recorded tape, one vjp per recorded evaluation of f, so its cost equals the forward evaluations. The adjoint also
   * reports its backward (augmented) solves.
   */
  onSolve?: (info: OdeSolveInfo) => void
}

/** A right-hand side on a state of any shape: x′ = f(t, x, θ) with f(t, x, θ) shaped like x. */
export type ShapedRhs = (t: Scalar, x: Value, params: Value) => Value

const flatOf = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))

/**
 * The flow map of x′ = f(t, x, θ) sampled at `times` (increasing or decreasing): a function of x(t₀) (any shape) and θ
 * returning [x(t₀), x(t₁), …, x(t_m)], each shaped like x(t₀). Under `grad` it is differentiated by `gradient`:
 * `'backprop'` unrolls the solver on traced values (`unrolled`; Dormand–Prince chooses its steps on primal values, so
 * they are constants of the discrete solution), `'adjoint'` uses `odeAdjoint` on each interval. Without a transform it
 * simply solves.
 *
 * @example
 * const flow = odeFlow((t, x, k) => mul(neg(k), x), [0, 1], { gradient: 'adjoint', method: 'dormand-prince' })
 * grad((k: Value) => sum(flow(tensor([1, 2]), k)[1]))(0.5) // ≈ −3e^{−0.5}
 */
export function odeFlow(
  f: ShapedRhs,
  times: readonly Scalar[],
  options: OdeFlowOptions = {},
): (x0: Value, params: Value) => Value[] {
  const {
    gradient = 'backprop',
    method = 'rk4',
    stepSize = 0.1,
    rtol,
    atol,
    maxSteps = 10_000,
    checkpoints = 1,
    onSolve,
  } = options
  if (times.length < 2) throw new DomainError('odeFlow', 'odeFlow: needs at least two times')
  for (let i = 1; i < times.length; i++)
    if (!(Number.isFinite(times[i]) && times[i] !== times[i - 1]))
      throw new DomainError('odeFlow', 'odeFlow: the times must be finite and distinct')
  if (!(stepSize > 0 && Number.isFinite(stepSize)))
    throw new DomainError('odeFlow', 'odeFlow: stepSize must be positive')
  const name = 'odeFlow'

  return (x0, params) => {
    const shape = shapeOfValue(x0)
    const n = shape.reduce((a, b) => a * b, 1)
    // The solvers work on rank-1 states; f sees the caller's shape.
    const flatRhs = (t: Scalar, x: Value, p: Value): Value => reshape(f(t, reshape(x, shape), p), [n])
    const out: Value[] = [x0]
    let x: Value = reshape(x0, [n])
    for (let i = 1; i < times.length; i++) {
      const [from, to] = [times[i - 1], times[i]]
      if (gradient === 'adjoint') {
        x = odeAdjoint(flatRhs, [from, to], { method, stepSize, rtol, atol, maxSteps, checkpoints, onSolve })(x, params)
      } else {
        const rhs: Rhs = (t, z) => flatRhs(t, z, params)
        const h = Math.sign(to - from) * stepSize
        const alg = solverFor(rhs, to, from, { method, stepSize: h, rtol, atol })
        const s: OdeState = unrolled(alg, { x0: x as never, t0: from }, maxSteps)
        if (s.failure !== null || Math.abs(s.time - to) > 1e-9 * Math.max(1, Math.abs(to)))
          throw new NumericalError(
            name,
            `${name}: the solve from t = ${from} stopped at t = ${s.time} (${s.failure ?? 'maxSteps reached'})`,
            'not-converged',
          )
        onSolve?.({
          phase: 'forward',
          from,
          to,
          evaluations: s.evaluations,
          steps: s.t,
          rejected: s.rejected,
          x: flatOf(s.x as Value),
        })
        x = s.x as Value
      }
      out.push(reshape(x, shape))
    }
    return out
  }
}

// ── Divergence ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** The trace estimators of {@link jacobianTrace}. */
export type TraceEstimator = 'exact' | 'hutchinson'

/** The probe distributions of Hutchinson's estimator: both have E[εεᵀ] = I; Rademacher has the lower variance. */
export type ProbeKind = 'rademacher' | 'gaussian'

/** A probe of the given shape: independent ±1 signs (Rademacher) or standard normals. */
export function traceProbe(s: Stream, shape: readonly Size[], kind: ProbeKind = 'rademacher'): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  if (kind === 'gaussian') return normals(s, shape)
  return fromData(
    Float64Array.from(units(s, n), (u) => (u < 0.5 ? -1 : 1)),
    [...shape],
  )
}

/** Options of {@link jacobianTrace}. */
export type JacobianTraceOptions = {
  /** Default `'exact'`. */
  estimator?: TraceEstimator
  /** The probe ε, shaped like x (required by `'hutchinson'`, and by `'exact'` when `probeProduct` is wanted). */
  probe?: Value
  /** Also return εᵀ∂f/∂x per row (for the Jacobian Frobenius regulariser). Default false. */
  probeProduct?: boolean
}

/** The result of {@link jacobianTrace}. */
export type JacobianTrace = {
  /** f(x), shape [B, d]. */
  value: Value
  /** tr(∂fᵢ/∂xᵢ) per row (exact) or its estimate εᵢᵀ(∂fᵢ/∂xᵢ)εᵢ, shape [B]. */
  trace: Value
  /** εᵢᵀ ∂fᵢ/∂xᵢ per row, shape [B, d] (when asked for, or computed by the Hutchinson estimate anyway). */
  probeProduct?: Value
}

/**
 * The trace of the Jacobian of a row-wise map f: [B, d] → [B, d] (row i of the output depends on row i of x only), per
 * row: exactly by d forward-mode products J eₖ (cost d evaluations), or by Hutchinson's unbiased estimator εᵀJε from
 * one reverse-mode product εᵀJ (cost one evaluation, variance Σ_{j≠k} (J_jk² + J_jk J_kj) for Rademacher ε). Written
 * with transforms, so it nests under `grad` (training a CNF differentiates the trace).
 */
export function jacobianTrace(f: (x: Value) => Value, x: Value, options: JacobianTraceOptions = {}): JacobianTrace {
  const { estimator = 'exact', probe, probeProduct = false } = options
  const shape = shapeOfValue(x)
  if (shape.length !== 2) throw new ShapeError('jacobianTrace', 'jacobianTrace: x must be a batch [B, d]')
  const [b, d] = shape
  if ((estimator === 'hutchinson' || probeProduct) && probe === undefined)
    throw new DomainError('jacobianTrace', 'jacobianTrace: the Hutchinson estimate and the probe product need a probe')
  const productOf = () => {
    const { value, pullback } = vjp(f, x)
    return { value: value as Value, product: pullback(probe as never) as Value }
  }
  if (estimator === 'hutchinson') {
    const { value, product } = productOf()
    return { value, trace: sum(mul(product, probe!), 1), probeProduct: product }
  }
  let value: Value | null = null
  let trace: Value = 0
  for (let k = 0; k < d; k++) {
    const e = new Float64Array(b * d)
    for (let i = 0; i < b; i++) e[i * d + k] = 1
    const basis = fromData(e, [b, d])
    const r = jvp(f, x, basis as never)
    value ??= r.value as Value
    trace = add(trace, sum(mul(r.tangent as Value, basis), 1))
  }
  const out: JacobianTrace = { value: value!, trace }
  if (probeProduct) out.probeProduct = productOf().product
  return out
}

// ── Augmented dynamics ───────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link augmentedDynamics}. */
export type AugmentedDynamicsOptions = {
  /** The dimension d of each row of x. */
  dim: Size
  /** Integrate the change in log density, −tr(∂f/∂x), exactly or by Hutchinson's estimator. Default none. */
  logDensity?: TraceEstimator | null
  /** Integrate the kinetic energy ‖f‖² per row (RNODE). Default false. */
  kinetic?: boolean
  /** Integrate ‖εᵀ ∂f/∂x‖², an unbiased estimate of the Jacobian's squared Frobenius norm, per row (RNODE). */
  jacobianFrobenius?: boolean
  /** The probe ε, [B, d], held fixed for the whole solve (FFJORD draws one per solve). */
  probe?: Value
}

/** The parts of an augmented state, each per row. */
export type AugmentedParts = {
  /** x, [B, d]. */
  x: Value
  /** Δ(t) = −∫ tr(∂f/∂x) dt from the start of the solve, [B] (when integrated). */
  logDensityChange?: Value
  /** ∫ ‖f‖² dt, [B]. */
  kinetic?: Value
  /** ∫ ‖εᵀ∂f/∂x‖² dt, [B]. */
  jacobianFrobenius?: Value
}

/** The augmented system and its packing. */
export type AugmentedDynamics = {
  /** The right-hand side on the packed state, for `odeFlow`. */
  rhs: ShapedRhs
  /** The packed state at the start: x with every integral at 0. */
  pack: (x: Value) => Value
  /** The parts of a packed state. */
  unpack: (z: Value) => AugmentedParts
}

/**
 * x′ = f(t, x, θ) for a batch x of shape [B, d], augmented with the integrals a continuous normalising flow and its
 * regularisers need, packed as one vector [x (B·d), Δ (B), kinetic (B), Frobenius (B)] (absent parts omitted). Along
 * x′ = f, the log density obeys d log p(x(t))/dt = −tr(∂f/∂x) (the instantaneous change of variables, Chen et al.,
 * 2018, theorem 1), so log p_{t₁}(x(t₁)) = log p_{t₀}(x(t₀)) + Δ(t₁) with Δ(t₀) = 0, in either direction of time.
 */
export function augmentedDynamics(
  f: (t: Scalar, x: Value, params: Value) => Value,
  options: AugmentedDynamicsOptions,
): AugmentedDynamics {
  const { dim: d, logDensity = null, kinetic = false, jacobianFrobenius = false, probe } = options
  const extras = (logDensity ? 1 : 0) + (kinetic ? 1 : 0) + (jacobianFrobenius ? 1 : 0)
  if (extras === 0)
    throw new DomainError('augmentedDynamics', 'augmentedDynamics: nothing to integrate besides x (use f itself)')
  if ((logDensity === 'hutchinson' || jacobianFrobenius) && probe === undefined)
    throw new DomainError(
      'augmentedDynamics',
      'augmentedDynamics: the Hutchinson estimate and the Frobenius regulariser need a probe',
    )
  const rowsOf = (z: Value) => {
    const total = shapeOfValue(z).reduce((a, b) => a * b, 1)
    const b = total / (d + extras)
    if (!Number.isInteger(b))
      throw new ShapeError('augmentedDynamics', 'augmentedDynamics: the state does not split into rows')
    return b
  }
  const unpack = (z: Value): AugmentedParts => {
    const flat = reshape(z, [shapeOfValue(z).reduce((a, b) => a * b, 1)])
    const b = rowsOf(flat)
    const parts: AugmentedParts = { x: reshape(slice(flat, [0, b * d]), [b, d]) }
    let at = b * d
    const next = () => {
      const v = slice(flat, [at, at + b])
      at += b
      return v
    }
    if (logDensity) parts.logDensityChange = next()
    if (kinetic) parts.kinetic = next()
    if (jacobianFrobenius) parts.jacobianFrobenius = next()
    return parts
  }
  const pack = (x: Value): Value => {
    const [b] = shapeOfValue(x)
    return concat([reshape(x, [b * d]), fromData(new Float64Array(b * extras), [b * extras])])
  }
  const rhs: ShapedRhs = (t, z, params) => {
    const { x } = unpack(z)
    const b = shapeOfValue(x)[0]
    const field = (y: Value) => f(t, y, params)
    let value: Value
    const pieces: Value[] = []
    if (logDensity || jacobianFrobenius) {
      const r = jacobianTrace(field, x, {
        estimator: logDensity ?? 'hutchinson',
        probe,
        probeProduct: jacobianFrobenius,
      })
      value = r.value
      pieces.push(reshape(value, [b * d]))
      if (logDensity) pieces.push(neg(r.trace))
      if (kinetic) pieces.push(sum(square(value), 1))
      if (jacobianFrobenius) pieces.push(sum(square(r.probeProduct!), 1))
    } else {
      value = field(x)
      pieces.push(reshape(value, [b * d]))
      pieces.push(sum(square(value), 1))
    }
    return concat(pieces)
  }
  return { rhs, pack, unpack }
}
