/**
 * Differentiable ODE solves for neural ODEs and continuous normalising flows (Chen, Rubanova, Bettencourt & Duvenaud,
 * 2018, "Neural ordinary differential equations", NeurIPS; Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud, 2019,
 * "FFJORD", ICLR; Finlay, Jacobsen, Nurbekyan & Oberman, 2020, "How to train your neural ODE", ICML).
 *
 * - `odeFlow`: the states $\xvec(t_1), \dots, \xvec(t_m)$ of $\xvec' = f(t, \xvec, \thetavec)$ from $\xvec(t_0)$,
 *   differentiable with respect to $\xvec(t_0)$ and $\thetavec$ either by backpropagation through the solver's steps
 *   (discretise-then-optimise: the exact gradient of the discrete solution, memory linear in the steps) or by the
 *   adjoint method (optimise-then-discretise: one backward solve, memory constant in the steps, the gradient of the
 *   exact flow to the solver's accuracy). Work is reported per solve.
 * - `jacobianTrace`: $\trace(\partial f/\partial\xvec)$ for each row of a batch, exactly (one jvp per dimension) or
 *   by Hutchinson's estimator $\epsilonvec^\top (\partial f/\partial\xvec) \epsilonvec$ with a Rademacher or Gaussian
 *   probe $\epsilonvec$ (Hutchinson, 1989; drawn by `traceProbe`), from one vjp.
 * - `augmentedDynamics`: $f$ with the integrals a CNF and its regularisers need appended to the state: the change in
 *   log density, $d\Delta/dt = -\trace(\partial f/\partial\xvec)$ (the instantaneous change of variables), the
 *   kinetic energy $\lVert f \rVert^2$ and the Jacobian's Frobenius norm
 *   $\lVert \epsilonvec^\top \partial f/\partial\xvec \rVert^2$ (RNODE).
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
  /**
   * The step size of a fixed-step method (its magnitude, positive, or `DomainError` is thrown; the sign follows each
   * interval). Default 0.1.
   */
  stepSize?: Scalar
  /** The relative tolerance of Dormand–Prince (default 1e-3). */
  rtol?: Scalar
  /** The absolute tolerance of Dormand–Prince (default 1e-6). */
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

/**
 * A right-hand side on a state of any shape: $\xvec' = f(t, \xvec, \thetavec)$ with $f(t, \xvec, \thetavec)$ shaped
 * like $\xvec$.
 */
export type ShapedRhs = (t: Scalar, x: Value, params: Value) => Value

/**
 * The values of a number or a (possibly traced) tensor as a flat array, read from the primal value.
 *
 * @param v The number or tensor.
 * @returns Its entries in row-major order (one entry for a number).
 */
const flatOf = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))

/**
 * The flow map of $\xvec' = f(t, \xvec, \thetavec)$ sampled at `times`: a function of $\xvec(t_0)$ (any shape) and
 * $\thetavec$ returning $[\xvec(t_0), \xvec(t_1), \dots, \xvec(t_m)]$, each shaped like $\xvec(t_0)$. Each interval
 * $[t_{i-1}, t_i]$ is solved in its own direction (increasing or decreasing). Under `grad` it is differentiated by
 * `gradient`: `'backprop'` unrolls the solver on traced values (`unrolled`; Dormand–Prince chooses its steps on primal
 * values, so they are constants of the discrete solution), `'adjoint'` uses `odeAdjoint` on each interval. Without a
 * transform it simply solves. A solve that fails or needs more than `maxSteps` steps throws `NumericalError`
 * 'not-converged'; fewer than two times, a time that is not finite or equals the one before it, or a step size that is
 * not positive throws `DomainError`.
 *
 * @param f The right-hand side $f(t, \xvec, \thetavec)$, called with $\xvec$ in the shape of $\xvec(t_0)$ and
 *   returning the derivative in that shape. It is written with tensor primitives, so that it can be differentiated.
 * @param times The times $t_0, t_1, \dots, t_m$ to sample: at least two, finite, each different from the one before.
 * @param options The gradient method, the solver and its step size or tolerances, the step limit, the adjoint's
 *   checkpoints and a callback reporting the work of each solve.
 * @returns The map from $\xvec(t_0)$ and $\thetavec$ to the list of states, one per time, the first being
 *   $\xvec(t_0)$ itself.
 *
 * @example States of a decay at three times, and their gradient both ways
 * const f = (t, x, k) => mul(neg(k), x)
 * print('x(0), x(0.5), x(1) =', odeFlow(f, [0, 0.5, 1])(tensor([1, 2]), 0.5))
 * for (const gradient of ['backprop', 'adjoint']) {
 *   const flow = odeFlow(f, [0, 1], { gradient })
 *   print(`${gradient}: d sum x(1) / dk =`, grad((k) => sum(flow(tensor([1, 2]), k)[1]))(0.5))
 * }
 * print('exact -3 e^-0.5 =', -3 * Math.exp(-0.5))
 *
 * @example The work of a gradient by each method
 * const f = (t, x, k) => mul(neg(k), x)
 * for (const gradient of ['backprop', 'adjoint']) {
 *   const solves = []
 *   const flow = odeFlow(f, [0, 1], { gradient, method: 'dormand-prince', onSolve: (s) => solves.push(s) })
 *   grad((k) => sum(flow(tensor([1, 2]), k)[1]))(0.5)
 *   print(`${gradient}:`, solves.map((s) => `${s.phase} ${s.steps} steps, ${s.evaluations} evaluations`))
 * }
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
  for (let i = 0; i < times.length; i++)
    if (!(Number.isFinite(times[i]) && (i === 0 || times[i] !== times[i - 1])))
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

/**
 * The probe distributions of Hutchinson's estimator: both have $\expect[\epsilonvec\epsilonvec^\top] = \Imat$;
 * Rademacher has the lower variance.
 */
export type ProbeKind = 'rademacher' | 'gaussian'

/**
 * A probe for Hutchinson's estimator: independent signs $\pm 1$ with equal probability (Rademacher) or standard
 * normals, drawn from the stream (which advances).
 *
 * @param s The stream to draw from.
 * @param shape The shape of the probe, that of the batch it multiplies ($B \times d$ for `jacobianTrace`).
 * @param kind `'rademacher'` or `'gaussian'`.
 * @returns A tensor of the given shape.
 *
 * @example Rademacher and Gaussian probes
 * const s = stream(0)
 * print('rademacher =', traceProbe(s, [2, 3]))
 * print('gaussian =', traceProbe(s, [2, 3], 'gaussian'))
 */
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
  /**
   * `'exact'` (default): one forward-mode product per dimension. `'hutchinson'`: one reverse-mode product with
   * `probe`.
   */
  estimator?: TraceEstimator
  /**
   * The probe $\epsilonvec$, shaped like $\xvec$ (required by `'hutchinson'`, and by `'exact'` when `probeProduct` is
   * wanted).
   */
  probe?: Value
  /**
   * Also return $\epsilonvec^\top \partial f/\partial\xvec$ per row (for the Jacobian Frobenius regulariser).
   * Default false.
   */
  probeProduct?: boolean
}

/** The result of {@link jacobianTrace}. */
export type JacobianTrace = {
  /** $f(\xvec)$, shape $B \times d$. */
  value: Value
  /**
   * $\trace(\partial f_i/\partial\xvec_i)$ per row $i$ (exact) or its estimate
   * $\epsilonvec_i^\top (\partial f_i/\partial\xvec_i) \epsilonvec_i$, $B$ values.
   */
  trace: Value
  /**
   * $\epsilonvec_i^\top \partial f_i/\partial\xvec_i$ per row, shape $B \times d$ (when asked for, or computed by the
   * Hutchinson estimate anyway).
   */
  probeProduct?: Value
}

/**
 * The trace of the Jacobian of a row-wise map $f: \reals^{B \times d} \to \reals^{B \times d}$ (row $i$ of the output
 * depends on row $i$ of $\xvec$ only), per row: exactly by $d$ forward-mode products $\Jmat\evec_k$ (cost $d$
 * evaluations), or by Hutchinson's unbiased estimator $\epsilonvec^\top\Jmat\epsilonvec$ from one reverse-mode
 * product $\epsilonvec^\top\Jmat$ (cost one evaluation, variance $\sum_{j \ne k} (J_{jk}^2 + J_{jk} J_{kj})$ for
 * Rademacher $\epsilonvec$). The exact trace with `probeProduct` costs one reverse-mode product more. Written with
 * transforms, so it nests under `grad` (training a CNF differentiates the trace). An $\xvec$ that is not a matrix
 * throws `ShapeError`; the Hutchinson estimate or the probe product without a `probe` throws `DomainError`.
 *
 * @param f The row-wise map, written with tensor primitives; it takes and returns a $B \times d$ batch.
 * @param x The batch $\xvec$, $B \times d$, at which the Jacobian is taken.
 * @param options The estimator, the probe and whether to return the probe product.
 * @returns $f(\xvec)$, the trace (or estimate) per row and, when computed, the probe product.
 *
 * @example An elementwise map: the Hutchinson estimate is exact for a diagonal Jacobian
 * const f = (x) => mul(x, x)
 * const x = tensor([[1, 2], [3, 4]])
 * print('exact =', jacobianTrace(f, x).trace)
 * const probe = traceProbe(stream(0), [2, 2])
 * print('hutchinson =', jacobianTrace(f, x, { estimator: 'hutchinson', probe }).trace)
 *
 * @example A coupled map: each estimate is noisy, their mean is the trace
 * // Every row maps by A = [[1, 2], [3, 4]], whose trace is 5.
 * const f = (x) => matmul(x, transpose(tensor([[1, 2], [3, 4]])))
 * const x = normals(stream(1), [1000, 2])
 * print('exact, first rows =', slice(jacobianTrace(f, x).trace, [0, 3]))
 * const { trace } = jacobianTrace(f, x, { estimator: 'hutchinson', probe: traceProbe(stream(2), [1000, 2]) })
 * print('hutchinson, first rows =', slice(trace, [0, 3]))
 * print('hutchinson, mean of 1000 =', mean(trace))
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
  /** The dimension $d$ of each row of $\xvec$. */
  dim: Size
  /**
   * Integrate the change in log density, $-\trace(\partial f/\partial\xvec)$, exactly (`'exact'`) or by
   * Hutchinson's estimator (`'hutchinson'`, which needs `probe`). Default null: not integrated.
   */
  logDensity?: TraceEstimator | null
  /** Integrate the kinetic energy $\lVert f \rVert^2$ per row (RNODE). Default false. */
  kinetic?: boolean
  /**
   * Integrate $\lVert \epsilonvec^\top \partial f/\partial\xvec \rVert^2$, an unbiased estimate of the Jacobian's
   * squared Frobenius norm, per row (RNODE; needs `probe`). Default false.
   */
  jacobianFrobenius?: boolean
  /** The probe $\epsilonvec$, $B \times d$, held fixed for the whole solve (FFJORD draws one per solve). */
  probe?: Value
}

/** The parts of an augmented state, each per row. */
export type AugmentedParts = {
  /** $\xvec$, $B \times d$. */
  x: Value
  /**
   * $\Delta(t) = -\int \trace(\partial f/\partial\xvec) \, dt$ from the start of the solve, $B$ values (when
   * integrated).
   */
  logDensityChange?: Value
  /** $\int \lVert f \rVert^2 \, dt$, $B$ values (when integrated). */
  kinetic?: Value
  /** $\int \lVert \epsilonvec^\top \partial f/\partial\xvec \rVert^2 \, dt$, $B$ values (when integrated). */
  jacobianFrobenius?: Value
}

/** The augmented system and its packing. */
export type AugmentedDynamics = {
  /** The right-hand side on the packed state, for `odeFlow`. */
  rhs: ShapedRhs
  /** The packed state at the start, from $\xvec$ ($B \times d$): $\xvec$ flattened, with every integral at 0. */
  pack: (x: Value) => Value
  /** The parts of a packed state (of any shape holding $B (d + \text{extras})$ values). */
  unpack: (z: Value) => AugmentedParts
}

/**
 * $\xvec' = f(t, \xvec, \thetavec)$ for a batch $\xvec$ of shape $B \times d$, augmented with the integrals a
 * continuous normalising flow and its regularisers need, packed as one vector: $\xvec$ ($Bd$ values), then $\Delta$,
 * the kinetic energy and the Frobenius term ($B$ values each; absent parts omitted). Along $\xvec' = f$, the log
 * density obeys $\frac{d}{dt} \log p(\xvec(t)) = -\trace(\partial f/\partial\xvec)$ (the instantaneous change of
 * variables, Chen et al., 2018, theorem 1), so $\log p_{t_1}(\xvec(t_1)) = \log p_{t_0}(\xvec(t_0)) + \Delta(t_1)$
 * with $\Delta(t_0) = 0$, in either direction of time. Asking for no integral, or for the Hutchinson estimate or the
 * Frobenius term without a probe, throws `DomainError`; unpacking a state that does not split into rows throws
 * `ShapeError`.
 *
 * @param f The field $f(t, \xvec, \thetavec)$ on the batch: row-wise, taking and returning $B \times d$, written with
 *   tensor primitives (its Jacobian trace is taken by `jacobianTrace`).
 * @param options The row dimension $d$, which integrals to append, and the probe the Hutchinson estimate and the
 *   Frobenius term need.
 * @returns The packed right-hand side for `odeFlow`, with `pack` and `unpack` to move between $\xvec$ and the packed
 *   state.
 *
 * @example The log density and kinetic energy of a contraction
 * // x′ = −x on two rows of d = 2: tr(∂f/∂x) = −2, so Δ(1) = 2; ∫‖f‖² dt = ‖x₀‖² (1 − e⁻²) / 2.
 * const aug = augmentedDynamics((t, x, theta) => mul(neg(theta), x), { dim: 2, logDensity: 'exact', kinetic: true })
 * const [, end] = odeFlow(aug.rhs, [0, 1])(aug.pack(tensor([[1, 0], [0, 2]])), 1)
 * const parts = aug.unpack(end)
 * print('x(1) =', parts.x)
 * print('Δ(1) =', parts.logDensityChange)
 * print('kinetic =', parts.kinetic, ' exact =', [1, 4].map((r) => (r * (1 - Math.exp(-2))) / 2))
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
