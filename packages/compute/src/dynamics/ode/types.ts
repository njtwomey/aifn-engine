/**
 * Types shared by every initial-value solver in `aifn-compute/dynamics/ode` (Hairer, Nørsett & Wanner, 1993, "Solving
 * Ordinary Differential Equations I", §II.1, for the conventions).
 */

import type { Tensor, Value, Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'

/**
 * The right-hand side of x′ = f(t, x): time t and state x (a rank-1 tensor of length n) to the derivative (length
 * n). Solvers that need the Jacobian ∂f/∂x by automatic differentiation call f with a traced x, so f must then be
 * written with `aifn-compute/foundation/tensor` primitives (`get`, `stack`, `mul`, `sin`, …) rather than by reading `x.data`.
 * The result may be a tensor, a plain array or (from primitives) a `Value`; outside autodiff it must be a length-n
 * vector.
 */
export type Rhs = (t: Scalar, x: Tensor) => VectorLike | Value

/** How an implicit solver obtains ∂f/∂x: by `aifn-compute/foundation/autodiff` (default), by forward differences, or a given function. */
export type JacobianOption = 'autodiff' | 'finite-difference' | ((t: Scalar, x: Tensor) => MatrixLike)

/** The initial value passed to `init`: x(t₀) = x₀, at the initial time `t0` (default 0). */
export type InitialValue = { x0: VectorLike; t0?: Scalar }

/**
 * Fields every solver state carries. `t` counts accepted steps (the runner's `Status`); the solution's time is
 * `time`.
 */
export interface OdeState extends Status {
  /** Accepted steps so far (0 in the initial state). */
  t: Size
  /** The current time. */
  time: Scalar
  /** The current state x(time) (length n). */
  x: Vector
  /** The size of the last accepted step (0 at t₀). */
  stepSize: Scalar
  /** The estimated local error of the last step in the solver's error norm, or NaN for methods without an estimate. */
  error: Scalar
  /** Evaluations of f so far. */
  evaluations: Size
  /** Evaluations of the Jacobian ∂f/∂x so far (implicit methods). */
  jacobianEvaluations: Size
  /** Step attempts rejected by error control so far (adaptive methods). */
  rejected: Size
  /** True when the state stopped being finite or a step failed (see `failure`). */
  diverged: boolean
  /** Why the solver cannot continue, or null: `'not finite'`, `'newton failed'`, `'step size underflow'`. */
  failure: string | null
}

/** Options common to the fixed-step solvers. */
export type FixedStepOptions = {
  /** The step size h (negative integrates backwards in time). */
  stepSize: Scalar
  /** Stop on reaching this time; the last step is shortened to land on it exactly. Default: never (run n steps). */
  tEnd?: Scalar
}
