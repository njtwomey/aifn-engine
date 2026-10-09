/**
 * Objectives: functions to minimise (design S §2.6). An `Objective` is a displayable object whose `value` is written
 * with primitives, so `grad(value)` works. `ObjectiveFn` is the bare function that supplies its own gradient, which
 * the optimisers also accept.
 */

import type { Kinded } from './kinds'
import type { Info } from './registry'
import type { Status } from './algorithm'
import type { Index, MatrixLike, Scalar, Size, Tensor, Value, Vector, VectorLike } from './numbers'
import type { Space } from './space'

/**
 * What a differentiable objective returns at a point: its `value` and its gradient `grad` (same length as the point).
 */
export type Evaluation = { value: Scalar; grad: VectorLike }

/**
 * $f: \reals^n \to \reals$ as a bare function supplying its own gradient, `(x) => ({ value, grad })`. The point `x`
 * is a float64 vector of length $n$ and must not be mutated.
 */
export type ObjectiveFn = (x: Vector) => Evaluation

/**
 * A function to minimise (`kind: 'objective'`). `dim` is the length of `x`, or null when `x` is a pytree. `truth`
 * gives the known minimisers of a test surface (Rosenbrock, Himmelblau) so a figure can draw the optimum.
 */
export interface Objective<X = Tensor> extends Kinded<'objective'> {
  /** A readable name, for display. */
  readonly name: string
  /** The length of the point `x`, or null when `x` is a pytree. */
  readonly dim: Size | null
  /** $f(x)$, built from primitives, so `grad` differentiates it. */
  value(x: X): Value
  /** Where `x` may lie: a parameter space, or elementwise bounds `lower` and `upper`. */
  readonly domain?: Space | { readonly lower: Tensor; readonly upper: Tensor }
  /** The known optimum: the global `minimisers` and the `minimum` value of $f$. */
  readonly truth?: { readonly minimisers: Tensor; readonly minimum: Scalar }
}

/**
 * Registry metadata of a named objective (a test surface): its parameters, its dimension (null when the
 * dimension is a parameter) and whether its known minimisers are given (`truth`).
 */
export interface ObjectiveInfo extends Info {
  /** The entry kind of a named objective. */
  readonly kind: 'objective'
  /** Its parameters, with their defaults. */
  readonly params: Space
  /** The length of the point, or null when it is a parameter. */
  readonly dim: Size | null
  /** True when its objectives carry their known minimisers (`truth`). */
  readonly truth: boolean
}

// ── Optimiser signatures and states ──────────────────────────────────────────────────────────────────────────────────

/** A function to minimise without derivatives: returns $f(\xvec)$, or an `Evaluation` whose gradient is ignored. */
export type ValueFunction = (x: Vector) => Scalar | { value: Scalar }

/** The Hessian $\nabla^2 f(\xvec)$ as an $n \times n$ matrix (symmetric; only the values matter, not the strides). */
export type Hessian = (x: Vector) => MatrixLike

/** A step-size schedule: the step size (learning rate) used on step $t = 0, 1, 2, \dots$ */
export type Schedule = (t: Index) => Scalar

/**
 * Fields every optimiser state carries. `x` is the current iterate and `value` is $f(\xvec)$; `evaluations` counts
 * calls of the objective so far (including the initial one). `converged` is set when the method's stopping test passes
 * and `diverged` when the value or iterate is not finite or exceeds the divergence threshold; either stops the runners.
 */
export interface IterateState extends Status {
  /** The current iterate $\xvec$. */
  x: Vector
  /** $f(\xvec)$ at the current iterate. */
  value: Scalar
  /** Calls of the objective so far, the initial one included. */
  evaluations: Size
  /** The method's stopping test has passed. */
  converged: boolean
  /** The value or iterate is not finite, or the value exceeds `divergeAbove`. */
  diverged: boolean
}

/** Options shared by the gradient-based methods. */
export type StoppingOptions = {
  /** Stop (converged) when $\lVert \nabla f(\xvec) \rVert_2 \le$ `tolerance`. Default 1e-6. */
  tolerance?: Scalar
  /** Flag divergence when $\lvert f(\xvec) \rvert$ exceeds this (or anything is not finite). Default 1e15. */
  divergeAbove?: Scalar
}
