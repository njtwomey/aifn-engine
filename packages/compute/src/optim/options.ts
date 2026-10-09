/**
 * Options and evaluation shared by every method of `aifn-compute/optim` (the family's shared layer): the starting
 * point, the standard stopping options, the options of the one-call runners, and the adapters that turn an
 * `Objective` (a value written with primitives) into the value-and-gradient function the methods call. Stopping rules
 * follow Nocedal & Wright (2006), "Numerical Optimization", 2nd ed., §3: a method has converged when
 * $\lVert \nabla f(\xvec) \rVert_2 \le$ `tolerance`, and has diverged when $f(\xvec)$ or $\xvec$ is not finite or
 * $\lvert f(\xvec) \rvert$ exceeds `divergeAbove`. Both are reported in the state, not thrown.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Stream } from 'aifn-compute/foundation/random'
import { dense, isTensor, isTraced, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type {
  Evaluation,
  Objective,
  ObjectiveFn,
  Scalar,
  Size,
  StoppingOptions,
  ValueFunction,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

export type { StoppingOptions } from 'aifn-compute/foundation/contracts'

/** The starting point passed to `init`: `x0`, the vector $\xvec_0$. */
export type StartOptions = { x0: VectorLike }

/** Options of the one-call runners (`minimize`, `leastSquares`, `linprog`, …): the step budget and the root stream. */
export type RunOptions = {
  /** Stop after at most this many steps. Default 1000 unless the runner says otherwise. */
  maxSteps?: Size
  /** The root stream of stochastic methods. Default `stream(0)`. */
  stream?: Stream
}

/** Default gradient-norm tolerance of `StoppingOptions.tolerance`. */
export const DEFAULT_TOLERANCE = 1e-6
/** Default divergence threshold of `StoppingOptions.divergeAbove`. */
export const DEFAULT_DIVERGE = 1e15

/**
 * The stopping options with their defaults filled in.
 *
 * @param options The caller's stopping options, any of which may be left out.
 * @returns `tolerance` (default `DEFAULT_TOLERANCE`, 1e-6) and `divergeAbove` (default `DEFAULT_DIVERGE`, 1e15).
 */
export function stopping(options: StoppingOptions): Required<StoppingOptions> {
  return { tolerance: options.tolerance ?? DEFAULT_TOLERANCE, divergeAbove: options.divergeAbove ?? DEFAULT_DIVERGE }
}

/**
 * The scalar an objective returned: a number as is, a traced value passed through, or the entry of a one-element
 * tensor; anything else throws `DomainError`.
 *
 * @param v What the objective's value function returned.
 * @param where The objective's name, for error messages.
 * @returns The value as a scalar (a traced value when `v` is traced).
 */
const scalarOf = (v: unknown, where: string): Scalar => {
  if (typeof v === 'number') return v
  // A traced value (an objective evaluated inside a transform, e.g. under `unrolled`) passes through.
  if (isTraced(v)) return v as unknown as Scalar
  if (isTensor(v) && v.shape.reduce((a, b) => a * b, 1) === 1) return toFlat(v)[0]
  throw new DomainError(where, `${where}: the objective must return a scalar`)
}

/**
 * Whether a value is an `Objective` (an object with `kind: 'objective'`) rather than a bare function.
 *
 * @param f The objective or function to test.
 * @returns True for an `Objective`.
 */
const isObjective = (f: unknown): f is Objective<Tensor> =>
  typeof f === 'object' && f !== null && (f as { kind?: unknown }).kind === 'objective'

/**
 * The value-and-gradient function of an objective: an `ObjectiveFn` is returned unchanged; an `Objective` is
 * differentiated by reverse-mode autodiff (`valueAndGrad` of its `value`). The returned function throws `DomainError`
 * when the objective's value is not a scalar.
 *
 * @param f An `Objective`, whose `value` is written with primitives, or a function already returning
 *   `{ value, grad }`.
 * @returns A function from a point $\xvec$ to $f(\xvec)$ and $\nabla f(\xvec)$.
 *
 * @example The gradient of an objective written with primitives
 * // f(x) = x₁² + 10x₂², whose gradient at (1, 1) is (2, 20).
 * const bowl = { kind: 'objective', name: 'bowl', dim: 2, value: (x) => sum(mul(tensor([1, 10]), mul(x, x))) }
 * const { value, grad } = objectiveFn(bowl)(tensor([1, 1]))
 * print('f(1, 1) =', value)
 * print('gradient =', grad)
 */
export function objectiveFn(f: Objective<Tensor> | ObjectiveFn): ObjectiveFn {
  if (!isObjective(f)) return f
  const vg = valueAndGrad((x: Tensor) => f.value(x))
  return (x) => {
    const { value, grad } = vg(x)
    return { value: scalarOf(value, f.name), grad: grad as Tensor }
  }
}

/**
 * The value function of an objective (derivative-free methods): `f.value` for an `Objective`, else `f` itself.
 *
 * @param f An `Objective`, a function returning `{ value, grad }`, or a function returning the value.
 * @returns A function from a point to $f(\xvec)$; for a bare function, `f` itself, so it may still return an object
 *   with a `value`.
 */
export function valueFunction(f: Objective<Tensor> | ObjectiveFn | ValueFunction): ValueFunction {
  if (!isObjective(f)) return f as ValueFunction
  return (x) => scalarOf(f.value(x), f.name)
}

/**
 * $f(\xvec)$ and $\nabla f(\xvec)$ from an objective, checked for shape. An objective that returns no gradient throws
 * `DomainError`; a gradient whose length is not that of $\xvec$ throws `ShapeError`.
 *
 * @param f The objective, returning `{ value, grad }`.
 * @param x The point, as a float64 working array, passed to the objective wrapped as a vector (not copied).
 * @param where The caller's name, for error messages.
 * @returns The value and the gradient as a float64 working array.
 */
export function evaluate(f: ObjectiveFn, x: dense.F64, where: string): { value: Scalar; grad: dense.F64 } {
  const out = f(dense.vec(x)) as Evaluation | number
  if (typeof out === 'number' || out.grad === undefined)
    throw new DomainError(where, `${where}: the objective must return { value, grad }; this method uses the gradient`)
  const grad = dense.toF64(out.grad, where)
  if (grad.length !== x.length)
    throw new ShapeError(where, `${where}: the gradient has length ${grad.length}, but x has length ${x.length}`)
  return { value: out.value, grad }
}

/**
 * $f(\xvec)$ from a value function or an objective function.
 *
 * @param f A function returning the value, or an object with a `value` (such as `{ value, grad }`).
 * @param x The point, as a float64 working array.
 * @returns The value.
 */
export function evaluateValue(f: ValueFunction | ObjectiveFn, x: dense.F64): Scalar {
  const out = f(dense.vec(x)) as number | { value: number }
  return typeof out === 'number' ? out : out.value
}

/**
 * The divergence test shared by the optimisers: anything non-finite, or $\lvert f \rvert$ above the threshold.
 *
 * @param value $f(\xvec)$.
 * @param x The iterate $\xvec$, every entry of which must be finite.
 * @param limit The threshold, `divergeAbove`.
 * @returns True when the value or an entry of $\xvec$ is not finite, or $\lvert f(\xvec) \rvert$ exceeds `limit`.
 */
export function divergedAt(value: Scalar, x: ArrayLike<number>, limit: Scalar): boolean {
  return !Number.isFinite(value) || Math.abs(value) > limit || !dense.allFinite(x)
}
