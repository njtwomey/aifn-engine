/**
 * Options and evaluation shared by every method of `aifn-compute/optim` (the family's shared layer): the starting point, the
 * standard stopping options, the options of the one-call runners, and the adapters that turn an `Objective` (a value
 * written with primitives) into the value-and-gradient function the methods call. Stopping rules follow Nocedal &
 * Wright (2006), "Numerical Optimization", 2nd ed., §3.
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

/** The starting point x₀ passed to `init`. */
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

/** The stopping options with their defaults filled in. */
export function stopping(options: StoppingOptions): Required<StoppingOptions> {
  return { tolerance: options.tolerance ?? DEFAULT_TOLERANCE, divergeAbove: options.divergeAbove ?? DEFAULT_DIVERGE }
}

const scalarOf = (v: unknown, where: string): Scalar => {
  if (typeof v === 'number') return v
  // A traced value (an objective evaluated inside a transform, e.g. under `unrolled`) passes through.
  if (isTraced(v)) return v as unknown as Scalar
  if (isTensor(v) && v.shape.reduce((a, b) => a * b, 1) === 1) return toFlat(v)[0]
  throw new DomainError(where, `${where}: the objective must return a scalar`)
}

const isObjective = (f: unknown): f is Objective<Tensor> =>
  typeof f === 'object' && f !== null && (f as { kind?: unknown }).kind === 'objective'

/**
 * The value-and-gradient function of an objective: an `ObjectiveFn` is returned unchanged; an `Objective` is
 * differentiated by reverse-mode autodiff (`valueAndGrad` of its `value`).
 */
export function objectiveFn(f: Objective<Tensor> | ObjectiveFn): ObjectiveFn {
  if (!isObjective(f)) return f
  const vg = valueAndGrad((x: Tensor) => f.value(x))
  return (x) => {
    const { value, grad } = vg(x)
    return { value: scalarOf(value, f.name), grad: grad as Tensor }
  }
}

/** The value function of an objective (derivative-free methods): `f.value` for an `Objective`, else `f` itself. */
export function valueFunction(f: Objective<Tensor> | ObjectiveFn | ValueFunction): ValueFunction {
  if (!isObjective(f)) return f as ValueFunction
  return (x) => scalarOf(f.value(x), f.name)
}

/** f(x) and ∇f(x) from an objective, checked for shape. */
export function evaluate(f: ObjectiveFn, x: dense.F64, where: string): { value: Scalar; grad: dense.F64 } {
  const out = f(dense.vec(x)) as Evaluation | number
  if (typeof out === 'number' || out.grad === undefined)
    throw new DomainError(where, `${where}: the objective must return { value, grad }; this method uses the gradient`)
  const grad = dense.toF64(out.grad, where)
  if (grad.length !== x.length)
    throw new ShapeError(where, `${where}: the gradient has length ${grad.length}, but x has length ${x.length}`)
  return { value: out.value, grad }
}

/** f(x) from a value function or an objective function. */
export function evaluateValue(f: ValueFunction | ObjectiveFn, x: dense.F64): Scalar {
  const out = f(dense.vec(x)) as number | { value: number }
  return typeof out === 'number' ? out : out.value
}

/** The divergence test shared by the optimisers: anything non-finite, or |f| above the threshold. */
export function divergedAt(value: Scalar, x: ArrayLike<number>, limit: Scalar): boolean {
  return !Number.isFinite(value) || Math.abs(value) > limit || !dense.allFinite(x)
}
