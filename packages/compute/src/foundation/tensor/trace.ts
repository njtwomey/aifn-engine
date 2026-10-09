/**
 * The hook between primitives and the function transforms of `aifn-compute/foundation/autodiff` (design K §4.1). A
 * transform (grad, jvp, vmap, …) runs a function on **tracers**: values that stand for a number or tensor and belong to
 * one **interpreter** at one **level**. Every primitive goes through `apply`: with no tracer among its inputs it runs
 * its forward rule directly (the fast path, which is how most of aifn runs); otherwise the tracer of the highest level
 * hands the application to its interpreter, which treats tracers of lower levels as constants.
 *
 * This file defines only the protocol. The three interpreters (reverse, forward, batch) and the transforms live in
 * `aifn-compute/foundation/autodiff`. The design is JAX's (Bradbury et al., 2018): primitives with rules, and
 * interpreters that nest by level, which rules out perturbation confusion by construction (Siskind and Pearlmutter,
 * 2005).
 */

import type { Aval, Traced, TracedBrand, Value } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import type { Primitive, Raw } from './registry'

export type { Aval, Traced, Value } from 'aifn-compute/foundation/contracts'

/** The brand every tracer carries (on `Tracer.prototype`); registered with `Symbol.for` so that copies agree. */
const TRACED: TracedBrand = Symbol.for('aifn.traced') as TracedBrand

/**
 * An interpreter of one transform in progress. `process` applies primitive `p` to inputs at least one of which is a
 * tracer of this interpreter: it lowers its own tracers to the values they stand for, computes (by `apply`, so lower
 * levels see the application too) and wraps the result.
 */
export interface Interpreter {
  /** The level of its transform, from `nextLevel`: higher for a transform started inside another. */
  readonly level: number
  /** `reverse`, `forward` or `batch`, for messages and graphs. */
  readonly kind: string
  /** Apply `p` to `inputs` (at least one a tracer of this interpreter) and `params`, returning a traced result. */
  process(p: Primitive<unknown>, inputs: readonly Value[], params: unknown): Value
}

/**
 * A traced value of one interpreter. The concrete tracers are defined with their interpreters. A transform such as
 * `grad` hands the function it differentiates a tracer in place of each argument.
 *
 * @example The argument a transform passes in
 * const g = grad((x) => {
 *   print('a Tracer:', x instanceof Tracer)
 *   print('interpreter:', x.interpreter.kind, 'at level', x.level)
 *   print('aval:', x.aval)
 *   return mul(x, x)
 * })
 * print('gradient at 3:', g(3))
 */
export abstract class Tracer implements Traced {
  declare readonly [TRACED]: true
  /** The level of its interpreter. */
  abstract readonly level: number
  /** The abstract value (shape, dtype, kind) it stands for. */
  abstract readonly aval: Aval
  /** The interpreter it belongs to, which processes every primitive applied to it. */
  abstract readonly interpreter: Interpreter
  /** The value one level down that this tracer stands for (its primal); a batch tracer has none and throws. */
  abstract primal(): Value
}
Object.defineProperty(Tracer.prototype, TRACED, { value: true })

/**
 * True for a traced value (every traced value is a `Tracer` of some interpreter).
 *
 * @param x Any value.
 * @returns Whether `x` carries the tracer brand.
 *
 * @example Inside and outside a transform
 * print('a number:', isTraced(3))
 * print('a tensor:', isTraced(tensor([1, 2])))
 * grad((x) => {
 *   print('inside grad:', isTraced(x))
 *   return x
 * })(1)
 */
export function isTraced(x: unknown): x is Traced {
  return typeof x === 'object' && x !== null && (x as Record<symbol, unknown>)[TRACED] === true
}

/** The last level handed out by `nextLevel`. */
let levels = 0

/**
 * A fresh level for a transform starting now. Levels only grow, so a transform started inside another (later) has
 * the higher level, and a tracer from a finished transform can never be mistaken for one of a live transform.
 *
 * @returns A level higher than every one handed out before.
 *
 * @example Each call is higher than the last
 * const a = nextLevel()
 * const b = nextLevel()
 * print('a =', a, 'b =', b, 'b > a:', b > a)
 */
export function nextLevel(): number {
  return ++levels
}

/**
 * Apply primitive `p` to `inputs` (numbers, tensors or tracers) and its parameters. With no tracer this is `p.impl`;
 * otherwise the interpreter of the highest-level tracer processes it.
 *
 * @param p The primitive to apply.
 * @param inputs Its inputs, in order; any of them may be traced.
 * @param params Its non-differentiable parameters (axes, shapes, options), passed through unchanged.
 * @returns The result of `p.impl` when no input is traced, else the interpreter's (traced) result.
 *
 * @example Apply a registered primitive, directly and under `grad`
 * const exp_ = registry.get('foundation/tensor/exp')
 * print('exp([0, 1]) =', apply(exp_, [tensor([0, 1])], {}))
 * print('d/dx exp(x) at 0 =', grad((x) => apply(exp_, [x], {}))(0))
 */
export function apply<P>(p: Primitive<P>, inputs: readonly Value[], params: P): Value {
  let top: Tracer | null = null
  for (let k = 0; k < inputs.length; k++) {
    const x = inputs[k]
    if (typeof x === 'object' && isTraced(x) && (top === null || x.level > top.level)) top = x as Tracer
  }
  return top === null
    ? p.impl(inputs as Raw[], params)
    : top.interpreter.process(p as Primitive<unknown>, inputs, params)
}

/** The abstract value of every plain number: a float64 scalar that is a number, not a tensor. */
const NUMBER: Aval = { shape: [], dtype: 'float64', number: true }

/**
 * The abstract value (shape, dtype, kind) of a value; for a tracer, the value the traced function sees.
 *
 * @param x A number, a tensor or a traced value.
 * @returns Its shape, its dtype and whether it is a plain number (`number`); a number is a float64 scalar.
 *
 * @example Abstract values of a number and a tensor
 * print('number:', avalOf(2))
 * print('tensor:', avalOf(tensor([[1, 2, 3]], undefined, 'int32')))
 */
export function avalOf(x: Value): Aval {
  if (typeof x === 'number') return NUMBER
  if (isTraced(x)) return x.aval
  return { shape: x.shape, dtype: x.dtype, number: false }
}

/**
 * The concrete number or tensor behind a value: the value itself when it is raw, otherwise its primal, followed down
 * through every level. Inside `vmap` a value has no single concrete value, so reading one is an error: write the code
 * with primitives instead.
 *
 * @param x A number, a tensor or a traced value.
 * @returns The untraced number or tensor it stands for.
 *
 * @example The value behind a tracer
 * grad((x) => {
 *   print('x is traced:', isTraced(x))
 *   print('unwrap(x) =', unwrap(x))
 *   return mul(x, x)
 * })(3)
 * print('a raw value is returned as it is:', unwrap(5))
 */
export function unwrap(x: Value): Raw {
  let v: Value = x
  while (typeof v === 'object' && isTraced(v)) v = (v as Tracer).primal()
  return v
}

/**
 * The error raised by a batch tracer asked for its concrete value.
 *
 * @param where The caller's name, for the error message (and the error's operation).
 * @returns The error, to be thrown by the caller.
 *
 * @example The message a caller throws
 * print(batchedValueError('item').message)
 */
export function batchedValueError(where = 'unwrap'): AifnError {
  return new AifnError(
    where,
    `${where}: a value inside vmap has no single concrete value (it holds a whole batch); compute with primitives instead`,
  )
}
