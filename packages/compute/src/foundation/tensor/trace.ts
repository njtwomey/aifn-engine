/**
 * The hook between primitives and the function transforms of `aifn-compute/foundation/autodiff` (design K §4.1). A transform
 * (grad, jvp, vmap, …) runs a function on **tracers**: values that stand for a number or tensor and belong to one
 * **interpreter** at one **level**. Every primitive goes through `apply`: with no tracer among its inputs it runs its
 * forward rule directly (the fast path, which is how most of aifn runs); otherwise the tracer of the highest level
 * hands the application to its interpreter, which treats tracers of lower levels as constants.
 *
 * This file defines only the protocol. The three interpreters (reverse, forward, batch) and the transforms live in
 * `aifn-compute/foundation/autodiff`. The design is JAX's (Bradbury et al., 2018): primitives with rules, and interpreters that
 * nest by level, which rules out perturbation confusion by construction (Siskind and Pearlmutter, 2005).
 */

import type { Aval, Traced, TracedBrand, Value } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import type { Primitive, Raw } from './registry'

export type { Aval, Traced, Value } from 'aifn-compute/foundation/contracts'

const TRACED: TracedBrand = Symbol.for('aifn.traced') as TracedBrand

/**
 * An interpreter of one transform in progress. `process` applies primitive `p` to inputs at least one of which is a
 * tracer of this interpreter: it lowers its own tracers to the values they stand for, computes (by `apply`, so lower
 * levels see the application too) and wraps the result.
 */
export interface Interpreter {
  readonly level: number
  /** `reverse`, `forward` or `batch`, for messages and graphs. */
  readonly kind: string
  process(p: Primitive<unknown>, inputs: readonly Value[], params: unknown): Value
}

/** A traced value of one interpreter. The concrete tracers are defined with their interpreters. */
export abstract class Tracer implements Traced {
  declare readonly [TRACED]: true
  abstract readonly level: number
  abstract readonly aval: Aval
  abstract readonly interpreter: Interpreter
  /** The value one level down that this tracer stands for (its primal); a batch tracer has none and throws. */
  abstract primal(): Value
}
Object.defineProperty(Tracer.prototype, TRACED, { value: true })

/** True for a traced value (every traced value is a `Tracer` of some interpreter). */
export function isTraced(x: unknown): x is Traced {
  return typeof x === 'object' && x !== null && (x as Record<symbol, unknown>)[TRACED] === true
}

let levels = 0

/**
 * A fresh level for a transform starting now. Levels only grow, so a transform started inside another (later) has
 * the higher level, and a tracer from a finished transform can never be mistaken for one of a live transform.
 */
export function nextLevel(): number {
  return ++levels
}

/**
 * Apply primitive `p` to `inputs` (numbers, tensors or tracers) and its parameters. With no tracer this is `p.impl`;
 * otherwise the interpreter of the highest-level tracer processes it.
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

const NUMBER: Aval = { shape: [], dtype: 'float64', number: true }

/** The abstract value (shape, dtype, kind) of a value; for a tracer, the value the traced function sees. */
export function avalOf(x: Value): Aval {
  if (typeof x === 'number') return NUMBER
  if (isTraced(x)) return x.aval
  return { shape: x.shape, dtype: x.dtype, number: false }
}

/**
 * The concrete number or tensor behind a value: the value itself when it is raw, otherwise its primal, followed down
 * through every level. Inside `vmap` a value has no single concrete value, so reading one is an error: write the code
 * with primitives instead.
 */
export function unwrap(x: Value): Raw {
  let v: Value = x
  while (typeof v === 'object' && isTraced(v)) v = (v as Tracer).primal()
  return v
}

/** The error raised by a batch tracer asked for its concrete value. */
export function batchedValueError(where = 'unwrap'): AifnError {
  return new AifnError(
    where,
    `${where}: a value inside vmap has no single concrete value (it holds a whole batch); compute with primitives instead`,
  )
}
