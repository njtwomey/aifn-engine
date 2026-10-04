/**
 * The batch interpreter: `vmap`, which runs a function written for one example on a whole batch at once.
 *
 * A batch tracer holds a value with one extra axis, the batch axis, at a known position; the traced function sees one
 * example (the value without that axis). Each primitive is applied once to the whole batch by its batching rule, which
 * says where the batch axis of the output ends up. Elementwise primitives batch by moving the batch axis to the front
 * and broadcasting. A primitive without a batching rule falls back to a loop: apply it to each example and stack the
 * results, which is always correct and only slower (design K §4.1).
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  apply,
  avalOf,
  batchedValueError,
  isTraced,
  nextLevel,
  slice,
  stack,
  sum,
  Tracer,
  unwrap,
  type Aval,
  type Interpreter,
  type Primitive,
  type Raw,
  type SliceSpec,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** A value traced by a batch interpreter: `value` (one level down) holds every example along axis `axis`. */
export class BatchTracer extends Tracer {
  readonly interpreter: BatchInterpreter
  readonly value: Value
  readonly axis: number
  /** One example's abstract value: `value` without the batch axis. */
  readonly aval: Aval
  constructor(interpreter: BatchInterpreter, value: Value, axis: number, number = false) {
    super()
    this.interpreter = interpreter
    this.value = value
    this.axis = axis
    const v = avalOf(value)
    this.aval = { shape: v.shape.filter((_, k) => k !== axis), dtype: v.dtype, number }
  }
  get level(): number {
    return this.interpreter.level
  }
  primal(): Value {
    throw batchedValueError()
  }
}

/** Example `b` of a batched value: index the batch axis (to a number when an example is a number). */
export function example(value: Value, axis: number, b: number, number: boolean): Value {
  const specs: SliceSpec[] = Array.from({ length: axis + 1 }, (_, k) => (k === axis ? b : null))
  const one = slice(value, ...specs)
  return number ? sum(one) : one
}

/**
 * The concrete examples behind a value: one (the value) when it is concrete, one per batch element inside `vmap` (each
 * batch tracer's value split along its axis, level by level, so nested `vmap`s give every combination), or null when
 * some level holds no concrete value. A derivative tracer is read through its primal. For checks and stopping tests
 * that need numbers inside `vmap`: the value of each example, where `unwrap` would refuse a batch.
 */
export function batchExamples(x: Value): Raw[] | null {
  if (!isTraced(x)) return [x as Raw]
  if (x instanceof BatchTracer) {
    const inner = batchExamples(x.value)
    if (inner === null) return null
    return inner.flatMap((r) => {
      const size = avalOf(r).shape[x.axis]
      return Array.from({ length: size }, (_, b) => unwrap(example(r, x.axis, b, false)))
    })
  }
  return x instanceof Tracer ? batchExamples(x.primal()) : null
}

/** The batch interpreter of one `vmap`, over a batch of `size` examples. */
export class BatchInterpreter implements Interpreter {
  readonly level = nextLevel()
  readonly kind = 'batch'
  readonly size: number
  constructor(size: number) {
    this.size = size
  }

  owns(x: unknown): x is BatchTracer {
    return x instanceof BatchTracer && x.interpreter === this
  }

  /** A tracer for a batched value whose batch axis is `axis`. */
  wrap(value: Value, axis: number, number = false): BatchTracer {
    const n = avalOf(value).shape[axis]
    if (n !== this.size) throw new ShapeError('vmap', `vmap: batch axis ${axis} has length ${n}, expected ${this.size}`)
    return new BatchTracer(this, value, axis, number)
  }

  process(p: Primitive<unknown>, inputs: readonly Value[], params: unknown): Value {
    const values = inputs.map((x) => (this.owns(x) ? x.value : x))
    const axes = inputs.map((x) => (this.owns(x) ? x.axis : null))
    const examples = inputs.map((x) => avalOf(x))
    if (p.batch !== null) {
      const [out, axis] = p.batch(values, axes, params, this.size)
      const number = p.shape !== null ? p.shape(examples, params).number : false
      return new BatchTracer(this, out, axis, number)
    }
    // The fallback: one application per example, stacked along a new leading axis.
    const outs: Value[] = []
    for (let b = 0; b < this.size; b++) {
      const args = inputs.map((x) => (this.owns(x) ? example(x.value, x.axis, b, x.aval.number) : x))
      outs.push(apply(p, args, params))
    }
    if (outs.length === 0) throw new AifnError('vmap', `vmap: ${p.name} has no batching rule and the batch is empty`)
    return new BatchTracer(this, stack(outs, 0), 0, avalOf(outs[0]).number)
  }
}
