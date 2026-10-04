/**
 * The reverse interpreter: reverse-mode automatic differentiation by recording (a tape) and a backward sweep.
 *
 * Reverse mode follows Griewank and Walther (2008), "Evaluating Derivatives", ch. 3–4. The forward pass records every
 * primitive application whose inputs include a tracer of this interpreter, in evaluation order (a topological order
 * of the computation graph). The backward sweep visits the records in reverse, pulling the output's cotangent back
 * through each primitive's vjp rule and summing the contributions that reach each value.
 *
 * Nesting needs no special case (design K §4.1). A record keeps its inputs *lowered*: this interpreter's tracers are
 * replaced by the values they stand for, which are raw at the outermost level and tracers of enclosing transforms when
 * nested. The sweep runs the vjp rules on those lowered values, so the backward computation of an inner `grad` is
 * itself traced by the enclosing transform, and a first-order sweep computes on raw values and records nothing.
 *
 * Complex values are pairs of reals (design K §8.1): the cotangent of z = x + iy is x̄ + iȳ, and a real value that
 * receives a complex cotangent keeps its real part.
 */

import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  add,
  apply,
  avalOf,
  nextLevel,
  projectReal,
  Tracer,
  type Aval,
  type Interpreter,
  type Primitive,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** One record: a primitive application, or an input leaf of the transform (`primitive` null). */
export type TapeRecord = {
  /** The primitive applied; null for an input leaf. */
  readonly primitive: Primitive<unknown> | null
  readonly params: unknown
  /** The inputs, lowered (this interpreter's tracers replaced by their values). */
  readonly inputs: readonly Value[]
  /** For each input, the record it came from (this interpreter's tracer), or −1 for a constant. */
  readonly sources: readonly number[]
  /** The output, lowered. */
  readonly output: Value
  /** A label for an input leaf (its path in the argument tree). */
  readonly label?: string
}

/** A value traced by a reverse interpreter: `value` (one level down) and the record that produced it. */
export class ReverseTracer extends Tracer {
  readonly interpreter: ReverseInterpreter
  readonly value: Value
  readonly record: number
  constructor(interpreter: ReverseInterpreter, value: Value, record: number) {
    super()
    this.interpreter = interpreter
    this.value = value
    this.record = record
  }
  get level(): number {
    return this.interpreter.level
  }
  private known: Aval | undefined
  get aval(): Aval {
    return (this.known ??= avalOf(this.value))
  }
  primal(): Value {
    return this.value
  }
}

/** The result of a backward sweep. */
export type Backward = {
  /** The cotangent of each requested input leaf, or null where the outputs do not depend on it. */
  cotangents: (Value | null)[]
  /** With `keepAll`, the cotangent of every record that received one, by record index. */
  all: Map<number, Value>
}

/** The reverse interpreter of one transform: its records and its backward sweep. */
export class ReverseInterpreter implements Interpreter {
  readonly level = nextLevel()
  readonly kind = 'reverse'
  readonly records: TapeRecord[] = []

  /** True when `x` is a tracer of this interpreter. */
  owns(x: unknown): x is ReverseTracer {
    return x instanceof ReverseTracer && x.interpreter === this
  }

  /** `x` one level down: its value if it is this interpreter's tracer, otherwise `x` itself. */
  lower(x: Value): Value {
    return this.owns(x) ? x.value : x
  }

  /** An input leaf of the transform: a record with no primitive, and its tracer. */
  input(value: Value, label?: string): ReverseTracer {
    this.records.push({ primitive: null, params: undefined, inputs: [], sources: [], output: value, label })
    return new ReverseTracer(this, value, this.records.length - 1)
  }

  process(p: Primitive<unknown>, inputs: readonly Value[], params: unknown): Value {
    const lowered = inputs.map((x) => this.lower(x))
    const output = apply(p, lowered, params)
    // A piecewise-constant primitive has a zero derivative: its output is a constant for this transform.
    if (p.zeroDerivative) return output
    const sources = inputs.map((x) => (this.owns(x) ? x.record : -1))
    this.records.push({ primitive: p, params, inputs: lowered, sources, output })
    return new ReverseTracer(this, output, this.records.length - 1)
  }

  /**
   * Pull `seeds` (one cotangent per output, of the output's kind and shape) back to the input leaves `wrt`. Records
   * are visited from the last output backwards; each rule runs on lowered values, so it is traced by any enclosing
   * transform and by nothing here. A primitive without a derivative on a path to an output is an error, never a
   * silent zero.
   */
  backward(
    outputs: readonly Value[],
    seeds: readonly Value[],
    wrt: readonly ReverseTracer[],
    keepAll = false,
  ): Backward {
    const all = new Map<number, Value>()
    const cotangent: (Value | undefined)[] = new Array(this.records.length)
    let end = -1
    outputs.forEach((y, k) => {
      if (!this.owns(y)) return
      const prev = cotangent[y.record]
      cotangent[y.record] = prev === undefined ? seeds[k] : add(prev, seeds[k])
      if (y.record > end) end = y.record
    })
    let start = Infinity
    for (const w of wrt) if (w.record < start) start = w.record
    for (let id = end; id >= start; id--) {
      const g = cotangent[id]
      if (g === undefined) continue
      if (keepAll) all.set(id, g)
      const r = this.records[id]
      if (r.primitive === null) continue
      const needed = r.sources.map((s) => s >= 0)
      const rule = r.primitive.vjp
      if (rule === null) throw new NotDifferentiableError(r.primitive.name)
      const gs = rule(g, r.inputs, r.output, r.params, needed)
      r.sources.forEach((s, i) => {
        const raw = gs[i]
        if (s < 0 || raw === null || raw === undefined) return
        // ℝ² convention (design K §8.1): a real value receives the real part of a complex cotangent.
        const gi = projectReal(raw, avalOf(this.records[s].output).dtype)
        const prev = cotangent[s]
        cotangent[s] = prev === undefined ? gi : add(prev, gi)
      })
    }
    return { cotangents: wrt.map((w) => cotangent[w.record] ?? null), all }
  }

  /**
   * Pull a cotangent `g` of record `id` back through that record's rule alone: one entry per input, null for a
   * constant input or where the rule gives none. With g = 1 on a scalar record this is its local partial derivative
   * with respect to each input. For inspection (`traceGraph`).
   */
  pullback(id: number, g: Value): (Value | null)[] {
    const r = this.records[id]
    const rule = r.primitive?.vjp ?? null
    if (rule === null) return r.sources.map(() => null)
    const gs = rule(
      g,
      r.inputs,
      r.output,
      r.params,
      r.sources.map((s) => s >= 0),
    )
    return r.sources.map((s, i) => (s < 0 ? null : (gs[i] ?? null)))
  }
}
