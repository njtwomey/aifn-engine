/**
 * The forward interpreter: forward-mode automatic differentiation with dual numbers.
 *
 * A forward tracer carries a primal value and a tangent (its directional derivative along the direction being pushed
 * forward). Each primitive computes its primal output from the primal inputs and its output tangent from the input
 * tangents by its jvp rule, in the same single pass (Griewank and Walther, 2008, §3.1; Wengert, 1964). A null tangent is
 * a symbolic zero: constants cost nothing.
 *
 * A primitive without a jvp rule falls back, for that application alone, to the **transpose trick**: its vjp
 * u ↦ Jᵀu is linear in u, so J·t is the gradient in u of ⟨Jᵀu, t⟩, obtained by one reverse sweep over the one rule.
 *
 * Complex values are pairs of reals (design K §8.1): a tangent of a complex value is complex, and a tangent of a real
 * output is real.
 */

import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  add,
  apply,
  avalOf,
  conj,
  mul,
  nextLevel,
  projectReal,
  realPart,
  sum,
  Tracer,
  zerosOf,
  type Aval,
  type Interpreter,
  type Primitive,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { ReverseInterpreter } from './reverse'

/** A value traced by a forward interpreter: a primal value (one level down) and its tangent (null for zero). */
export class ForwardTracer extends Tracer {
  readonly interpreter: ForwardInterpreter
  readonly value: Value
  readonly tangent: Value | null
  constructor(interpreter: ForwardInterpreter, value: Value, tangent: Value | null) {
    super()
    this.interpreter = interpreter
    this.value = value
    this.tangent = tangent
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

/**
 * J·t for a primitive with a vjp but no jvp, by the transpose trick: ⟨vjp(u), t⟩ is linear in u, and its gradient in u
 * (at any u, here zero) is J·t. One reverse sweep of this one rule.
 */
function jvpByTranspose(
  p: Primitive<unknown>,
  tangents: readonly (Value | null)[],
  inputs: readonly Value[],
  out: Value,
  params: unknown,
): Value | null {
  const vjp = p.vjp
  if (vjp === null) throw new NotDifferentiableError(p.name)
  const rev = new ReverseInterpreter()
  const u = rev.input(zerosOf(avalOf(out)))
  const cts = vjp(
    u,
    inputs,
    out,
    params,
    tangents.map((t) => t !== null),
  )
  let s: Value | null = null
  cts.forEach((c, i) => {
    const t = tangents[i]
    if (c === null || c === undefined || t === null) return
    // The ℝ² inner product Re Σ conj(c)·t (the plain Σ c·t for real values), so complex primitives work too.
    const term = sum(realPart(mul(conj(c), t)))
    s = s === null ? term : add(s, term)
  })
  if (s === null || !rev.owns(s)) return null
  return rev.backward([s], [1], [u]).cotangents[0]
}

/** The forward interpreter of one transform (one direction pushed forward). */
export class ForwardInterpreter implements Interpreter {
  readonly level = nextLevel()
  readonly kind = 'forward'

  owns(x: unknown): x is ForwardTracer {
    return x instanceof ForwardTracer && x.interpreter === this
  }

  /** A tracer for `value` moving in direction `tangent`. */
  seed(value: Value, tangent: Value | null): ForwardTracer {
    return new ForwardTracer(this, value, tangent)
  }

  /** The primal of `x` (one level down). */
  primalOf(x: Value): Value {
    return this.owns(x) ? x.value : x
  }

  /** The tangent of `x` in this interpreter's direction: null (zero) for anything that is not its tracer. */
  tangentOf(x: Value): Value | null {
    return this.owns(x) ? x.tangent : null
  }

  process(p: Primitive<unknown>, inputs: readonly Value[], params: unknown): Value {
    const primals = inputs.map((x) => this.primalOf(x))
    const out = apply(p, primals, params)
    if (p.zeroDerivative) return out
    const tangents = inputs.map((x) => this.tangentOf(x))
    if (tangents.every((t) => t === null)) return out
    const t = p.jvp !== null ? p.jvp(tangents, primals, out, params) : jvpByTranspose(p, tangents, primals, out, params)
    // A real output's tangent is real: a rule computing Re(…) as a complex value is projected (design K §8.1).
    return t === null ? out : new ForwardTracer(this, out, projectReal(t, avalOf(out).dtype))
  }
}
