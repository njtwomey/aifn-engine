/**
 * The forward interpreter: forward-mode automatic differentiation with dual numbers.
 *
 * A forward tracer carries a primal value and a tangent (its directional derivative along the direction being pushed
 * forward). Each primitive computes its primal output from the primal inputs and its output tangent from the input
 * tangents by its jvp rule, in the same single pass (Griewank and Walther, 2008, §3.1; Wengert, 1964). A null tangent
 * is a symbolic zero: constants cost nothing.
 *
 * A primitive without a jvp rule falls back, for that application alone, to the **transpose trick**: its vjp
 * $\uvec \mapsto \Jmat^\top\uvec$ is linear in $\uvec$, so $\Jmat\tvec$ is the gradient in $\uvec$ of
 * $\langle \Jmat^\top\uvec, \tvec \rangle$, obtained by one reverse sweep over the one rule.
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

/**
 * A value traced by a forward interpreter: a primal value (one level down) and its tangent (null for zero). A dual
 * number $x + \dot{x}\varepsilon$, with the tangent $\dot{x}$ of the primal's shape.
 *
 * @example A dual number pushed through $x^2$
 * const fwd = new ForwardInterpreter()
 * const x = fwd.seed(3, 1)
 * const y = mul(x, x)
 * print('is a ForwardTracer:', y instanceof ForwardTracer)
 * print('value =', y.value, 'tangent =', y.tangent)
 */
export class ForwardTracer extends Tracer {
  /** The forward interpreter that owns this tracer. */
  readonly interpreter: ForwardInterpreter
  /** The primal value, one level down. */
  readonly value: Value
  /** The tangent, of the primal's shape; null for a zero tangent. */
  readonly tangent: Value | null
  /** A tracer of `interpreter` for `value` moving in direction `tangent`. */
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
 * $\Jmat\tvec$ for a primitive with a vjp but no jvp, by the transpose trick:
 * $\langle \mathrm{vjp}(\uvec), \tvec \rangle$ is linear in $\uvec$, and its gradient in $\uvec$ (at any
 * $\uvec$, here zero) is $\Jmat\tvec$. One reverse sweep of this one rule. Throws `NotDifferentiableError` when the
 * primitive has no vjp either.
 *
 * @param p The primitive applied.
 * @param tangents The tangent of each input, null for a constant input.
 * @param inputs The primal inputs, one level down.
 * @param out The primal output already computed from `inputs`.
 * @param params The primitive's parameters for this application.
 * @returns The output's tangent, or null when it does not depend on any tangent.
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

/**
 * The forward interpreter of one transform (one direction pushed forward). Each primitive applied to its tracers
 * computes the primal output and its tangent in one pass. `jvp` creates one per call; using it directly shows the
 * mechanism.
 *
 * @example The derivative of $x \sin x$ at 1 in one forward pass
 * const fwd = new ForwardInterpreter()
 * const x = fwd.seed(1, 1)
 * const y = mul(x, sin(x))
 * print('y =', fwd.primalOf(y))
 * print("y' =", fwd.tangentOf(y))
 * print('sin 1 + cos 1 =', Math.sin(1) + Math.cos(1))
 */
export class ForwardInterpreter implements Interpreter {
  /** This interpreter's level, above every interpreter created before it. */
  readonly level = nextLevel()
  /** The kind of interpreter. */
  readonly kind = 'forward'

  /** True when `x` is a tracer of this interpreter. */
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

  /**
   * Apply primitive `p` to `inputs` (some of them this interpreter's tracers): the primal output, and its tangent by
   * the primitive's jvp rule (or the transpose trick). A constant output is returned untraced.
   */
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
