/**
 * Custom derivative rules and checkpointing for composite functions (design K §4.3).
 *
 * `customVjp(f, fwd, bwd)` and `customJvp(f, jvp)` give a function written with primitives a derivative of the
 * author's choosing: a stable gradient where the chain rule gives NaN (the norm at 0), a surrogate gradient (the
 * straight-through estimator), or an implicit one (`implicitFixedPoint`, `implicitRoot`). `checkpoint(f)` keeps f's
 * forward pass off the tape and recomputes it during the backward sweep, trading compute for memory.
 *
 * None of these is a primitive: each is a function that looks at the highest-level tracer among its arguments (as
 * `apply` does) and acts for that interpreter.
 *
 * - **Reverse.** The arguments are lowered one level and the function runs on them; the result gets one record per
 *   output leaf whose rule is the custom one (bwd; the transpose of jvp; or a recomputation).
 * - **Forward.** The tangent is the custom jvp; for `customVjp` it is obtained from bwd by the transpose trick (bwd is
 *   linear in the cotangent, so J·t is the gradient in u of ⟨bwd(u), t⟩).
 * - **Batch.** One call per example, stacked (always correct; the custom rule then applies to each example).
 *
 * The lowered call is itself the custom function, so levels below also use the custom rule. Residuals are computed
 * by `fwd` on the lowered arguments, so an enclosing transform differentiates `bwd` through them (second derivatives
 * are the derivatives of the custom rule, as in JAX). A custom rule must be written with primitives to be
 * differentiated again, and `bwd` must be linear in its cotangent.
 */

import { AifnError, NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  add,
  avalOf,
  conj,
  isTraced,
  mul,
  realPart,
  stack,
  sum,
  zerosOf,
  type Primitive,
  type Tracer,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { treeFlatten, treeUnflatten, type Flat } from 'aifn-compute/foundation/pytree'
import { BatchInterpreter, BatchTracer, example } from './batch'
import { ForwardInterpreter } from './forward'
import { ReverseInterpreter, ReverseTracer } from './reverse'
import type { TreeOf } from './transforms'

/** One cotangent per argument, of that argument's structure, or null for a zero cotangent. */
export type Cotangents<A extends unknown[]> = { [K in keyof A]: TreeOf<A[K], Value> | null }

// ── Arguments as leaves ──────────────────────────────────────────────────────────────────────────────────────────────

/** The arguments of one call, flattened: one tree per argument, and every leaf in order. */
type Args = { flats: Flat[]; leaves: Value[] }

function flattenArgs(args: readonly unknown[]): Args {
  const flats = args.map((a) => treeFlatten(a))
  return { flats, leaves: flats.flatMap((f) => f.leaves) }
}

/** Rebuild one argument per tree from a flat list of leaves (the inverse of `flattenArgs`). */
function rebuild(flats: readonly Flat[], leaves: readonly unknown[]): unknown[] {
  let at = 0
  return flats.map((flat) => {
    const part = leaves.slice(at, at + flat.leaves.length)
    at += flat.leaves.length
    return treeUnflatten(flat.treedef, part)
  })
}

/** The highest-level tracer among `leaves`, or null when none is traced. */
function topTracer(leaves: readonly unknown[]): Tracer | null {
  let top: Tracer | null = null
  for (const x of leaves) if (isTraced(x) && (top === null || x.level > top.level)) top = x as Tracer
  return top
}

const anyTraced = (leaves: readonly unknown[]): boolean => leaves.some(isTraced)

/**
 * Cotangents returned by a custom rule (one entry per argument, a tree or null), as one entry per input leaf aligned
 * with `flats`.
 */
function cotangentLeaves(where: string, cts: readonly unknown[], flats: readonly Flat[]): (Value | null)[] {
  if (!Array.isArray(cts) || cts.length !== flats.length)
    throw new AifnError(where, `${where}: the backward rule must return one cotangent per argument (${flats.length})`)
  return flats.flatMap((flat, k) => {
    const c = cts[k] as unknown
    if (c === null || c === undefined) return flat.leaves.map(() => null)
    const leaves = treeFlatten(c).leaves
    if (leaves.length !== flat.leaves.length)
      throw new AifnError(where, `${where}: the cotangent of argument ${k} must have the structure of the argument`)
    return leaves
  })
}

/**
 * A primitive object for records made here: it is never applied (its output is computed outside `apply`), only
 * pulled back through `vjp` during a backward sweep and shown by `traceGraph` under `name`.
 */
function recordPrimitive(name: string, vjp: NonNullable<Primitive<number>['vjp']>): Primitive<unknown> {
  const never = (): never => {
    throw new AifnError(name, `${name}: internal record, not applicable`)
  }
  const p: Primitive<number> = {
    id: name,
    module: '',
    name,
    kind: 'general',
    arity: 'variadic',
    apply: never,
    impl: never,
    vjp,
    jvp: null,
    transpose: null,
    linear: null,
    batch: null,
    shape: null,
    zeroDerivative: false,
    rules: { vjp: 'own', jvp: 'missing', batch: 'missing', shape: 'missing' },
    differentiable: true,
    doc: { summary: 'A record of a custom derivative rule.' },
    test: {},
  }
  return p as Primitive<unknown>
}

/**
 * Record `outs` (lowered values) on `rev` as outputs of one custom application of `inputs` (the lowered argument
 * leaves, `sources` their records), each with the custom rule `vjp` (params = the output leaf's position).
 */
function emit(
  rev: ReverseInterpreter,
  name: string,
  vjp: NonNullable<Primitive<number>['vjp']>,
  args: Args,
  lowered: readonly Value[],
  outs: readonly (Value | null)[],
): (Value | null)[] {
  const primitive = recordPrimitive(name, vjp)
  const sources = args.leaves.map((x) => (rev.owns(x) ? x.record : -1))
  return outs.map((o, k) => {
    if (o === null) return null
    rev.records.push({ primitive, params: k, inputs: lowered, sources, output: o })
    return new ReverseTracer(rev, o, rev.records.length - 1)
  })
}

/** The leaves of `args` one level below `interp`: its own tracers replaced by the values they stand for. */
function lowerFor(interp: ReverseInterpreter | ForwardInterpreter, leaves: readonly Value[]): Value[] {
  return interp instanceof ReverseInterpreter
    ? leaves.map((x) => interp.lower(x))
    : leaves.map((x) => interp.primalOf(x))
}

/**
 * The batch rule shared by every custom function: call `g` once per example and stack each output leaf along a new
 * leading axis. Examples are lowered one level, so the custom rule acts at the levels below.
 */
function perExample(batch: BatchInterpreter, g: (...a: unknown[]) => unknown, args: Args): unknown {
  let treedef: Flat['treedef'] | null = null
  const columns: Value[][] = []
  for (let b = 0; b < batch.size; b++) {
    const leaves = args.leaves.map((x) => (batch.owns(x) ? example(x.value, x.axis, b, x.aval.number) : x))
    const out = treeFlatten(g(...rebuild(args.flats, leaves)))
    treedef ??= out.treedef
    out.leaves.forEach((leaf, j) => (columns[j] ??= []).push(leaf))
  }
  if (treedef === null) throw new AifnError('vmap', 'vmap: a custom function on an empty batch')
  const leaves = columns.map((col) => new BatchTracer(batch, stack(col, 0), 0, avalOf(col[0]).number))
  return treeUnflatten(treedef, leaves)
}

/** The interpreter of `top`, as one of the three this module knows. */
function interpreterOf(where: string, top: Tracer): ReverseInterpreter | ForwardInterpreter | BatchInterpreter {
  const i = top.interpreter
  if (i instanceof ReverseInterpreter || i instanceof ForwardInterpreter || i instanceof BatchInterpreter) return i
  throw new AifnError(where, `${where}: unknown interpreter '${i.kind}'`)
}

// ── customVjp ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The forward half of a custom vjp: the output and what the backward rule needs. */
export type VjpForward<R, Res> = { out: R; residuals: Res }

/** Options of the internal custom vjp: residuals from the arguments and the (already computed) output. */
/**
 * The full specification of a custom-vjp function (`defineCustomVjp`): `customVjp`'s f, fwd and bwd, plus a name for
 * errors, an optional `residualsFrom` and an optional forward rule.
 */
export type CustomVjpSpec<A extends unknown[], R, Res> = {
  /** Names the function in errors. */
  name: string
  /** The function on values (never traced). */
  f: (...a: A) => R
  /** The output and the residuals the backward rule needs. */
  fwd: (...a: A) => VjpForward<R, Res>
  /** The cotangents of the arguments from the residuals and the output's cotangent, written with primitives. */
  bwd: (residuals: Res, cotangent: TreeOf<R, Value>) => Cotangents<A>
  /** Residuals from lowered arguments and the output one level down, instead of calling `fwd` a second time. */
  residualsFrom?: (args: A, out: R) => Res
  /**
   * A forward rule from the residuals and the arguments' tangents (zeros where constant), used by forward mode instead
   * of the transpose trick through bwd. Needed when bwd is linear only in exact arithmetic (an iterative solve whose
   * iteration count depends on the values; the trick runs bwd at a zero cotangent).
   */
  jvp?: (residuals: Res, tangents: A) => TreeOf<R, Value>
}

/**
 * A function with a custom reverse rule and, optionally, a custom forward rule: the machinery behind `customVjp` and
 * the implicit derivatives of `aifn-compute/numerics/implicit`. Under `vmap` f runs per example.
 */
export function defineCustomVjp<A extends unknown[], R, Res>(spec: CustomVjpSpec<A, R, Res>): (...a: A) => R {
  const { name, f, fwd, bwd, residualsFrom, jvp } = spec

  // The output and residuals on arguments one level down: the custom function itself for the output when levels below
  // are traced (so they use the custom rule too), and fwd (or `residualsFrom`) for the residuals.
  const below = (a: A, leaves: readonly Value[]): VjpForward<R, Res> => {
    if (!anyTraced(leaves)) return fwd(...a)
    const out = g(...a)
    return { out, residuals: residualsFrom ? residualsFrom(a, out) : fwd(...a).residuals }
  }

  const g = (...a: A): R => {
    const args = flattenArgs(a)
    const top = topTracer(args.leaves)
    if (top === null) return f(...a)
    const interp = interpreterOf(name, top)
    if (interp instanceof BatchInterpreter) return perExample(interp, g as never, args) as R

    const lowered = lowerFor(interp, args.leaves)
    const { out, residuals } = below(rebuild(args.flats, lowered) as A, lowered)
    const outFlat = treeFlatten(out)
    // bwd for a cotangent of the whole output: zeros except at the leaves given.
    const pull = (cts: readonly (Value | null)[]) =>
      cotangentLeaves(
        name,
        bwd(
          residuals,
          treeUnflatten(
            outFlat.treedef,
            cts.map((c, j) => c ?? zerosOf(avalOf(outFlat.leaves[j]))),
          ),
        ) as unknown[],
        args.flats,
      )

    if (interp instanceof ReverseInterpreter) {
      const rule = (ct: Value, _inputs: readonly Value[], _out: Value, k: number) =>
        pull(outFlat.leaves.map((_, j) => (j === k ? ct : null)))
      return treeUnflatten(outFlat.treedef, emit(interp, name, rule, args, lowered, outFlat.leaves))
    }

    // Forward: J·t by the transpose trick through bwd, one reverse sweep over the (linear) backward rule.
    const tangents = args.leaves.map((x) => interp.tangentOf(x))
    if (tangents.every((t) => t === null)) return out
    if (jvp !== undefined) {
      const full = rebuild(
        args.flats,
        tangents.map((t, i) => t ?? zerosOf(avalOf(lowered[i]))),
      ) as A
      const touts = treeFlatten(jvp(residuals, full)).leaves
      return treeUnflatten(
        outFlat.treedef,
        outFlat.leaves.map((o, j) => interp.seed(o, touts[j] ?? null)),
      )
    }
    const rev = new ReverseInterpreter()
    const us = outFlat.leaves.map((o) => rev.input(zerosOf(avalOf(o))))
    let s: Value | null = null
    pull(us).forEach((c, i) => {
      const t = tangents[i]
      if (c === null || t === null) return
      // The ℝ² inner product Re Σ conj(c)·t (the plain Σ c·t for real values), as forward.ts's transpose trick.
      const term = sum(realPart(mul(conj(c), t)))
      s = s === null ? term : add(s, term)
    })
    const touts: (Value | null)[] =
      s !== null && rev.owns(s) ? rev.backward([s], [1], us).cotangents : us.map(() => null)
    return treeUnflatten(
      outFlat.treedef,
      outFlat.leaves.map((o, j) => interp.seed(o, touts[j])),
    )
  }
  return g
}

/**
 * f with a custom reverse rule. `fwd(...args)` returns f's output and the residuals the rule needs; `bwd(residuals,
 * cotangent)` returns one cotangent per argument (null for zero). bwd must be linear in the cotangent and written with
 * primitives if it is to be differentiated again (higher derivatives differentiate bwd). Works under every transform:
 * forward mode obtains J·t from bwd by the transpose trick, `vmap` applies f per example.
 *
 * @example
 * // The straight-through estimator: round in the forward pass, identity in the backward pass.
 * const ste = customVjp((x: Value) => round(x), (x) => ({ out: round(x), residuals: null }), (_, g) => [g])
 */
export function customVjp<A extends unknown[], R, Res>(
  f: (...a: A) => R,
  fwd: (...a: A) => VjpForward<R, Res>,
  bwd: (residuals: Res, cotangent: TreeOf<R, Value>) => Cotangents<A>,
): (...a: A) => R {
  return defineCustomVjp({ name: 'customVjp', f, fwd, bwd })
}

// ── customJvp ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * f with a custom forward rule: `jvp(primals, tangents)` returns [f(primals), J·tangents], the tangent linear in
 * `tangents` and written with primitives. Reverse mode transposes it (one reverse sweep over jvp in the tangents), so
 * one rule serves both modes; `vmap` applies f per example.
 *
 * @example
 * // ‖x‖ with the subgradient 0 at x = 0 instead of NaN (primitives only, so it also runs under vmap).
 * const safeNorm = customJvp((x: Value) => norm(x), ([x], [t]) => {
 *   const n = norm(x)
 *   const zero = equalTo(n, 0)
 *   return [n, where(zero, 0, div(dot(x, t), where(zero, 1, n)))]
 * })
 */
export function customJvp<A extends unknown[], R>(
  f: (...a: A) => R,
  jvp: (primals: A, tangents: A) => [R, TreeOf<R, Value>],
): (...a: A) => R {
  const name = 'customJvp'
  const g = (...a: A): R => {
    const args = flattenArgs(a)
    const top = topTracer(args.leaves)
    if (top === null) return f(...a)
    const interp = interpreterOf(name, top)
    if (interp instanceof BatchInterpreter) return perExample(interp, g as never, args) as R

    const lowered = lowerFor(interp, args.leaves)
    const primals = rebuild(args.flats, lowered) as A
    const zeroTangents = (live: (Value | null)[]) =>
      rebuild(
        args.flats,
        live.map((t, i) => t ?? zerosOf(avalOf(lowered[i]))),
      ) as A

    if (interp instanceof ForwardInterpreter) {
      const tangents = args.leaves.map((x) => interp.tangentOf(x))
      if (tangents.every((t) => t === null)) return anyTraced(lowered) ? g(...primals) : f(...primals)
      const [value, tangent] = jvp(primals, zeroTangents(tangents))
      const out = anyTraced(lowered) ? g(...primals) : value
      const outFlat = treeFlatten(out)
      const tFlat = treeFlatten(tangent).leaves
      return treeUnflatten(
        outFlat.treedef,
        outFlat.leaves.map((o, j) => interp.seed(o, tFlat[j] ?? null)),
      )
    }

    // Reverse: the vjp is the transpose of t ↦ jvp(x, t), by one reverse sweep in the tangents at backward time.
    const out = anyTraced(lowered) ? g(...primals) : f(...primals)
    const outFlat = treeFlatten(out)
    const rule = (ct: Value, inputs: readonly Value[], _out: Value, k: number, needed: readonly boolean[]) => {
      const rev = new ReverseInterpreter()
      const ts = inputs.map((x, i) => (needed[i] ? rev.input(zerosOf(avalOf(x))) : null))
      const [, tangent] = jvp(rebuild(args.flats, inputs) as A, zeroTangents(ts))
      const tk = treeFlatten(tangent).leaves[k]
      const live = ts.filter((t): t is ReverseTracer => t !== null)
      if (tk === undefined || !rev.owns(tk)) return inputs.map(() => null)
      const cts = rev.backward([tk], [ct], live).cotangents
      let at = 0
      return ts.map((t) => (t === null ? null : cts[at++]))
    }
    return treeUnflatten(outFlat.treedef, emit(interp, name, rule, args, lowered, outFlat.leaves))
  }
  return g
}

// ── checkpoint ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * f with its forward pass kept off the tape of the reverse transform differentiating it: the backward sweep
 * recomputes f from its arguments under a fresh reverse level and pulls the cotangent back through that. Memory for f's
 * intermediates is then needed only while its own backward runs (Griewank and Walther, 2008, ch. 12). Checkpointing
 * every √n steps of an n-step loop stores O(√n) states and recomputes each segment once.
 *
 * f must receive every traced value it depends on as an argument: a closure over a tracer of the transform being
 * checkpointed raises an error. Under forward mode and `vmap`, f runs as it is (there is no tape to save).
 */
export function checkpoint<A extends unknown[], R>(f: (...a: A) => R): (...a: A) => R {
  const name = 'checkpoint'
  return (...a: A): R => {
    const args = flattenArgs(a)
    const top = topTracer(args.leaves)
    if (top === null) return f(...a)
    const rev = interpreterOf(name, top)
    if (!(rev instanceof ReverseInterpreter)) return f(...a)

    // Forward pass: run f once under a probe level that sees only the leaves this transform traces. The probe's
    // records say which outputs depend on them and are dropped when this call returns; nothing is kept on `rev`.
    const lowered = lowerFor(rev, args.leaves)
    const probe = new ReverseInterpreter()
    const probed = args.leaves.map((x, i) => (rev.owns(x) ? probe.input(lowered[i]) : lowered[i]))
    const outFlat = treeFlatten(f(...(rebuild(args.flats, probed) as A)))
    const leak = (v: Value) => isTraced(v) && (v as Tracer).level >= rev.level
    const outs = outFlat.leaves.map((o) => {
      const v = probe.owns(o) ? o.value : o
      if (leak(v))
        throw new AifnError(
          name,
          'checkpoint: f closes over a value traced by the transform being checkpointed; pass it as an argument',
        )
      return v
    })
    const dependent = outFlat.leaves.map((o) => probe.owns(o))

    // Backward: recompute f once under a fresh level, shared by the output leaves pulled back from it.
    let cache: { rev: ReverseInterpreter; ins: ReverseTracer[]; outs: Value[]; used: number } | null = null
    const rule = (ct: Value, inputs: readonly Value[], _out: Value, k: number, needed: readonly boolean[]) => {
      if (cache === null) {
        const again = new ReverseInterpreter()
        const ins: ReverseTracer[] = []
        const traced = inputs.map((x, i) => {
          if (!needed[i]) return x
          const t = again.input(x)
          ins.push(t)
          return t
        })
        const outs = treeFlatten(f(...(rebuild(args.flats, traced) as A))).leaves
        cache = { rev: again, ins, outs, used: 0 }
      }
      const c = cache
      if (++c.used >= dependent.filter(Boolean).length) cache = null
      const y = c.outs[k]
      if (!c.rev.owns(y)) return inputs.map(() => null)
      const cts = c.rev.backward([y], [ct], c.ins).cotangents
      let at = 0
      return needed.map((n) => (n ? cts[at++] : null))
    }
    const traced = emit(
      rev,
      name,
      rule,
      args,
      lowered,
      outs.map((v, j) => (dependent[j] ? v : null)),
    )
    return treeUnflatten(
      outFlat.treedef,
      outs.map((v, j) => traced[j] ?? v),
    )
  }
}

// ── Refusing traced inputs ───────────────────────────────────────────────────────────────────────────────────────────

/**
 * Throw `NotDifferentiableError` if any leaf of `tree` is traced. For code that computes on raw arrays (`dense`
 * loops, typed arrays) and so cannot carry a derivative: it refuses traced inputs rather than dropping the trace.
 */
export function refuseTraced(where: string, tree: unknown): void {
  if (treeFlatten(tree).leaves.some(isTraced))
    throw new NotDifferentiableError(
      where,
      `${where}: computes on raw arrays and cannot be differentiated; use unrolled or atConvergence with an algorithm written in primitives`,
    )
}
