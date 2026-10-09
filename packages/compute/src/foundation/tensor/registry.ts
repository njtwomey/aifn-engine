/**
 * The registry of primitives: every operation defined with `definePrimitive` or `elementwise` (and so every one made
 * by the `defineOp` wrapper) is recorded here under its id, `module/name` (`foundation/tensor/exp`,
 * `numerics/special/erf`, `numerics/linalg/cholesky`). Registering an id twice throws, so "defined once" is checked
 * when modules load. `registry.list()` feeds the generated primitive tests and the lab's reference pages.
 *
 * A name without a module prefix (`'softplus'` rather than `'nn/functional/softplus'`) defines a local primitive that
 * is not registered: per-call maps (`map`, `map2`), derived derivative primitives and one-off primitives in figures.
 */

import type { Raw } from 'aifn-compute/foundation/contracts'
import type { Tensor } from './core'
import type { ResultRule } from './dtype'
import { AifnError } from 'aifn-compute/foundation/errors'
import type { Aval, Value } from './trace'

/** An untraced value: a number or a tensor (`aifn-compute/foundation/contracts`' `Raw`). */
export type { Raw }

/** A general primitive applied to its inputs and parameters. */
export type Op<P> = (inputs: readonly Value[], params: P) => Value

/**
 * A primitive's reverse rule (vector–Jacobian product): `vjp(cotangent, inputs, output, params, needed)` returns one
 * cotangent per input (`null` for a zero cotangent). `params` are the primitive's non-differentiable arguments (axes,
 * shapes, options). `needed[i]` says whether input $i$ is being differentiated; a rule may skip the others
 * (returning null) and must not raise an error for them. Written with primitives, so that it can be differentiated
 * again.
 */
export type OpVjp<P> = (
  cotangent: Value,
  inputs: readonly Value[],
  output: Value,
  params: P,
  needed: readonly boolean[],
) => (Value | null)[]

/**
 * A primitive's forward rule (Jacobian–vector product): `jvp(tangents, inputs, output, params)` returns the tangent of
 * the output (same kind and shape as the output), or null for a zero tangent. `tangents[i]` is null where input $i$ is
 * a constant. Written with primitives, so that it can be differentiated again.
 */
export type OpJvp<P> = (
  tangents: readonly (Value | null)[],
  inputs: readonly Value[],
  output: Value,
  params: P,
) => Value | null

/**
 * The transpose of a primitive that is linear in input `which` (the others held fixed): the cotangent of that input
 * (same kind and shape as it) given the output's cotangent. `inputs` holds every input's value.
 */
export type OpTranspose<P> = (cotangent: Value, inputs: readonly Value[], which: number, params: P) => Value | null

/**
 * A primitive's batching rule, for `vmap`: `values[i]` carries a batch axis at `axes[i]` (null when input $i$ is not
 * batched; at least one is), and every example is `size` long along it. Returns the batched output and the position
 * of its batch axis. Written with primitives (values may be traced by enclosing transforms).
 */
export type OpBatch<P> = (
  values: readonly Value[],
  axes: readonly (number | null)[],
  params: P,
  size: number,
) => [Value, number]

/**
 * A primitive's shape rule (abstract evaluation): the output's shape, dtype and kind from its inputs', without data.
 */
export type ShapeRule<P> = (avals: readonly Aval[], params: P) => Aval

/**
 * How each rule of a primitive was obtained: written for it (`own`), derived from another of its rules (`derived`:
 * elementwise derivatives, linearity, a transpose), or missing. A missing jvp falls back to the transpose trick on its
 * vjp and a missing batch rule to a loop over the batch (design K §4.2), so `missing` is a gap, not an error.
 */
export type RuleSource = 'own' | 'derived' | 'missing'

/**
 * The result dtype rule of a primitive (design K §3.2; see `ResultRule` in dtype.ts): `same`, `float`, `bool`, `real`,
 * `index` or `complex`, applied to the promoted input dtype.
 */
export type DTypeRule = ResultRule

/** An interval of test inputs for one argument: uniform from `lo` to `hi`, or the integers in it when `integer`. */
export type Domain = { readonly lo: number; readonly hi: number; readonly integer?: boolean }

/** Draws test tensors from a keyed stream: a tensor of `shape` with entries in `domain` (default $[-2, 2]$). */
export type Draw = (shape: readonly number[], domain?: Domain) => Tensor

/** One test input of a general primitive: its inputs and parameters. */
export type PrimitiveCase = { readonly inputs: readonly Raw[]; readonly params?: unknown }

/** What the generated tests need to exercise a primitive (design K §10.1). */
export type PrimitiveTest = {
  /** Elementwise primitives: the domain of each argument (one entry applies to all). Default $[-2, 2]$. */
  readonly domain?: Domain | readonly Domain[]
  /** General primitives: inputs and parameters, drawn with `draw`. Without cases a general primitive is not tested. */
  readonly cases?: (draw: Draw) => readonly PrimitiveCase[]
  /** Relative tolerance of the derivative checks against central differences (default 1e-5). */
  readonly rtol?: number
  /** The derivative rule is itself differentiable, so second derivatives are checked too. */
  readonly secondOrder?: boolean
  /**
   * The primitive accepts complex128 inputs: the generated checks also draw complex inputs (the domain bounds the real
   * parts, imaginary parts in $[-1, 1]$) and differentiate by perturbing real and imaginary parts separately (the
   * $\reals^2$ convention, design K §8.1). Integer-domain arguments (conditions, indices) stay real.
   */
  readonly complex?: boolean
}

/** Documentation of a primitive, for the lab's reference pages and the catalog. */
export type PrimitiveDoc = {
  /** One sentence. */
  readonly summary?: string
  /** The defining formula, in TeX. */
  readonly formula?: string
  /** The slug of the site note that defines it. */
  readonly note?: string
  /** Keys of `content/references.yaml`. */
  readonly references?: readonly string[]
}

/** A primitive: an operation with its forward rule and its derivative, batching and shape rules (design K §5). */
export interface Primitive<P = unknown> {
  /** `module/name`, unique; a bare name for a local primitive that is not registered. */
  readonly id: string
  /** The aifn module that defines it (the part of the id before the slash; empty for a local primitive). */
  readonly module: string
  /** The name used in error messages and graphs (the part of the id after the slash). */
  readonly name: string
  /**
   * `elementwise` for a primitive applied element by element with broadcasting (made by `elementwise`, or declared so),
   * whose batching rule is then derived; `general` otherwise.
   */
  readonly kind: 'elementwise' | 'general'
  /** Number of inputs; `variadic` for a list (concat, einsum). */
  readonly arity: number | 'variadic'
  /** Apply to inputs (numbers, tensors or traced values) and parameters. */
  readonly apply: Op<P>
  /** The forward rule on untraced inputs. */
  readonly impl: (inputs: Raw[], params: P) => Raw
  /** The reverse rule, own or derived; null when the primitive has no derivative. */
  readonly vjp: OpVjp<P> | null
  /** The forward rule, own or derived; null when missing (forward mode then uses the transpose trick on `vjp`). */
  readonly jvp: OpJvp<P> | null
  /** The transpose, for a primitive linear in some inputs; null otherwise. */
  readonly transpose: OpTranspose<P> | null
  /** `linear`: linear in all its inputs jointly; `multilinear`: linear in each input separately; null otherwise. */
  readonly linear: 'linear' | 'multilinear' | null
  /** The batching rule, own or derived; null when missing (`vmap` then loops over the batch and stacks). */
  readonly batch: OpBatch<P> | null
  /** The shape rule; null when missing. */
  readonly shape: ShapeRule<P> | null
  /**
   * Piecewise constant (comparisons, sign, stopGradient): every derivative is zero, so the derivative transforms treat
   * the output as a constant rather than tracing it.
   */
  readonly zeroDerivative: boolean
  /** How each rule was obtained. */
  readonly rules: {
    readonly vjp: RuleSource
    readonly jvp: RuleSource
    readonly batch: RuleSource
    readonly shape: RuleSource
  }
  /** Which inputs have a derivative rule (none when there is no rule at all). */
  readonly differentiable: readonly boolean[] | boolean
  /** The result dtype rule (declared by every tensor primitive; see `DTypeRule`). */
  readonly dtype?: DTypeRule
  /** Its documentation, for the reference pages and the catalog. */
  readonly doc: PrimitiveDoc
  /** What the generated tests need to exercise it. */
  readonly test: PrimitiveTest
}

/** The registered primitives by id. */
const table = new Map<string, Primitive>()

/**
 * Split an id into module and name at its last slash; null for a local (unprefixed) name. An id whose module is not a
 * path of lower-case segments (letters, digits and hyphens, each starting with a letter), or whose name is empty, is an
 * `AifnError`.
 *
 * @param id The primitive's id, `module/name` or a bare local name.
 * @returns The module (everything before the last slash) and the name (everything after it), or null without a slash.
 */
export function parseId(id: string): { module: string; name: string } | null {
  const slash = id.lastIndexOf('/')
  if (slash < 0) return null
  const module = id.slice(0, slash)
  const name = id.slice(slash + 1)
  if (!/^[a-z][a-z0-9-]*(\/[a-z][a-z0-9-]*)*$/.test(module) || name === '') {
    throw new AifnError('definePrimitive', `definePrimitive: invalid primitive id '${id}' (expected module/name)`)
  }
  return { module, name }
}

/**
 * Add a primitive to the registry; an id already registered throws (`AifnError`). For `definePrimitive` and
 * `elementwise`.
 *
 * @param p The primitive, recorded under `p.id`.
 */
export function register<P>(p: Primitive<P>): void {
  if (table.has(p.id)) {
    throw new AifnError(
      'definePrimitive',
      `definePrimitive: primitive '${p.id}' is already registered (each primitive is defined once)`,
    )
  }
  table.set(p.id, p as Primitive)
}

/** The registered primitives. */
export const registry = {
  /** Every registered primitive, in registration order (the order modules loaded). */
  list(): Primitive[] {
    return [...table.values()]
  },
  /** The primitive with this id, or undefined. */
  get(id: string): Primitive | undefined {
    return table.get(id)
  },
  /** True when a primitive with this id is registered. */
  has(id: string): boolean {
    return table.has(id)
  },
}
