/**
 * Primitives: every operation is defined once, with its forward rule and the rules the function transforms need
 * (design K §4.2, §5). A primitive accepts numbers, tensors and traced values and returns the same kind: `exp(2)` is a
 * number, `exp(t)` a tensor, and `exp(tracer)` a tracer of the transform in progress.
 *
 * What the author of each kind of primitive writes, and what is derived here:
 *
 * - Elementwise: a scalar `f` and one derivative $\partial_i$ per argument. Derived: the vjp $g\,\partial_i$, the jvp
 *   $\sum_i t_i\,\partial_i$, batching by broadcasting, and the shape rule.
 * - Linear: `transpose` (the adjoint map) and `linear`. Derived: the vjp is the transpose, and the jvp is the primitive
 *   applied to the tangents.
 * - Other: `vjp` and `jvp` (and optionally `batch`, `shape`). Nothing is derived.
 * - No derivative: `vjp: null`, or `zeroDerivative` if piecewise constant. Nothing is derived.
 *
 * A missing jvp is not an error: forward mode falls back to the transpose trick on the vjp for that primitive alone.
 * A missing batch rule falls back to a loop over the batch (both in `aifn-compute/foundation/autodiff`).
 */

import type { Binary, Unary } from 'aifn-compute/foundation/contracts'
import { promote, scalarDType, type DType, type Tensor } from './core'
import { zeros } from './create'
import { AifnError, DTypeError, NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { resultType } from './dtype'
import { binaryKernel, complexKernel, ternaryKernel, unaryKernel, type Arithmetic, type ComplexRule } from './kernels'
import {
  parseId,
  register,
  type DTypeRule,
  type Op,
  type OpBatch,
  type OpJvp,
  type OpTranspose,
  type OpVjp,
  type Primitive,
  type PrimitiveDoc,
  type PrimitiveTest,
  type Raw,
  type RuleSource,
  type ShapeRule,
} from './registry'
import { apply, avalOf, type Aval, type Traced, type Value } from './trace'
import { add, mul, neg } from './elementwise'
import { conj, realPart } from './complex'
import { broadcastTo, permute, reshape, sumTo } from './structure'
import { sum } from './reduce'
import { broadcastShapes } from './views'

export type { Binary, Op, OpBatch, OpJvp, OpTranspose, OpVjp, Raw, ShapeRule, Unary }

/** The specification of a primitive for `definePrimitive`. */
export type PrimitiveSpec<P> = {
  /**
   * `module/name`, e.g. `numerics/linalg/cholesky`: registered, and unique (a second registration throws). A bare name
   * defines a local primitive that is not registered (see `registry`).
   */
  readonly id: string
  /** Number of inputs, or `variadic` for a list. */
  readonly arity?: number | 'variadic'
  /** The forward rule on untraced inputs (numbers and tensors). */
  readonly impl: (inputs: Raw[], params: P) => Raw
  /**
   * The reverse rule (vector–Jacobian product), written with primitives so that it can be differentiated again; `null`
   * for an operation without a derivative (differentiating through it is then an error). Omit it for a linear
   * primitive (it is derived from `transpose`) or a piecewise-constant one (`zeroDerivative`).
   */
  readonly vjp?: OpVjp<P> | null
  /** The forward rule (Jacobian–vector product). Omit it for a linear or multilinear primitive (derived). */
  readonly jvp?: OpJvp<P>
  /** For a linear or multilinear primitive: the adjoint map in each input. Gives the vjp. */
  readonly transpose?: OpTranspose<P>
  /**
   * `linear`: linear in all inputs jointly (add, reshape, sum, slice, concat), so its jvp is the primitive applied to
   * the tangents. `multilinear`: linear in each input separately (matmul, einsum), so its jvp is
   * $\sum_i p(\dots, t_i, \dots)$, the primitive with input $i$ replaced by its tangent $t_i$, summed over $i$.
   */
  readonly linear?: 'linear' | 'multilinear'
  /** The batching rule for `vmap` (see `OpBatch`). Elementwise primitives get one by broadcasting. */
  readonly batch?: OpBatch<P>
  /** The shape rule (abstract evaluation). */
  readonly shape?: ShapeRule<P>
  /** Piecewise constant: every derivative is zero, and transforms treat the output as a constant. */
  readonly zeroDerivative?: boolean
  /** Which inputs the rule differentiates (default: all when a derivative exists). */
  readonly differentiable?: readonly boolean[] | boolean
  /**
   * `elementwise` for a primitive applied entry by entry with broadcasting, which gets a batching rule by broadcasting;
   * `general` (the default) otherwise.
   */
  readonly kind?: 'elementwise' | 'general'
  /** The result dtype rule (`same`, `float`, `bool`, `real`, `index`, `complex`; design K §3.2). */
  readonly dtype?: DTypeRule
  /** The registry's documentation of the primitive: summary, formula, note (see `PrimitiveDoc`). */
  readonly doc?: PrimitiveDoc
  /** What the generated tests of the primitive draw: domains, cases, complex and second-order checks. */
  readonly test?: PrimitiveTest
}

/** Rules derived by a definer (e.g. `elementwise`) rather than written in the spec. */
type Derived<P> = {
  /** The derived reverse rule, used when the spec gives none and the primitive is not linear. */
  vjp?: OpVjp<P>
  /** The derived forward rule, used when the spec gives none and the primitive is not linear. */
  jvp?: OpJvp<P>
  /** The derived shape rule, used when the spec gives none. */
  shape?: ShapeRule<P>
}

/**
 * A zero of the kind and shape of an abstract value: 0 for a number, else a tensor of zeros (complex128 for a complex
 * value, float64 for any real dtype).
 *
 * @param aval The abstract value: its shape, dtype and whether it is a number (as `avalOf` gives it).
 * @returns 0, or a tensor of zeros of the shape of `aval`.
 *
 * @example Zeros like a number and like a matrix
 * print('like 3:', zerosOf(avalOf(3)))
 * print('like a 2 × 2 matrix:', zerosOf(avalOf(eye(2))))
 */
export function zerosOf(aval: Aval): Raw {
  return aval.number ? 0 : zeros(aval.shape, aval.dtype === 'complex128' ? 'complex128' : 'float64')
}

/**
 * A complex cotangent or tangent `v` projected onto a real value of dtype `dtype`: its real part (design K §8.1). A
 * real input embedded in $\complex$ as $x + 0i$ has, in the $\reals^2$ convention, the cotangent
 * $\operatorname{Re}(\bar{z})$ of the complex cotangent $\bar{z}$.
 *
 * @param v The cotangent or tangent, complex or real.
 * @param dtype The dtype of the value `v` belongs to.
 * @returns The real part of `v` when `v` is complex and `dtype` is not; otherwise `v` itself.
 *
 * @example A complex cotangent of a real input keeps its real part
 * const ct = complex(tensor([1, 2]), tensor([3, 4]))
 * print('for a float64 input:', projectReal(ct, 'float64'))
 * print('for a complex128 input, as (re, im) pairs:', projectReal(ct, 'complex128'))
 */
export function projectReal(v: Value, dtype: DType): Value {
  return dtype !== 'complex128' && avalOf(v).dtype === 'complex128' ? realPart(v) : v
}

/**
 * `v` (a cotangent or tangent) made of the kind, shape and field of `aval`: broadcast up to a tensor, summed to a
 * number when `aval` is a number, a complex value to its real part when `aval` is real, and a real value to a complex
 * one when it is complex.
 *
 * @param v The cotangent or tangent to fit; its shape must broadcast to that of `aval` unless `aval` is a number.
 * @param aval The abstract value it must match: shape, dtype, and whether it is a number.
 * @returns `v` of the kind, shape and field of `aval`.
 *
 * @example Broadcast up, summed down, embedded in the complex numbers
 * print('1 fitted to 2 × 3:', fitTo(1, avalOf(zeros([2, 3]))))
 * print('[1, 2, 3] fitted to a number:', fitTo(tensor([1, 2, 3]), avalOf(0)))
 * print('[1, 2] fitted to complex, as (re, im) pairs:', fitTo(tensor([1, 2]), avalOf(zeros([2], 'complex128'))))
 */
export function fitTo(v: Value, aval: Aval): Value {
  const from = avalOf(v)
  const same = from.shape.length === aval.shape.length && from.shape.every((d, k) => d === aval.shape[k])
  const w = projectReal(v, aval.dtype)
  // A real tangent of a complex output (complex(re, im) in re) is embedded in ℂ, broadcast on the way.
  if (aval.dtype === 'complex128' && from.dtype !== 'complex128') return add(w, zeros(aval.shape, 'complex128'))
  if (same && from.number === aval.number) return w
  if (aval.number) return sum(w)
  return broadcastTo(w, aval.shape)
}

/**
 * Build the primitive object of a spec (and register it when its id has a module). The spec's own rules come first,
 * then those derived from `linear` and `transpose`, then `derived`. Throws `AifnError` for a linear primitive without a
 * transpose rule.
 *
 * @param spec The primitive's specification.
 * @param derived Rules a definer derived (the vjp, jvp and shape rule of `elementwise`), used where the spec and its
 *   linearity give none (default none).
 * @returns The primitive object, registered when its id has a module.
 */
function build<P>(spec: PrimitiveSpec<P>, derived: Derived<P> = {}): Primitive<P> {
  const parsed = parseId(spec.id)
  const kind = spec.kind ?? 'general'
  const linear = spec.linear ?? null
  const transpose = spec.transpose ?? null
  const zeroDerivative = spec.zeroDerivative ?? false
  if (linear !== null && transpose === null) {
    throw new AifnError('definePrimitive', `definePrimitive: ${spec.id} is ${linear} but has no transpose rule`)
  }
  const source = (own: unknown, other: unknown): RuleSource =>
    own !== undefined && own !== null ? 'own' : other !== undefined && other !== null ? 'derived' : 'missing'

  // A transpose of a complex-valued map may give a complex cotangent to a real input: its real part is the cotangent.
  const vjpFromTranspose: OpVjp<P> | null =
    transpose &&
    ((ct, inputs, _out, params, needed) =>
      inputs.map((x, i) => {
        if (!needed[i]) return null
        const c = transpose(ct, inputs, i, params)
        return c === null ? null : projectReal(c, avalOf(x).dtype)
      }))
  const vjp: OpVjp<P> | null = zeroDerivative
    ? (_ct, inputs) => inputs.map(() => null)
    : (spec.vjp ?? vjpFromTranspose ?? derived.vjp ?? null)

  // The jvp of a linear primitive is the primitive applied to the tangents (constants contribute zero tangents); of a
  // multilinear one, the sum over inputs of the primitive with that input replaced by its tangent.
  const linearJvp: OpJvp<P> | null =
    linear === 'linear'
      ? (tangents, inputs, _out, params) =>
          tangents.every((t) => t === null)
            ? null
            : apply(
                prim,
                tangents.map((t, i) => t ?? zerosOf(avalOf(inputs[i]))),
                params,
              )
      : linear === 'multilinear'
        ? (tangents, inputs, _out, params) => {
            let acc: Value | null = null
            tangents.forEach((t, i) => {
              if (t === null) return
              const term = apply(
                prim,
                inputs.map((x, j) => (j === i ? t : x)),
                params,
              )
              acc = acc === null ? term : add(acc, term)
            })
            return acc
          }
        : null
  const jvp: OpJvp<P> | null = zeroDerivative ? () => null : (spec.jvp ?? linearJvp ?? derived.jvp ?? null)
  const batch: OpBatch<P> | null =
    spec.batch ?? (kind === 'elementwise' ? (values, axes, params) => broadcastBatch(prim, values, axes, params) : null)
  const shape = spec.shape ?? derived.shape ?? null

  const prim: Primitive<P> = {
    id: spec.id,
    module: parsed?.module ?? '',
    name: parsed?.name ?? spec.id,
    kind,
    arity: spec.arity ?? 'variadic',
    apply: (inputs, params) => apply(prim, inputs, params),
    impl: spec.impl,
    vjp,
    jvp,
    transpose,
    linear,
    batch,
    shape,
    zeroDerivative,
    rules: {
      vjp: zeroDerivative ? 'derived' : source(spec.vjp, vjpFromTranspose ?? derived.vjp),
      jvp: zeroDerivative ? 'derived' : source(spec.jvp, linearJvp ?? derived.jvp),
      batch: source(spec.batch, kind === 'elementwise' ? true : null),
      shape: source(spec.shape, derived.shape),
    },
    differentiable: spec.differentiable ?? (vjp !== null && !zeroDerivative),
    dtype: spec.dtype,
    doc: spec.doc ?? {},
    test: spec.test ?? {},
  }
  if (parsed) register<P>(prim)
  return prim
}

/**
 * Define a primitive: its forward rule `impl` and its rules for the transforms (see `PrimitiveSpec`), registered under
 * `id` (design K §5). The result applies the primitive to a list of inputs (numbers, tensors or traced values) and its
 * parameters: untraced inputs go straight to `impl`, traced ones to the interpreter of the innermost transform. A
 * second registration of the same `id` throws, and so does a linear primitive without a transpose.
 *
 * @param spec The primitive: its `id`, forward rule `impl`, and the rules the transforms need.
 * @returns The primitive, applied as `op(inputs, params)`.
 *
 * @example A cube primitive with its reverse rule
 * // A bare id defines a local primitive, which is not registered.
 * const cube = definePrimitive({
 *   id: 'cube',
 *   arity: 1,
 *   impl: ([x]) => map(x, (v) => v ** 3),
 *   vjp: (g, [x]) => [mul(g, mul(3, square(x)))],
 * })
 * print('cube([1, 2, 3]) =', cube([tensor([1, 2, 3])], undefined))
 * print('d/dx x^3 at 2 =', grad((x) => cube([x], undefined))(2))
 */
export function definePrimitive<P = undefined>(spec: PrimitiveSpec<P>): Op<P> {
  return build(spec).apply
}

/** Registry metadata a `define*` wrapper may carry: documentation and test domains. */
export type PrimitiveMeta = { readonly doc?: PrimitiveDoc; readonly test?: PrimitiveTest }

/**
 * Define a general primitive from its forward rule and vjp (a thin wrapper over `definePrimitive`); `rules` may add a
 * jvp, a batching rule, a shape rule and metadata. Pass `vjp: null` for an operation without a derivative.
 *
 * @param name The primitive's id: `module/name` to register it, or a bare name for a local primitive.
 * @param forward The forward rule on untraced inputs (numbers and tensors), given the parameters.
 * @param vjp The reverse rule, written with primitives so that it can be differentiated again; `null` when the
 *   operation has no derivative.
 * @param rules The rest of the spec: `arity`, a jvp, a batching rule, a shape rule, linearity, dtype and metadata.
 * @returns The primitive, applied as `op(inputs, params)`.
 *
 * @example A cube from its forward rule and vjp
 * const cube = defineOp('cube', ([x]) => map(x, (v) => v ** 3), (g, [x]) => [mul(g, mul(3, square(x)))])
 * print('cube([1, 2, 3]) =', cube([tensor([1, 2, 3])], undefined))
 * print('gradient of the sum =', grad((x) => sum(cube([x], undefined)))(tensor([1, 2, 3])))
 */
export function defineOp<P = undefined>(
  name: string,
  forward: (inputs: Raw[], params: P) => Raw,
  vjp: OpVjp<P> | null,
  rules: Omit<PrimitiveSpec<P>, 'id' | 'impl' | 'vjp'> = {},
): Op<P> {
  return definePrimitive<P>({ id: name, impl: forward, vjp, ...rules })
}

// ── Batching helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A batched value with its batch axis moved to the front and `pad` axes of length 1 inserted after it, so that it
 * broadcasts (NumPy rules, aligned from the right) against unbatched values of higher rank.
 *
 * @param v The batched value.
 * @param axis The index of its batch axis.
 * @param pad The number of axes of length 1 to insert after the batch axis (default none).
 * @returns `v` with the batch axis first and `pad` axes of length 1 after it.
 *
 * @example Move batch axis 1 to the front and pad one axis
 * print('shape:', shapeOfValue(batchToFront(zeros([3, 5]), 1, 1)))
 */
export function batchToFront(v: Value, axis: number, pad = 0): Value {
  const rank = avalOf(v).shape.length
  let out = v
  if (axis !== 0) {
    const order = [axis, ...Array.from({ length: rank }, (_, k) => k).filter((k) => k !== axis)]
    out = permute(out, order)
  }
  if (pad > 0) {
    const shape = avalOf(out).shape
    out = reshape(out, [shape[0], ...new Array<number>(pad).fill(1), ...shape.slice(1)])
  }
  return out
}

/**
 * Batch a broadcasting primitive: move every batch axis to the front, pad each batched value to the rank of the
 * largest example so the batch axes line up, and apply the primitive once. Elementwise primitives batch this way.
 *
 * @param p The primitive to apply.
 * @param values Its inputs, each batched or not.
 * @param axes The batch axis of each input, or null for an unbatched one.
 * @param params The primitive's parameters.
 * @returns The batched result and its batch axis, which is always 0.
 *
 * @example A batch of two vectors plus one unbatched vector
 * const addPrimitive = registry.get('foundation/tensor/add')
 * // Batch axis 1 of a 3 × 2 tensor: the examples are its columns.
 * const xs = tensor([[1, 10], [2, 20], [3, 30]])
 * const [out, axis] = broadcastBatch(addPrimitive, [xs, tensor([100, 200, 300])], [1, null], undefined)
 * print('out =', out)
 * print('batch axis', axis)
 */
export function broadcastBatch<P>(
  p: Primitive<P>,
  values: readonly Value[],
  axes: readonly (number | null)[],
  params: P,
): [Value, number] {
  const ranks = values.map((v, i) => avalOf(v).shape.length - (axes[i] === null ? 0 : 1))
  const rank = Math.max(0, ...ranks)
  const moved = values.map((v, i) => {
    const axis = axes[i]
    return axis === null ? v : batchToFront(v, axis, rank - ranks[i])
  })
  return [apply(p, moved, params), 0]
}

// ── Elementwise primitives ───────────────────────────────────────────────────────────────────────────────────────────

/**
 * Reduce a cotangent `v` (shaped like a broadcast result) to the kind and shape of `like`: summed over broadcast axes,
 * and summed to a number when `like` is a number. A number `v` is broadcast up to a tensor `like`, and a complex `v`
 * of a real `like` keeps its real part.
 *
 * @param v The cotangent, of the broadcast shape of the operation's arguments.
 * @param like The argument it is the cotangent of, whose kind, shape and field the result takes.
 * @returns `v` summed back to the kind and shape of `like`.
 *
 * @example The cotangent of a broadcast row, column and number
 * const g = ones([2, 3])
 * print('to a row:', sumLike(g, zeros([3])))
 * print('to a column:', sumLike(g, zeros([2, 1])))
 * print('to a number:', sumLike(g, 0))
 */
export function sumLike(v: Value, like: Value): Value {
  const target = avalOf(like)
  const w = projectReal(v, target.dtype)
  const value = avalOf(w)
  if (target.number) return value.number ? w : sum(w)
  if (value.number) return broadcastTo(w, target.shape)
  const same = value.shape.length === target.shape.length && value.shape.every((d, k) => d === target.shape[k])
  return same ? w : sumTo(w, target.shape)
}

/** An elementwise function of three broadcast arguments: numbers give a number, tensors a tensor, traced a traced. */
export interface Ternary {
  (a: number, b: number, c: number): number
  (a: Traced, b: Value, c: Value): Traced
  (a: Value, b: Traced, c: Value): Traced
  (a: Value, b: Value, c: Traced): Traced
  (a: Tensor, b: Raw, c: Raw): Tensor
  (a: Raw, b: Tensor, c: Raw): Tensor
  (a: Raw, b: Raw, c: Tensor): Tensor
  (a: Value, b: Value, c: Value): Value
}

/** A derivative of an elementwise primitive with respect to one argument, written with primitives: `(...args, y)`. */
export type ElementwiseDerivative = (...argsAndOutput: Value[]) => Value

/** The specification of an elementwise primitive for `elementwise`. */
export type ElementwiseSpec = {
  /** `module/name`, registered (see `definePrimitive`). */
  readonly id: string
  /** The scalar rule. Its arity (1, 2 or 3) is the primitive's. */
  readonly f: (...x: number[]) => number
  /**
   * One derivative per argument, $\partial y/\partial x_i$ as a function of the arguments and the output $y$, written
   * with primitives so that it can be differentiated again. `'zero'` where $y$ is piecewise constant in that argument
   * (a comparison, a condition); `null` where the primitive is not differentiable in it (doing so throws
   * `NotDifferentiableError`).
   */
  readonly derivative: readonly (ElementwiseDerivative | 'zero' | null)[]
  /**
   * The result dtype rule (default `float`): `same` keeps the promoted dtype (int32 stays int32), `float` turns
   * integers into float64, `bool` for comparisons, `complex` for a complex result from real arguments (`expj`).
   */
  readonly dtype?: DTypeRule
  /**
   * The complex scalar rule (see `ComplexRule`), used whenever the result dtype is complex128. Without one, complex
   * arguments are a `DTypeError`: the primitive is defined on the real line only.
   */
  readonly complex?: ComplexRule
  /**
   * The function is holomorphic, so the derivatives are complex derivatives $f'(z)$, and the derived vjp conjugates
   * them ($\bar{g} \operatorname{conj}(f')$, the $\reals^2$ convention of design K §8.1) while the jvp is
   * $\dot{z} f'$. Required with `complex`.
   */
  readonly holomorphic?: boolean
  /** An arithmetic operation with a dedicated kernel loop (see `Arithmetic` in kernels.ts). Binary only. */
  readonly kernel?: Arithmetic
  /** A forward rule on raw arguments replacing the scalar loop (e.g. `where`). */
  readonly impl?: (args: Raw[]) => Raw
  /** The registry's documentation of the primitive: summary, formula, note (see `PrimitiveDoc`). */
  readonly doc?: PrimitiveDoc
  /** What the generated tests draw, merged over the defaults (second order when differentiable, complex when given). */
  readonly test?: PrimitiveTest
}

/**
 * The result dtype of an elementwise rule: tensors' dtypes promote, numbers are weak scalars (as in `binaryDType`; an
 * abstract number counts as a non-integer), and an int32 result becomes float64 unless the rule maps integers to
 * integers. With no tensor among the arguments, the rule is applied to float64.
 *
 * @param args The arguments, raw values or abstract values.
 * @param rule The primitive's dtype rule (see `ResultRule`).
 * @returns The dtype of the result.
 */
export function elementwiseDType(args: readonly (Raw | Aval)[], rule: DTypeRule): DType {
  let dtype: DType | null = null
  const tensorDType = (v: Raw | Aval): DType | null =>
    typeof v === 'number'
      ? null
      : 'number' in v && typeof v.number === 'boolean'
        ? v.number
          ? null
          : v.dtype
        : v.dtype
  for (const v of args) {
    const d = tensorDType(v)
    if (d !== null) dtype = dtype === null ? d : promote(dtype, d)
  }
  if (dtype === null) return resultType(rule, 'float64')
  for (const v of args) if (tensorDType(v) === null) dtype = scalarDType(typeof v === 'number' ? v : 0.5, dtype)
  return resultType(rule, dtype)
}

/**
 * Apply a scalar rule of any arity elementwise to broadcast raw arguments: numbers give a number (a rank-0 complex
 * tensor for a complex result). A complex result uses the complex rule, which a real-only primitive lacks: it then
 * throws `DTypeError`. More than three arguments throw `AifnError`.
 *
 * @param name The primitive's name, for error messages.
 * @param args The raw arguments (numbers and tensors), broadcast together.
 * @param f The scalar rule, applied to real arguments.
 * @param rule The dtype rule of the result.
 * @param kernel An arithmetic operation with a dedicated loop, for two arguments; left out, `f` is called per element.
 * @param complex The complex scalar rule, used when the result is complex128.
 * @returns A number when every argument is a number (and the result is real), otherwise a tensor of the broadcast
 *   shape.
 */
function elementwiseRaw(
  name: string,
  args: readonly Raw[],
  f: (...v: number[]) => number,
  rule: DTypeRule,
  kernel?: Arithmetic,
  complex?: ComplexRule,
): Raw {
  const numbers = args.every((v) => typeof v === 'number')
  if (numbers && rule !== 'complex') return f(...(args as number[]))
  const dtype = elementwiseDType(args, rule)
  if (dtype === 'complex128') {
    if (complex === undefined) {
      const dtypes = args.map((v) => (typeof v === 'number' ? 'number' : v.dtype))
      throw new DTypeError(name, `${name}: not defined for complex values (got ${dtypes.join(', ')})`, dtypes)
    }
    return complexKernel(args, complex)
  }
  if (args.length === 1) return unaryKernel(args[0] as Tensor, f, dtype)
  if (args.length === 2) return binaryKernel(args[0], args[1], f, dtype, kernel)
  if (args.length === 3) return ternaryKernel(args[0], args[1], args[2], f, dtype)
  throw new AifnError('elementwise', `elementwise: arity ${args.length} is not supported`)
}

/**
 * $g \cdot d$, skipping the multiplication when either factor is the number 1 (a seed, or the derivative of add) or
 * $d$ is $-1$ (negating instead). A number $g$ means a number output, whose arguments are all numbers, so $d$ already
 * has the right kind.
 *
 * @param g The cotangent or tangent.
 * @param d The derivative it is multiplied by.
 * @returns $g \cdot d$.
 */
function scale(g: Value, d: Value): Value {
  if (g === 1) return d
  if (d === 1) return g
  if (d === -1) return neg(g)
  return mul(g, d)
}

/**
 * The rules of an elementwise primitive derived from its derivatives (design K §4.2): the vjp of argument $x_i$ is
 * $g\,\partial_i$ ($g \operatorname{conj}(\partial_i)$ when holomorphic) summed back to the shape of $x_i$ by
 * `sumLike`, the jvp is $\sum_i t_i\,\partial_i$ (fitted to the output), and the shape rule is that of broadcasting.
 *
 * @param name The primitive's name, for error messages.
 * @param derivative One derivative per argument: a rule of the arguments and the output, `'zero'`, or `null`, where
 *   differentiating throws `NotDifferentiableError`.
 * @param rule The dtype rule of the result, for the shape rule.
 * @param holomorphic Whether the derivatives are complex derivatives, which the vjp conjugates.
 * @returns The derived vjp, jvp and shape rule.
 */
function elementwiseRules(
  name: string,
  derivative: readonly (ElementwiseDerivative | 'zero' | null)[],
  rule: DTypeRule,
  holomorphic: boolean,
): Derived<undefined> {
  const notDifferentiable = (i: number) =>
    new NotDifferentiableError(name, `${name}: no derivative with respect to argument ${i + 1}`, i)
  return {
    vjp: (g, args, y, _params, needed) =>
      args.map((input, i) => {
        if (!needed[i]) return null
        const d = derivative[i]
        if (d === 'zero') return null
        if (d === null) throw notDifferentiable(i)
        const di = d(...args, y)
        // ℝ² convention: ḡ·conj(f′) for a holomorphic f (conj is the identity on real values); sumLike takes the real
        // part for a real argument.
        return sumLike(scale(g, holomorphic ? conj(di) : di), input)
      }),
    jvp: (tangents, args, y) => {
      let acc: Value | null = null
      tangents.forEach((t, i) => {
        if (t === null) return
        const d = derivative[i]
        if (d === 'zero') return
        if (d === null) throw notDifferentiable(i)
        const term = scale(t, d(...args, y))
        acc = acc === null ? term : add(acc, term)
      })
      return acc === null ? null : fitTo(acc, avalOf(y))
    },
    shape: (avals) => ({
      shape: broadcastShapes(...avals.map((a) => a.shape)),
      dtype: elementwiseDType(avals, rule),
      number: avals.every((a) => a.number),
    }),
  }
}

/**
 * Define an elementwise primitive from its scalar rule and **one** derivative per argument (design K §4.2): the vjp,
 * the jvp, the batching rule and the shape rule are all derived from them, and it is differentiable to any order
 * because the derivatives are primitives. Numbers give numbers and tensors broadcast (NumPy rules). Throws
 * `AifnError` unless there are 1 to 3 derivatives, at least as many as `f` takes arguments, and unless a complex rule
 * comes with `holomorphic`.
 *
 * @param spec The primitive: its `id`, scalar rule `f`, one derivative per argument, and optionally a dtype rule, a
 *   complex rule, a forward rule on raw values and metadata.
 * @returns The primitive as a function of 1, 2 or 3 broadcast arguments (as many as there are derivatives).
 *
 * @example Softplus with its derivative, differentiable twice
 * // A bare id defines a local primitive, which is not registered.
 * const softplus = elementwise({
 *   id: 'softplus',
 *   f: (x) => Math.log1p(Math.exp(x)),
 *   derivative: [(x) => div(1, add(1, exp(neg(x))))],
 * })
 * print('softplus([0, 1]) =', softplus(tensor([0, 1])))
 * print('slope at 0 =', grad(softplus)(0))
 * print('curvature at 0 =', grad(grad(softplus))(0))
 */
export function elementwise(spec: ElementwiseSpec & { readonly f: (x: number) => number }): Unary
export function elementwise(spec: ElementwiseSpec & { readonly f: (a: number, b: number) => number }): Binary
export function elementwise(spec: ElementwiseSpec): Ternary
export function elementwise(spec: ElementwiseSpec): Unary | Binary | Ternary {
  const arity = spec.derivative.length
  if (arity < 1 || arity > 3 || spec.f.length > arity) {
    throw new AifnError(
      'elementwise',
      `elementwise: ${spec.id} needs one derivative per argument (1 to 3), got ${arity} for f of arity ${spec.f.length}`,
    )
  }
  const rule: DTypeRule = spec.dtype ?? 'float'
  const name = spec.id.slice(spec.id.lastIndexOf('/') + 1)
  if (spec.complex !== undefined && spec.holomorphic !== true)
    throw new AifnError('elementwise', `elementwise: ${spec.id} has a complex rule, so it must be holomorphic`)
  const zero = spec.derivative.every((d) => d === 'zero')
  const none = spec.derivative.every((d) => d === null)
  const f = spec.f
  const impl = spec.impl ?? ((args: Raw[]) => elementwiseRaw(name, args, f, rule, spec.kernel, spec.complex))
  const derived = elementwiseRules(name, spec.derivative, rule, spec.holomorphic === true)
  const op = build<undefined>(
    {
      id: spec.id,
      kind: 'elementwise',
      arity,
      dtype: rule,
      impl: (args) => impl(args),
      zeroDerivative: zero,
      ...(none ? { vjp: null } : {}),
      differentiable: spec.derivative.map((d) => d !== null && d !== 'zero'),
      doc: spec.doc,
      test: { secondOrder: !none && !zero, complex: spec.complex !== undefined, ...spec.test },
    },
    none ? { shape: derived.shape } : derived,
  ).apply
  // Scalar fast paths: numbers are never traced, so all-number arguments go straight to the scalar rule (except for a
  // complex result, which a number cannot hold).
  if (rule === 'complex') return ((...args: Value[]) => op(args, undefined)) as Unary & Binary & Ternary
  if (arity === 1) return ((x: Value) => (typeof x === 'number' ? f(x) : op([x], undefined))) as Unary
  if (arity === 2)
    return ((a: Value, b: Value) =>
      typeof a === 'number' && typeof b === 'number' ? f(a, b) : op([a, b], undefined)) as Binary
  return ((a: Value, b: Value, c: Value) =>
    typeof a === 'number' && typeof b === 'number' && typeof c === 'number'
      ? f(a, b, c)
      : op([a, b, c], undefined)) as Ternary
}

/** The result kind of a tensor-valued primitive of `X`: traced when `X` is traced, otherwise a tensor. */
export type TensorResult<X> = X extends Traced ? Traced : Tensor

/** The result kind of a number-valued primitive of `X` (e.g. a full reduction): traced or a number. */
export type NumberResult<X> = X extends Traced ? Traced : number

/** The result kind of a primitive of two inputs whose untraced result is `R`. */
export type Result2<A, B, R> = A extends Traced ? Traced : B extends Traced ? Traced : R
