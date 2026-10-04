/**
 * Internal helpers for the derivative and batching rules of the linear-algebra primitives: constant masks, the
 * symmetric reading of a lower triangle and its adjoint, packing several outputs into the one output of a primitive,
 * the F-matrix of spectral rules with its degeneracy check, and batching by folding or looping.
 */

import {
  abs,
  add,
  avalOf,
  batchToFront,
  broadcastTo,
  concat,
  definePrimitive,
  div,
  flatten,
  float64Data,
  fromData,
  greater,
  max,
  mul,
  permute,
  reshape,
  shapeOfValue,
  slice,
  stack,
  sub,
  sum,
  toFlat,
  transpose,
  unwrap,
  registry,
  type Aval,
  type Op,
  type OpBatch,
  type Primitive,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { AifnError, NumericalError } from 'aifn-compute/foundation/errors'
import { BatchInterpreter, batchExamples } from 'aifn-compute/foundation/autodiff'
import { EPS, matrix, vector } from './dense'

/** The concrete value behind x, or null inside `vmap` (where a value holds a whole batch and has no single value). */
export function concrete(x: Value): Raw | null {
  try {
    return unwrap(x)
  } catch (e) {
    if (e instanceof AifnError) return null
    throw e
  }
}

/** A constant n×n mask with entries `f(i, j)`. */
export function mask(n: number, f: (i: number, j: number) => number, m = n): Value {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = f(i, j)
  return matrix(out, m, n)
}

/** Φ: the lower triangle with its diagonal halved (Murray, 2016). */
export const phiMask = (n: number): Value => mask(n, (i, j) => (i === j ? 0.5 : j < i ? 1 : 0))
/** The lower triangle, with or without the diagonal. */
export const lowerMask = (n: number, diagonal = true): Value =>
  mask(n, (i, j) => (j < i || (diagonal && i === j) ? 1 : 0))
/** The upper triangle, with or without the diagonal (m×n, for trapezoids). */
export const upperMask = (n: number, diagonal = true, m = n): Value =>
  mask(n, (i, j) => (j > i || (diagonal && i === j) ? 1 : 0), m)

/** The symmetric matrix a function reading only the lower triangle of X sees: tril(X) + tril(X, −1)ᵀ. */
export function symmetricFromLower(x: Value, n: number): Value {
  return add(mul(x, lowerMask(n)), transpose(mul(x, lowerMask(n, false))))
}

/** The adjoint of `symmetricFromLower`: the cotangent of X given a cotangent G of the symmetric matrix, Φ(G + Gᵀ). */
export function lowerAdjoint(g: Value, n: number): Value {
  return mul(add(g, transpose(g)), phiMask(n))
}

/** A vector or a matrix as a matrix (a vector becomes a column). */
export const column = (v: Value): Value => (shapeOfValue(v).length === 1 ? reshape(v, [-1, 1]) : v)

/** The number of entries of a shape. */
const count = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

/**
 * Pack the parts of a multi-output primitive into its one flat output (raw), so every transform sees one primitive
 * application; `unpack` reads them back.
 */
export function packRaw(parts: readonly Float64Array[]): Raw {
  let length = 0
  for (const p of parts) length += p.length
  const out = new Float64Array(length)
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return vector(out)
}

/** The parts of a packed output (or of its tangent or cotangent), by slicing and reshaping (primitives). */
export function unpack(v: Value, shapes: readonly (readonly number[])[]): Value[] {
  let at = 0
  return shapes.map((shape) => {
    const size = count(shape)
    const part = reshape(slice(v, [at, at + size]), shape)
    at += size
    return part
  })
}

/** Pack parts (values of any level) into one flat vector, the tangent of a packed output. */
export function pack(parts: readonly Value[]): Value {
  return concat(
    parts.map((p) => flatten(p)),
    0,
  )
}

/**
 * The concrete row-major data of each example of v (see `batchExamples` in `aifn-compute/foundation/autodiff`): one entry
 * outside `vmap`, one per batch element inside it, so that checks on values run example by example. Null when there is
 * no concrete value.
 */
export function concreteExamples(v: Value): ArrayLike<number>[] | null {
  const raws = batchExamples(v)
  return raws === null ? null : raws.map((r) => (typeof r === 'number' ? [r] : (toFlat(r) as ArrayLike<number>)))
}

/**
 * Example b of per-example data, pairing a value batched by `vmap` with one that is not (a single entry serves every
 * example); undefined when the two cannot be paired.
 */
export function exampleOf<T>(list: readonly T[], b: number): T | undefined {
  return list.length === 1 ? list[0] : list[b]
}

/**
 * The F-matrix of the spectral rules, Fᵢⱼ = 1/(dⱼ − dᵢ) for i ≠ j and 0 on the diagonal, written with primitives so that
 * it can be differentiated again (Giles, 2008, §3.1). Pairs whose gap is within 64·n·ε·max|d| are degenerate: their
 * entries are 0 (the answer when the function is invariant to rotations within the degenerate subspace).
 * `invariant(i, j, b)` decides, from the concrete values of example b, whether that is so for the pair; when it is
 * not, the derivative does not exist and a `NumericalError` ('degenerate') is thrown. Inside `vmap` the check runs on
 * every example (failure is reported per example, never zeroed silently); it is skipped only when no level holds a
 * concrete value (`invariant` null).
 */
export function fMatrix(
  d: Value,
  where: string,
  invariant: ((i: number, j: number, b: number) => boolean) | null,
): Value {
  const n = shapeOfValue(d)[0]
  const gap = sub(reshape(d, [1, n]), reshape(d, [n, 1]))
  const tol = mul(64 * n * EPS, max(abs(d)))
  const offDiagonal = mask(n, (i, j) => (i === j ? 0 : 1))
  // Comparisons have zero derivative, so the mask is a constant to every transform.
  const keep = mul(offDiagonal, greater(abs(gap), tol))
  const examples = invariant === null ? null : concreteExamples(d)
  if (examples !== null && invariant !== null) {
    examples.forEach((values, b) => {
      const t = 64 * n * EPS * scaleOf(values)
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(values[j] - values[i]) <= t && !invariant(i, j, b)) {
            const which = examples.length > 1 ? ` of batch example ${b}` : ''
            throw new NumericalError(
              where,
              `${where}: values ${i} and ${j}${which} are equal (degenerate), and the function is not invariant ` +
                'within their subspace, so it has no derivative there',
              'degenerate',
            )
          }
        }
      }
    })
  }
  // Masked entries divide by 1 and are then zeroed, so no NaN arises.
  return mul(keep, div(1, add(gap, sub(1, keep))))
}

/** The relative test of the degeneracy checks: |value| ≤ 1e-8·scale (scale the size of the quantities compared). */
export function negligible(value: number, scale: number): boolean {
  return Math.abs(value) <= 1e-8 * (scale + 1e-300)
}

/** The concrete row-major data of a matrix value, or null inside vmap. */
export function concreteData(v: Value): ArrayLike<number> | null {
  const c = concrete(v)
  if (c === null) return null
  if (typeof c === 'number') return [c]
  return toFlat(c) as ArrayLike<number>
}

/** Largest absolute entry of concrete data. */
export function scaleOf(a: ArrayLike<number>): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s = Math.max(s, Math.abs(a[k]))
  return s
}

/**
 * Batch a solve with an unbatched matrix by folding the batch into the right-hand side's columns: B examples of
 * b (n or n×r) become one n×(B·r) right-hand side, solved once, and unfolded. `solveWith` applies the primitive.
 */
export function foldColumns(solveWith: (rhs: Value) => Value, b: Value, axis: number, size: number): [Value, number] {
  const front = batchToFront(b, axis)
  const shape = shapeOfValue(front)
  if (shape.length === 2) {
    // [B, n] → [n, B] → solve → [B, n].
    return [transpose(solveWith(transpose(front))), 0]
  }
  const [, n, r] = shape
  const folded = reshape(permute(front, [1, 0, 2]), [n, size * r])
  return [permute(reshape(solveWith(folded), [n, size, r]), [1, 0, 2]), 0]
}

/** The abstract value of a float64 result of shape `shape` (a number when `number`), for shape rules. */
export const float64Aval = (shape: readonly number[], number = false): Aval => ({
  shape: [...shape],
  dtype: 'float64',
  number,
})

/** The parameters of a batched kernel: the base primitive's, which inputs carry the leading batch axis, and its size. */
type KernelParams<P> = { readonly params: P; readonly axes: readonly (0 | null)[]; readonly size: number }

/** The batched kernel of each base primitive, by id (built once, on first use). */
const kernels = new Map<string, Op<KernelParams<unknown>>>()

/**
 * The parameters of a primitive with every per-example cache removed (`factor`, `found`, `failed`: computed by a
 * wrapper for one concrete input, so they belong to no example of a batch).
 */
export function withoutCaches<P>(params: P): P {
  if (params === null || typeof params !== 'object') return params
  const { factor: _f, found: _d, failed: _x, ...rest } = params as Record<string, unknown>
  return rest as P
}

/**
 * The batching rule of a matrix primitive: move every batch axis to the front and apply the primitive's batched kernel
 * once (see `batchedKernel`), which loops over the contiguous examples inside one impl call rather than through the
 * interpreter. The parameters lose their per-example caches (`withoutCaches`), so each example is factored itself.
 */
export function kernelBatch<P>(id: string): OpBatch<P> {
  return (values, axes, params, size) => {
    const front = values.map((v, i) => {
      const axis = axes[i]
      return axis === null ? v : batchToFront(v, axis)
    })
    const kernel = batchedKernel<P>(id)
    return [kernel(front, { params: withoutCaches(params), axes: axes.map((a) => (a === null ? null : 0)), size }), 0]
  }
}

/** The examples of a raw value with a leading batch axis, as views of its contiguous float64 data (no copy). */
function examplesOf(x: Raw, size: number): Raw[] {
  if (typeof x === 'number' || x.dtype === 'complex128') {
    return Array.from({ length: size }, (_, b) => unwrap(slice(x, b)))
  }
  const data = float64Data(x)
  const shape = x.shape.slice(1)
  const per = size === 0 ? 0 : data.length / size
  return Array.from({ length: size }, (_, b) => fromData(data.subarray(b * per, (b + 1) * per), shape))
}

/** Stack raw per-example outputs along a new leading axis: one float64 buffer when they are numbers or float64. */
function stackRaw(outs: readonly Raw[], empty: () => Raw): Raw {
  if (outs.length === 0) return empty()
  const first = outs[0]
  if (typeof first === 'number') {
    if (outs.every((o) => typeof o === 'number')) return fromData(Float64Array.from(outs as number[]), [outs.length])
  } else if (outs.every((o) => typeof o !== 'number' && o.dtype === 'float64')) {
    const per = count(first.shape)
    const out = new Float64Array(outs.length * per)
    outs.forEach((o, b) => out.set(float64Data(o as Tensor), b * per))
    return fromData(out, [outs.length, ...first.shape])
  }
  return unwrap(stack(outs, 0))
}

/**
 * Run a rule of `base` written for one example on batched arguments, under a fresh batch interpreter of `size`
 * examples: `batched[i]` says whether argument i carries the leading batch axis (else it is shared by every example).
 * Each result is returned with its batch axis in front; the result for a shared input (`shared[k]`) is summed over the
 * examples instead, as the cotangent of a value every example reads.
 */
function perExample(
  size: number,
  args: readonly (Value | null)[],
  batched: readonly boolean[],
  numbers: readonly boolean[],
  rule: (args: (Value | null)[]) => readonly (Value | null)[],
  shared: readonly boolean[],
): (Value | null)[] {
  const interpreter = new BatchInterpreter(size)
  const wrapped = args.map((v, i) => (v === null || !batched[i] ? v : interpreter.wrap(v, 0, numbers[i])))
  return rule(wrapped).map((r, k) => {
    if (r === null) return null
    if (shared[k]) return interpreter.owns(r) ? sum(r.value, r.axis) : mul(size, r)
    return interpreter.owns(r) ? batchToFront(r.value, r.axis) : broadcastTo(r, [size, ...avalOf(r).shape])
  })
}

/**
 * The batched kernel of a registered primitive (a local primitive, built once): its inputs marked in `axes` carry a
 * leading batch axis of `size` examples. The impl loops the base impl over the contiguous examples, with no
 * interpreter in between. Its derivative rules are the base rules run on batch tracers (so each check on concrete
 * values, such as the degeneracy of `eigh`, still runs example by example), and a second `vmap` merges its batch axis
 * with the first.
 */
function batchedKernel<P>(id: string): Op<KernelParams<P>> {
  const known = kernels.get(id)
  if (known) return known as Op<KernelParams<P>>
  const base = registry.get(id) as Primitive<P> | undefined
  if (!base) throw new AifnError('vmap', `vmap: no primitive ${id} to batch`)
  const exampleAvals = (avals: readonly Aval[], axes: readonly (0 | null)[]): Aval[] =>
    avals.map((a, i) => (axes[i] === null ? a : { shape: a.shape.slice(1), dtype: a.dtype, number: false }))
  const outputIsNumber = (inputs: readonly Value[], kp: KernelParams<P>): boolean =>
    base.shape !== null && base.shape(exampleAvals(inputs.map(avalOf), kp.axes), kp.params).number
  const op: Op<KernelParams<P>> = definePrimitive<KernelParams<P>>({
    id: `${base.name}Batched`,
    arity: base.arity,
    impl: (inputs, { params, axes, size }) => {
      const views = inputs.map((x, i) => (axes[i] === null ? null : examplesOf(x, size)))
      const outs = Array.from({ length: size }, (_, b) =>
        base.impl(
          inputs.map((x, i) => views[i]?.[b] ?? x),
          params,
        ),
      )
      return stackRaw(outs, () => {
        if (base.shape === null) throw new AifnError('vmap', `vmap: ${base.name} of an empty batch has no shape`)
        const aval = base.shape(exampleAvals(inputs.map(avalOf), axes), params)
        return fromData(new Float64Array(0), [0, ...aval.shape])
      })
    },
    vjp:
      base.vjp &&
      ((g, inputs, out, kp, needed) => {
        const vjp = base.vjp as NonNullable<typeof base.vjp>
        const number = outputIsNumber(inputs, kp)
        const n = inputs.length
        return perExample(
          kp.size,
          [g, out, ...inputs],
          [true, true, ...kp.axes.map((a) => a !== null)],
          [number, number, ...inputs.map(() => false)],
          ([gb, outb, ...xs]) => vjp(gb as Value, xs as Value[], outb as Value, kp.params, needed),
          Array.from({ length: n }, (_, i) => kp.axes[i] === null),
        )
      }),
    jvp:
      base.jvp === null
        ? undefined
        : (tangents, inputs, out, kp) => {
            const jvp = base.jvp as NonNullable<typeof base.jvp>
            const n = inputs.length
            const batched = kp.axes.map((a) => a !== null)
            const [t] = perExample(
              kp.size,
              [out, ...inputs, ...tangents],
              [true, ...batched, ...batched],
              [outputIsNumber(inputs, kp), ...Array.from({ length: 2 * n }, () => false)],
              ([outb, ...rest]) => [jvp(rest.slice(n), rest.slice(0, n) as Value[], outb as Value, kp.params)],
              [false],
            )
            return t
          },
    // A second vmap: move its batch axis to the front of every input, broadcast so that every batched input carries
    // both axes, merge them into one, apply the kernel once and split the axes again.
    batch: (values, axes, kp, size) => {
      const inner = kp.size
      const merged = values.map((v, i) => {
        const [isInner, isOuter] = [kp.axes[i] !== null, axes[i] !== null]
        if (!isInner && !isOuter) return v
        let w = isOuter ? batchToFront(v, axes[i] as number) : reshape(v, [1, ...avalOf(v).shape])
        let shape = avalOf(w).shape
        if (!isInner) w = reshape(w, [shape[0], 1, ...shape.slice(1)])
        shape = avalOf(w).shape
        const example = shape.slice(2)
        return reshape(broadcastTo(w, [size, inner, ...example]), [size * inner, ...example])
      })
      const both = values.map((_, i) => (kp.axes[i] !== null || axes[i] !== null ? 0 : null))
      const out = op(merged, { params: kp.params, axes: both, size: size * inner })
      return [reshape(out, [size, inner, ...avalOf(out).shape.slice(1)]), 0]
    },
    shape:
      base.shape === null
        ? undefined
        : (avals, { params, axes, size }) => {
            const aval = (base.shape as NonNullable<typeof base.shape>)(exampleAvals(avals, axes), params)
            return { shape: [size, ...aval.shape], dtype: aval.dtype, number: false }
          },
  })
  kernels.set(id, op as Op<KernelParams<unknown>>)
  return op
}
