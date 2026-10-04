/**
 * Reductions along axes, as NumPy's: `axis` is one axis, several, or (omitted or null) all of them, and `keepDims`
 * keeps reduced axes with length 1. Reducing every axis without `keepDims` returns a number. Sums and statistics
 * accumulate and return float64; `max` and `min` keep the dtype. Each is a primitive with its derivative rule, or a
 * composition of primitives.
 *
 * Complex values: `sum`, `mean` and `cumsum` accept complex128 (reducing the real and imaginary views separately), and
 * a complex reduction over every axis gives a rank-0 complex tensor, since a number cannot hold it. `norm` takes the
 * modulus first. The reductions that need an ordering or are defined on the reals only (max, min, argmax, argmin,
 * prod, logsumexp, variance) raise `DTypeError`.
 */

import { AifnError, DTypeError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  forEachOffset2,
  fromData,
  normaliseAxes,
  normaliseAxis,
  rowMajorStrides,
  size,
  sizeOf,
  type Axes,
  type Tensor,
} from './core'
import { div, equalTo, mul, abs, square, sqrt, pow, exp, sub, notEqualTo, where } from './elementwise'
import { joinComplex, pairwiseSum, reducedShape, reduceKernel, splitComplex, type GroupReducer } from './kernels'
import {
  batchToFront,
  definePrimitive,
  fitTo,
  type Op,
  type OpBatch,
  type PrimitiveSpec,
  type Raw,
  type ShapeRule,
  type TensorResult,
} from './primitive'
import type { PrimitiveTest } from './registry'
import type { Reduction } from 'aifn-compute/foundation/contracts'
import { broadcastTo, reshape, shapeOfValue } from './structure'
import { avalOf, isTraced, type Traced, type Value } from './trace'

export type { Reduction }

type ReduceParams = { axis: Axes | null | undefined; keepDims: boolean }

/** True when the reduction covers every axis and drops them, so the result is a number. */
function toNumber({ axis, keepDims }: ReduceParams): boolean {
  return (axis === undefined || axis === null) && !keepDims
}

/**
 * Run a per-group reducer as a reduction on a raw input. A complex input is refused (`DTypeError` naming `name`)
 * unless `complex` (the reducer is linear, like a sum): then the real and imaginary views are reduced separately and
 * the result is a complex tensor (rank 0 for a full reduction).
 */
function reduceRaw(
  x: Raw,
  p: ReduceParams,
  fn: GroupReducer,
  keepDType = false,
  name = 'reduce',
  complex = false,
): Raw {
  if (typeof x === 'number') {
    if (p.axis !== undefined && p.axis !== null) normaliseAxes(p.axis, 0, 'reduce')
    return fn([x], 0, 1)
  }
  if (x.dtype === 'complex128') {
    if (!complex) throw new DTypeError(name, `${name}: not defined for complex values (got complex128)`, [x.dtype])
    const [re, im] = splitComplex(x)
    const part = (t: Tensor) => reduceKernel(t, p.axis, p.keepDims, fn, 'float64')
    return joinComplex(part(re), part(im!))
  }
  const out = reduceKernel(x, toNumber(p) ? null : p.axis, p.keepDims, fn, keepDType ? x.dtype : 'float64')
  return toNumber(p) ? out.data[0] : out
}

/** Number of elements reduced into each output. */
function groupSize(x: Value, axis: Axes | null | undefined): number {
  const shape = shapeOfValue(x)
  return sizeOf(normaliseAxes(axis, shape.length, 'reduce').map((k) => shape[k]))
}

/** A reduced value (output or cotangent) broadcast back to the input's shape. */
function expand(v: Value, x: Value, p: ReduceParams): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 0) return v
  const keep = reducedShape(shape, normaliseAxes(p.axis, shape.length, 'reduce'), true)
  return broadcastTo(reshape(v, keep), shape)
}

/** Test cases shared by the reductions: all axes, one axis, and one axis kept, on a 2×3 input. */
function reductionCases(extra: unknown[]): PrimitiveTest['cases'] {
  return (draw) => [
    { inputs: [draw([2, 3])], params: { axis: null, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: 1, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: [0], keepDims: true, extra } },
  ]
}

type ReductionParams<E> = ReduceParams & { extra: E }

// Every reduction shares its parameters, so they share one shape rule and one batching rule (design K §4.2).

/**
 * The shape rule of a reduction: the reduced shape, float64 (or the input dtype for max and min; complex128 stays
 * complex, and a complex full reduction is a rank-0 tensor rather than a number).
 */
function reductionShape<E>(keepDType: boolean): ShapeRule<ReductionParams<E>> {
  return ([x], p) => {
    const axes = normaliseAxes(p.axis, x.shape.length, 'reduce')
    const complex = x.dtype === 'complex128'
    return {
      shape: toNumber(p) ? [] : reducedShape(x.shape, axes, p.keepDims),
      dtype: keepDType || complex ? x.dtype : 'float64',
      number: toNumber(p) && (x.number || !complex),
    }
  }
}

/**
 * The batching rule of a reduction: move the batch axis to the front and reduce each example's axes, shifted by one.
 * Reducing every axis of an example to a number becomes a reduction to one value per example.
 */
function reductionBatch<E>(
  op: (inputs: readonly Value[], p: ReductionParams<E>) => Value,
): OpBatch<ReductionParams<E>> {
  return ([x], [axis], p) => {
    const rank = avalOf(x).shape.length - 1
    const axes = normaliseAxes(p.axis, rank, 'reduce').map((k) => k + 1)
    return [op([batchToFront(x, axis ?? 0)], { ...p, axis: axes, keepDims: p.keepDims }), 0]
  }
}

/**
 * The forward rule of a reduction as the contraction of the tangent with a weight shaped like the input:
 * ẏ = Σ over the reduced axes of t·w (the jvp of every non-linear reduction here has this form).
 */
function contract(t: Value, weight: Value, p: ReduceParams): Value {
  return sum(mul(t, weight), p.axis, p.keepDims)
}

/**
 * A reduction whose derivative is ∂y/∂x = w(x, y) elementwise (a weight shaped like the input): vjp g·w (g expanded
 * back over the reduced axes) and jvp Σ t·w. The weight is written with primitives and never reads concrete values,
 * so the rules work under every transform, `vmap` included.
 */
type Weight = (x: Value, y: Value, p: ReduceParams) => Value

function reduction<E extends unknown[]>(
  name: string,
  forward: (x: Raw, p: ReduceParams, extra: E) => Raw,
  weight: Weight | null,
  summary: string,
  extra: unknown[] = [],
  rules: Omit<PrimitiveSpec<ReductionParams<E>>, 'id' | 'impl' | 'vjp'> & { keepDType?: boolean } = {},
): Reduction<E> {
  const { keepDType = false, ...more } = rules
  const op: Op<ReductionParams<E>> = definePrimitive<ReductionParams<E>>({
    id: `foundation/tensor/${name}`,
    arity: 1,
    impl: ([x], p) => forward(x, p, p.extra),
    ...(weight && {
      vjp: (g, [x], y, p) => [mul(expand(g, x, p), weight(x, y, p))],
      jvp: ([t], [x], y, p) => (t === null ? null : contract(t, weight(x, y, p), p)),
    }),
    shape: reductionShape<E>(keepDType),
    batch: reductionBatch<E>((inputs, p) => op(inputs, p)),
    doc: { summary },
    test: { secondOrder: true, cases: reductionCases(extra), complex: name === 'sum' },
    ...more,
  })
  return ((x: Value, axis?: Axes | null, keepDims = false, ...extra: E) =>
    op([x], { axis, keepDims, extra })) as Reduction<E>
}

/**
 * Sum of elements, by pairwise summation above 128 terms per group (error O(ε log n), as NumPy). Complex values sum
 * their real and imaginary parts; a complex sum over every axis is a rank-0 complex tensor.
 */
export const sum: Reduction = reduction(
  'sum',
  (x, p) => reduceRaw(x, p, pairwiseSum, false, 'sum', true),
  null,
  'The sum of the elements.',
  [],
  // A worked example of a linear primitive: the transpose broadcasts the cotangent back over the summed axes; the vjp
  // and the jvp (the sum of the tangent) are derived from it.
  // fitTo keeps the input's kind: a rank-0 tensor input gets a rank-0 cotangent, not the number a full sum gives.
  { linear: 'linear', transpose: (ct, [x], _which, p) => fitTo(expand(ct, x, p), avalOf(x)), dtype: 'float' },
)

/** Arithmetic mean of elements. */
export const mean: Reduction = ((x: Value, axis?: Axes | null, keepDims = false) =>
  div(sum(x, axis, keepDims), groupSize(x, axis))) as Reduction

/**
 * Product of elements. Its derivative ∂y/∂xᵢ is the product of the other elements, computed without dividing by zero:
 * y/xᵢ where xᵢ ≠ 0; at a zero, the product of the non-zero elements if it is the group's only zero, else 0. The
 * weight is itself differentiable: first and second derivatives are exact for any number of zeros; third and higher
 * derivatives are exact except in groups with three or more zeros, where they are taken as 0.
 */
export const prod: Reduction = reduction(
  'prod',
  (x, p) =>
    reduceRaw(
      x,
      p,
      (v, start, width) => {
        let s = 1
        for (let i = start, end = start + width; i < end; i++) s *= v[i]
        return s
      },
      false,
      'prod',
    ),
  (x, y, p) => {
    const zero = equalTo(x, 0)
    const safe = where(zero, 1, x)
    const zeros = expand(sum(zero, p.axis, true), x, p)
    const nonZeroProduct = expand(prod(safe, p.axis, true), x, p)
    // At a zero xᵢ the weight is P·Πⱼ xⱼ over the group's other zeros j (P the product of the non-zeros). Its value is
    // P for a lone zero and 0 otherwise, but with exactly two zeros its derivative in the other zero xⱼ is P, the
    // mixed second derivative ∂²y/∂xᵢ∂xⱼ. The sum of the other zeros (all 0) carries that derivative exactly.
    const otherZeros = sub(expand(sum(where(zero, x, 0), p.axis, true), x, p), x)
    const atZero = where(
      equalTo(zeros, 1),
      nonZeroProduct,
      where(equalTo(zeros, 2), mul(nonZeroProduct, otherZeros), 0),
    )
    return where(zero, atZero, div(expand(y, x, p), safe))
  },
  'The product of the elements.',
)

function extreme(name: string, better: (a: number, b: number) => boolean, summary: string): Reduction {
  return reduction(
    name,
    (x, p) =>
      reduceRaw(
        x,
        p,
        (v, start, width) => {
          if (width === 0) throw new AifnError(name, `${name}: empty reduction`)
          let best = v[start]
          for (let i = start, end = start + width; i < end; i++) {
            if (v[i] !== v[i]) return NaN
            if (better(v[i], best)) best = v[i]
          }
          return best
        },
        true,
        name,
      ),
    // The derivative goes to the extreme elements, split equally among ties.
    (x, y, p) => {
      const hit = equalTo(x, expand(y, x, p))
      return div(hit, expand(sum(hit, p.axis, true), x, p))
    },
    summary,
    [],
    { keepDType: true },
  )
}

/** Largest element (NaN if any element is NaN). Ties share the derivative equally. */
export const max: Reduction = extreme('max', (a, b) => a > b, 'The largest element.')

/** Smallest element (NaN if any element is NaN). Ties share the derivative equally. */
export const min: Reduction = extreme('min', (a, b) => a < b, 'The smallest element.')

/**
 * log Σ exp(x), computed as m + log Σ exp(x − m) with m the maximum so that it neither overflows nor underflows
 * (−∞ when every element is −∞). Its derivative is the softmax of x; a group of −∞ alone has derivative 0, and one
 * holding +∞ splits it equally among its +∞ entries.
 */
export const logsumexp: Reduction = reduction(
  'logsumexp',
  (x, p) =>
    reduceRaw(
      x,
      p,
      (v, start, width) => {
        const end = start + width
        let m = -Infinity
        for (let i = start; i < end; i++) if (v[i] > m || v[i] !== v[i]) m = v[i]
        if (m !== m) return NaN
        if (m === -Infinity || m === Infinity) return m
        let s = 0
        for (let i = start; i < end; i++) s += Math.exp(v[i] - m)
        return m + Math.log(s)
      },
      false,
      'logsumexp',
    ),
  // The derivative is the softmax of x along the reduced axes. Where a group is entirely −∞ (a fully masked row) the
  // softmax is 0/0; the subgradient is taken as 0. Where it holds +∞ the derivative goes to the +∞ entries, split
  // equally (the limit of the softmax). Both branches are kept finite so second derivatives stay finite too.
  (x, y, p) => {
    const ye = expand(y, x, p)
    const up = equalTo(ye, Infinity)
    const down = equalTo(ye, -Infinity)
    const finite = (v: Value) => where(up, 0, where(down, 0, v))
    const soft = exp(sub(finite(x), finite(ye)))
    const hit = equalTo(x, ye)
    const share = div(hit, expand(sum(hit, p.axis, true), x, p))
    return where(down, 0, where(up, share, soft))
  },
  'log Σ exp(x), without overflow.',
)

/** The fused two-pass variance of one group: the mean first, then the centred sum of squares (Welford-free). */
function groupVariance(ddof: number): GroupReducer {
  return (v, start, width) => {
    const m = pairwiseSum(v, start, width) / width
    let s = 0
    for (let i = start, end = start + width; i < end; i++) s += (v[i] - m) * (v[i] - m)
    return s / (width - ddof)
  }
}

/**
 * Variance Σ(x − x̄)² / (n − ddof). `ddof` = 0 (default) is the population variance and 1 the unbiased sample
 * variance. A composition (design K §3.4, T9): traced inputs go through `mean` and `square`, so the derivative is theirs
 * and differentiable to any order; raw inputs take a fused two-pass kernel with the same value.
 */
export const variance: Reduction<[ddof?: number]> = ((x: Value, axis?: Axes | null, keepDims = false, ddof = 0) => {
  if (!isTraced(x)) return reduceRaw(x as Raw, { axis, keepDims }, groupVariance(ddof), false, 'variance')
  if (avalOf(x).dtype === 'complex128')
    throw new DTypeError('variance', 'variance: not defined for complex values (got complex128)', ['complex128'])
  const n = groupSize(x, axis)
  const centred = sub(x, mean(x, axis, true))
  return div(sum(square(centred), axis, keepDims), n - ddof)
}) as Reduction<[ddof?: number]>

/** Standard deviation, the square root of `variance` (same `ddof`). */
export const std: Reduction<[ddof?: number]> = ((x: Value, axis?: Axes | null, keepDims = false, ddof = 0) =>
  sqrt(variance(x, axis, keepDims, ddof))) as Reduction<[ddof?: number]>

/**
 * √Σx² of one group without overflow or underflow: the sum of squares of x / s with s the largest |x| (LAPACK's
 * `dnrm2` scaling, Blue 1978), times s.
 */
const groupEuclidean: GroupReducer = (v, start, width) => {
  const end = start + width
  let scale = 0
  for (let i = start; i < end; i++) {
    const a = Math.abs(v[i])
    if (a !== a) return NaN
    if (a > scale) scale = a
  }
  if (scale === 0 || scale === Infinity) return scale
  let s = 0
  for (let i = start; i < end; i++) {
    const r = v[i] / scale
    s += r * r
  }
  return scale * Math.sqrt(s)
}

/**
 * The Euclidean norm √Σx² along axes: the stable kernel of `norm` (no overflow for large entries, no underflow for
 * small ones). Its derivative is x/‖x‖, taken as 0 where ‖x‖ = 0 (the minimum-norm subgradient), so a zero vector has
 * a zero gradient rather than NaN.
 */
const euclidean: Reduction = reduction(
  'euclideanNorm',
  (x, p) => reduceRaw(x, p, groupEuclidean, false, 'norm'),
  // x/‖x‖, and 0 where ‖x‖ = 0 (the minimum-norm subgradient).
  (x, y, p) => {
    const zero = equalTo(y, 0)
    return where(expand(zero, x, p), 0, div(x, expand(where(zero, 1, y), x, p)))
  },
  'The Euclidean norm √Σx², scaled against overflow.',
)

/**
 * Vector p-norm of the elements along `axis` (of all elements when omitted, which for a matrix is the Frobenius
 * norm): `ord` = 2 (default; the scaled kernel above, so no overflow and a zero gradient at 0), 1, ∞ (largest |x|),
 * −∞ (smallest |x|), 0 (count of non-zeros) or any p > 0. At x = 0 the gradient is 0 for every order p > 1.
 */
export const norm: Reduction<[ord?: number]> = ((x0: Value, axis?: Axes | null, keepDims = false, ord = 2) => {
  // A complex vector's norms are those of its moduli.
  const x = avalOf(x0).dtype === 'complex128' ? abs(x0) : x0
  if (ord === 2) return euclidean(x, axis, keepDims)
  if (ord === 1) return sum(abs(x), axis, keepDims)
  if (ord === Infinity) return max(abs(x), axis, keepDims)
  if (ord === -Infinity) return min(abs(x), axis, keepDims)
  if (ord === 0) return sum(notEqualTo(x, 0), axis, keepDims)
  if (!(ord > 0)) throw new AifnError('norm', `norm: unsupported order ${ord}`)
  // (Σ|x|ᵖ)^{1/p}, with the root guarded at 0: its derivative there is ∞ · 0, so the zero group takes the
  // minimum-norm subgradient 0 instead (for p > 1; for p < 1 |x|ᵖ itself has an infinite slope at 0).
  const s = sum(pow(abs(x), ord), axis, keepDims)
  const zero = equalTo(s, 0)
  return where(zero, 0, pow(where(zero, 1, s), 1 / ord))
}) as Reduction<[ord?: number]>

type ArgParams = { axis: number | undefined; keepDims: boolean }

/** An index reduction: a number without `axis`, else an int32 tensor; traced inputs give traced results. */
export interface ArgReduction {
  (x: number | Tensor): number
  (x: Tensor, axis: number, keepDims?: boolean): Tensor
  (x: Traced, axis?: number, keepDims?: boolean): Traced
  (x: Value, axis?: number, keepDims?: boolean): Value
}

/**
 * Index of the extreme element in each group, first occurrence winning; a NaN counts as the extreme. A primitive with
 * zero derivative (piecewise constant), so it can be batched and appear inside differentiated code.
 */
function argExtreme(name: string, better: (a: number, b: number) => boolean): ArgReduction {
  const pick: GroupReducer = (v, start, width) => {
    if (width === 0) throw new AifnError(name, `${name}: empty reduction`)
    let best = start
    for (let i = start, end = start + width; i < end; i++) {
      if (v[i] !== v[i]) return i - start
      if (better(v[i], v[best])) best = i
    }
    return best - start
  }
  const op: Op<ArgParams> = definePrimitive<ArgParams>({
    id: `foundation/tensor/${name}`,
    arity: 1,
    impl: ([x], { axis, keepDims }) => {
      if (typeof x === 'number') return 0
      if (x.dtype === 'complex128')
        throw new DTypeError(name, `${name}: complex values have no ordering (got complex128)`, [x.dtype])
      if (axis === undefined) return reduceKernel(x, null, false, pick, 'int32').data[0]
      return reduceKernel(x, axis, keepDims, pick, 'int32')
    },
    zeroDerivative: true,
    dtype: 'index',
    shape: ([x], { axis, keepDims }) =>
      axis === undefined
        ? { shape: [], dtype: 'int32', number: true }
        : {
            shape: reducedShape(x.shape, normaliseAxes(axis, x.shape.length, name), keepDims),
            dtype: 'int32',
            number: false,
          },
    // A flat index over the whole example is an index along the second axis of the batch flattened per example.
    batch: ([x], [b], { axis, keepDims }) => {
      const front = batchToFront(x, b ?? 0)
      const shape = shapeOfValue(front)
      if (axis === undefined) return [op([reshape(front, [shape[0], -1])], { axis: 1, keepDims: false }), 0]
      return [op([front], { axis: normaliseAxis(axis, shape.length - 1, name) + 1, keepDims }), 0]
    },
    doc: { summary: `The index of the ${name === 'argmax' ? 'largest' : 'smallest'} element.` },
    test: { cases: (draw) => [{ inputs: [draw([2, 3])], params: { axis: 1, keepDims: false } }] },
  })
  return ((x: Value, axis?: number, keepDims: boolean = false) => op([x], { axis, keepDims })) as ArgReduction
}

/**
 * Index of the largest element: a flat row-major index when `axis` is omitted, else an int32 tensor of indices along
 * `axis`. The first occurrence wins ties. Piecewise constant: its derivative is zero, and inside `vmap` it is batched.
 */
export const argmax: ArgReduction = argExtreme('argmax', (a, b) => a > b)

/** Index of the smallest element; see `argmax`. */
export const argmin: ArgReduction = argExtreme('argmin', (a, b) => a < b)

// ── Cumulative sum ───────────────────────────────────────────────────────────────────────────────────────────────────

type CumsumParams = { axis: number; reverse: boolean }

/** Running sums along one axis (from the end with `reverse`), float64 (complex128 for complex input). */
function cumsumKernel(x: Tensor, axis: number, reverse: boolean): Tensor {
  if (x.dtype === 'complex128') {
    const [re, im] = splitComplex(x)
    return joinComplex(cumsumKernel(re, axis, reverse), cumsumKernel(im!, axis, reverse))
  }
  const a = normaliseAxis(axis, x.shape.length, 'cumsum')
  const n = x.shape[a]
  const outStrides = rowMajorStrides(x.shape)
  const out = new Float64Array(size(x))
  const lines = x.shape.map((d, k) => (k === a ? 1 : d))
  const src = x.data
  const [sx, so] = [x.strides[a], outStrides[a]]
  // One line per position of the other axes: accumulate along axis a.
  forEachOffset2(lines, x.strides, x.offset, outStrides, 0, (i, o) => {
    let s = 0
    for (let j = 0; j < n; j++) {
      const q = reverse ? n - 1 - j : j
      s += src[i + q * sx]
      out[o + q * so] = s
    }
  })
  return fromData(out, x.shape)
}

// Linear: its transpose is the cumulative sum in the other direction (the adjoint of a lower-triangular matrix of
// ones is the upper one).
const cumsumOp: Op<CumsumParams> = definePrimitive<CumsumParams>({
  id: 'foundation/tensor/cumsum',
  arity: 1,
  dtype: 'float',
  impl: ([x], { axis, reverse }) => {
    if (typeof x === 'number') throw new ShapeError('cumsum', 'cumsum: needs a tensor of rank ≥ 1, got a number')
    if (x.shape.length === 0) throw new ShapeError('cumsum', 'cumsum: needs a tensor of rank ≥ 1')
    return cumsumKernel(x, axis, reverse)
  },
  linear: 'linear',
  transpose: (ct, _xs, _which, p) => cumsumOp([ct], { ...p, reverse: !p.reverse }),
  shape: ([x], { axis }) => {
    normaliseAxis(axis, x.shape.length, 'cumsum')
    return { shape: x.shape, dtype: x.dtype === 'complex128' ? 'complex128' : 'float64', number: false }
  },
  batch: ([x], [b], p) => {
    const rank = avalOf(x).shape.length - 1
    return [cumsumOp([batchToFront(x, b ?? 0)], { ...p, axis: normaliseAxis(p.axis, rank, 'cumsum') + 1 }), 0]
  },
  doc: { summary: 'Running sums along an axis.', formula: 'y_k = \\sum_{j \\le k} x_j' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([5])], params: { axis: 0, reverse: false } },
      { inputs: [draw([2, 3])], params: { axis: 1, reverse: true } },
    ],
  },
})

/**
 * Running sums along `axis` (default the last): y[k] = Σ_{j ≤ k} x[j], or Σ_{j ≥ k} with `reverse`. float64, or
 * complex128 for complex input. Linear; its adjoint is the reverse cumulative sum.
 */
export function cumsum<X extends Value>(x: X, axis = -1, reverse = false): TensorResult<X> {
  return cumsumOp([x], { axis, reverse }) as TensorResult<X>
}
