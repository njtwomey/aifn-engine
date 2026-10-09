/**
 * Reductions along axes, as NumPy's: `axis` is one axis, several, or (omitted or null) all of them, and `keepDims`
 * keeps reduced axes with length $1$. Reducing every axis without `keepDims` returns a number. Sums and statistics
 * accumulate and return float64; `max` and `min` keep the dtype. Each is a primitive with its derivative rule, or a
 * composition of primitives.
 *
 * Every non-linear reduction primitive $y = f(\xvec)$ here (`prod`, `max`, `min`, `logsumexp` and the Euclidean norm)
 * has a derivative of one form: a weight $w_i = \partial y / \partial x_i$ shaped like the input, so the reverse rule
 * is $\bar{x}_i = \bar{y}\, w_i$ (the cotangent broadcast back over the reduced axes) and the forward rule is
 * $\dot{y} = \sum_i \dot{x}_i w_i$ over the reduced axes. The linear ones (`sum`, `cumsum`) give their transpose
 * instead, `argmax` and `argmin` have zero derivative, and the rest are compositions.
 *
 * Complex values: `sum`, `mean` and `cumsum` accept complex128 (reducing the real and imaginary views separately), and
 * a complex reduction over every axis gives a rank-0 complex tensor, since a number cannot hold it. `norm` takes the
 * modulus first. The reductions that need an ordering or are defined on the reals only (`max`, `min`, `argmax`,
 * `argmin`, `prod`, `logsumexp`, `variance`) raise `DTypeError`.
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

/**
 * The parameters every reduction shares: `axis`, the axis or axes reduced (`null` or undefined for all of them), and
 * `keepDims`, whether reduced axes stay with length $1$.
 */
type ReduceParams = { axis: Axes | null | undefined; keepDims: boolean }

/**
 * True when the reduction covers every axis and drops them, so the result is a number.
 *
 * @param options The reduction's parameters.
 * @param options.axis The axis or axes reduced; `null` or undefined means all of them.
 * @param options.keepDims Whether reduced axes are kept with length $1$; when true the result is never a number.
 * @returns Whether the result is a number.
 */
function toNumber({ axis, keepDims }: ReduceParams): boolean {
  return (axis === undefined || axis === null) && !keepDims
}

/**
 * Run a per-group reducer as a reduction on a raw input. A complex input is refused (`DTypeError` naming `name`)
 * unless `complex` (the reducer is linear, like a sum): then the real and imaginary views are reduced separately and
 * the result is a complex tensor (rank 0 for a full reduction).
 *
 * @param x The input: a number (one group of one element; any `axis` given must be valid for rank $0$) or a tensor.
 * @param p Which axes to reduce, and whether to keep them.
 * @param fn Folds one group of elements, given as a range of an array, to one number.
 * @param keepDType Whether the result keeps the input's dtype (`max`, `min`) rather than becoming float64.
 * @param name The caller's name, for error messages.
 * @param complex Whether a complex input is allowed, reduced part by part (only for a linear `fn`).
 * @returns A number for a number input or a full reduction without `keepDims` of a real tensor, otherwise a tensor.
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

/**
 * Number of elements reduced into each output.
 *
 * @param x The value being reduced; only its shape is read.
 * @param axis The axis or axes reduced (`null` or undefined for all of them).
 * @returns The product of the lengths of the reduced axes ($1$ for a number).
 */
function groupSize(x: Value, axis: Axes | null | undefined): number {
  const shape = shapeOfValue(x)
  return sizeOf(normaliseAxes(axis, shape.length, 'reduce').map((k) => shape[k]))
}

/**
 * A reduced value (output or cotangent) broadcast back to the input's shape.
 *
 * @param v The reduced value, shaped as the reduction's output (with or without the kept axes).
 * @param x The reduction's input; only its shape is read.
 * @param p The reduction's parameters, which say which axes were reduced.
 * @returns `v` reshaped to keep the reduced axes with length $1$, then broadcast to `x`'s shape (`v` itself when `x`
 *   is a number or rank 0).
 */
function expand(v: Value, x: Value, p: ReduceParams): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 0) return v
  const keep = reducedShape(shape, normaliseAxes(p.axis, shape.length, 'reduce'), true)
  return broadcastTo(reshape(v, keep), shape)
}

/**
 * Test cases shared by the reductions: all axes, one axis, and one axis kept, on a $2 \times 3$ input.
 *
 * @param extra The reduction's extra trailing arguments, passed unchanged in every case.
 * @returns The generator of cases for the reduction's primitive tests.
 */
function reductionCases(extra: unknown[]): PrimitiveTest['cases'] {
  return (draw) => [
    { inputs: [draw([2, 3])], params: { axis: null, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: 1, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: [0], keepDims: true, extra } },
  ]
}

/** The parameters of a reduction primitive: the shared ones, and `extra`, its extra trailing arguments. */
type ReductionParams<E> = ReduceParams & { extra: E }

// Every reduction shares its parameters, so they share one shape rule and one batching rule (design K §4.2).

/**
 * The shape rule of a reduction: the reduced shape, float64 (or the input dtype for max and min; complex128 stays
 * complex, and a complex full reduction is a rank-0 tensor rather than a number).
 *
 * @param keepDType Whether the output keeps the input's dtype rather than becoming float64.
 * @returns The shape rule.
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
 *
 * @param op The reduction primitive, applied once to the whole batch.
 * @returns The batching rule, whose result has its batch axis first.
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
 * $\dot{y} = \sum_i t_i w_i$ over the reduced axes (the jvp of every non-linear reduction here has this form).
 *
 * @param t The tangent $\tvec$ of the input, shaped like it.
 * @param weight The weight $\wvec$ of the reduction, shaped like the input.
 * @param p The reduction's parameters: the same axes are summed, and kept when it keeps them.
 * @returns The tangent $\dot{y}$ of the output, shaped like the output.
 */
function contract(t: Value, weight: Value, p: ReduceParams): Value {
  return sum(mul(t, weight), p.axis, p.keepDims)
}

/**
 * A reduction whose derivative is $\partial y / \partial x_i = w_i(\xvec, y)$ elementwise (a weight shaped like the
 * input): vjp $\bar{y}\, w_i$ ($\bar{y}$ expanded back over the reduced axes) and jvp $\sum_i \dot{x}_i w_i$. The
 * weight is written with primitives and never reads concrete values, so the rules work under every transform, `vmap`
 * included. Its arguments are the input, the output and the reduction's parameters.
 */
type Weight = (x: Value, y: Value, p: ReduceParams) => Value

/**
 * Define a reduction primitive (registered as `foundation/tensor/<name>`) and return it as a `Reduction`, with the
 * shared shape and batching rules and, given a weight, its reverse and forward rules.
 *
 * @param name The reduction's name, the last part of its primitive id and the name in error messages.
 * @param forward The value on a raw input: given the input, the axes to reduce and the extra arguments.
 * @param weight The derivative as a weight shaped like the input, or `null` when `rules` supplies the derivative (a
 *   linear reduction's `transpose`).
 * @param summary One line for the primitive's documentation.
 * @param extra Extra trailing arguments used in the primitive's generated test cases.
 * @param rules More of the primitive's specification (a transpose, a dtype rule), with `keepDType`: whether the
 *   output keeps the input's dtype (default false, float64).
 * @returns The reduction, called as `(x, axis, keepDims, ...extra)`.
 */
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
 * Sum of elements, by pairwise summation above 128 terms per group (error $O(\varepsilon \log n)$, as NumPy). Complex
 * values sum their real and imaginary parts; a complex sum over every axis is a rank-0 complex tensor. Integers and
 * booleans sum to float64. Linear: its transpose broadcasts the cotangent back over the summed axes.
 *
 * @param x The values to sum: a number, a tensor or a traced value.
 * @param axis The axis or axes to sum over (negative counts from the end); omitted or `null`, all of them.
 * @param keepDims Whether summed axes stay with length $1$ (default false).
 * @returns The sums: a number when every axis is summed without `keepDims` (for real input), else a tensor.
 *
 * @example All elements, by row and by column
 * const x = tensor([[1, 2, 3], [4, 5, 6]])
 * print('total =', sum(x))
 * print('row sums =', sum(x, 1))
 * print('column sums, kept =', sum(x, 0, true))
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

/**
 * Arithmetic mean of elements: `sum` divided by the number of elements in each group, so differentiable and complex
 * values are accepted.
 *
 * @param x The values to average: a number, a tensor or a traced value.
 * @param axis The axis or axes to average over; omitted or `null`, all of them.
 * @param keepDims Whether averaged axes stay with length $1$ (default false).
 * @returns The means: a number when every axis is averaged without `keepDims`, else a tensor.
 *
 * @example The mean of each column
 * const x = tensor([[1, 2], [3, 6]])
 * print('mean =', mean(x))
 * print('column means =', mean(x, 0))
 */
export const mean: Reduction = ((x: Value, axis?: Axes | null, keepDims = false) =>
  div(sum(x, axis, keepDims), groupSize(x, axis))) as Reduction

/**
 * Product of elements. Its derivative $\partial y / \partial x_i$ is the product of the other elements, computed
 * without dividing by zero: $y / x_i$ where $x_i \ne 0$; at a zero, the product of the non-zero elements if it is the
 * group's only zero, else $0$. The weight is itself differentiable: first and second derivatives are exact for any
 * number of zeros; third and higher derivatives are exact except in groups with three or more zeros, where they are
 * taken as $0$. Complex input throws `DTypeError`.
 *
 * @param x The values to multiply: a number, a tensor or a traced value.
 * @param axis The axis or axes to multiply over; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @returns The products, in float64: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example Products, and the gradient at a zero
 * print('prod =', prod(tensor([[1, 2], [3, 4]]), 1))
 * print('gradient at [2, 0, 5] =', grad((x) => prod(x))(tensor([2, 0, 5])))
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

/**
 * Define `max` or `min`: the extreme element of each group, NaN when the group holds a NaN, in the input's dtype. An
 * empty group throws `AifnError`. The derivative goes to the extreme elements, split equally among ties.
 *
 * @param name The reduction's name (`'max'` or `'min'`), for its primitive id and error messages.
 * @param better Whether its first argument beats its second (`>` for the maximum, `<` for the minimum).
 * @param summary One line for the primitive's documentation.
 * @returns The reduction.
 */
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

/**
 * Largest element (NaN if any element is NaN). Ties share the derivative equally. Keeps the input's dtype; an empty
 * group throws `AifnError` and complex input `DTypeError`.
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes to reduce; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @returns The largest values: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example The largest element, and of each row
 * const x = tensor([[1, 5, 2], [7, 0, 3]])
 * print('max =', max(x))
 * print('row maxima =', max(x, 1))
 *
 * @example A tie shares the gradient
 * print('gradient =', grad((x) => max(x))(tensor([3, 1, 3])))
 */
export const max: Reduction = extreme('max', (a, b) => a > b, 'The largest element.')

/**
 * Smallest element (NaN if any element is NaN). Ties share the derivative equally. Keeps the input's dtype; an empty
 * group throws `AifnError` and complex input `DTypeError`.
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes to reduce; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @returns The smallest values: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example The smallest element of each column
 * const x = tensor([[1, 5, 2], [7, 0, 3]])
 * print('min =', min(x))
 * print('column minima =', min(x, 0))
 */
export const min: Reduction = extreme('min', (a, b) => a < b, 'The smallest element.')

/**
 * $\log \sum_i \exp x_i$, computed as $m + \log \sum_i \exp(x_i - m)$ with $m$ the maximum so that it neither
 * overflows nor underflows ($-\infty$ when every element is $-\infty$). Its derivative is the softmax of $\xvec$; a
 * group of $-\infty$ alone has derivative $0$, and one holding $+\infty$ splits it equally among its $+\infty$
 * entries. Complex input throws `DTypeError`.
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes to reduce; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @returns The log-sum-exps, in float64: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example Finite where the direct formula overflows
 * const x = tensor([1000, 1000])
 * print('logsumexp =', logsumexp(x))
 * print('log(sum(exp(x))) =', Math.log(sum(exp(x))))
 *
 * @example Its gradient is the softmax
 * print('gradient =', grad((x) => logsumexp(x))(tensor([0, Math.log(3)])))
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

/**
 * The fused two-pass variance of one group: the mean first, then the centred sum of squares (Welford-free).
 *
 * @param ddof The delta degrees of freedom: the centred sum of squares is divided by the group size less `ddof`.
 * @returns The group reducer.
 */
function groupVariance(ddof: number): GroupReducer {
  return (v, start, width) => {
    const m = pairwiseSum(v, start, width) / width
    let s = 0
    for (let i = start, end = start + width; i < end; i++) s += (v[i] - m) * (v[i] - m)
    return s / (width - ddof)
  }
}

/**
 * Variance $\sum_i (x_i - \bar{x})^2 / (n - d)$, with $n$ the number of elements in each group and $d$ = `ddof`.
 * `ddof` = 0 (default) is the population variance and 1 the unbiased sample variance. A composition (design K §3.4,
 * T9): traced inputs go through `mean` and `square`, so the derivative is theirs and differentiable to any order; raw
 * inputs take a fused two-pass kernel with the same value. Complex input throws `DTypeError`.
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes over which the variance is taken; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @param ddof The delta degrees of freedom $d$ subtracted from $n$ in the denominator (default 0).
 * @returns The variances: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example Population and sample variance
 * const x = tensor([2, 4, 4, 4, 5, 5, 7, 9])
 * print('population =', variance(x))
 * print('sample =', variance(x, null, false, 1))
 */
export const variance: Reduction<[ddof?: number]> = ((x: Value, axis?: Axes | null, keepDims = false, ddof = 0) => {
  if (!isTraced(x)) return reduceRaw(x as Raw, { axis, keepDims }, groupVariance(ddof), false, 'variance')
  if (avalOf(x).dtype === 'complex128')
    throw new DTypeError('variance', 'variance: not defined for complex values (got complex128)', ['complex128'])
  const n = groupSize(x, axis)
  const centred = sub(x, mean(x, axis, true))
  return div(sum(square(centred), axis, keepDims), n - ddof)
}) as Reduction<[ddof?: number]>

/**
 * Standard deviation, the square root of `variance` (same `ddof`).
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes over which it is taken; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @param ddof The delta degrees of freedom, as `variance` takes it (default 0).
 * @returns The standard deviations: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example The spread of each row
 * const x = tensor([[2, 4, 4, 4, 5, 5, 7, 9], [1, 1, 1, 1, 1, 1, 1, 1]])
 * print('std of each row =', std(x, 1))
 */
export const std: Reduction<[ddof?: number]> = ((x: Value, axis?: Axes | null, keepDims = false, ddof = 0) =>
  sqrt(variance(x, axis, keepDims, ddof))) as Reduction<[ddof?: number]>

/**
 * $\sqrt{\sum_i x_i^2}$ of one group without overflow or underflow: $s \sqrt{\sum_i (x_i / s)^2}$ with $s$ the
 * largest $\lvert x_i \rvert$ (LAPACK's `dnrm2` scaling, Blue 1978). NaN when the group holds a NaN, and $s$ itself
 * when $s$ is $0$ or $\infty$.
 *
 * @param v The array holding the group's elements; not modified.
 * @param start The position in `v` of the group's first element.
 * @param width The number of elements in the group.
 * @returns The Euclidean norm of the group.
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
 * The Euclidean norm $\sqrt{\sum_i x_i^2}$ along axes: the stable kernel of `norm` (no overflow for large entries, no
 * underflow for small ones). Its derivative is $\xvec / \lVert \xvec \rVert$, taken as $0$ where
 * $\lVert \xvec \rVert = 0$ (the minimum-norm subgradient), so a zero vector has a zero gradient rather than NaN.
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
 * Vector $p$-norm of the elements along `axis` (of all elements when omitted, which for a matrix is the Frobenius
 * norm): `ord` = 2 (default; the scaled kernel above, so no overflow and a zero gradient at $\zeros$), 1, `Infinity`
 * (largest $\lvert x_i \rvert$), `-Infinity` (smallest $\lvert x_i \rvert$), 0 (count of non-zeros) or any
 * $p > 0$, $(\sum_i \lvert x_i \rvert^p)^{1/p}$. At $\xvec = \zeros$ the gradient is $\zeros$ for every order
 * $p > 1$. Complex values are replaced by their moduli first. Any other order throws `AifnError`.
 *
 * @param x The values: a number, a tensor or a traced value.
 * @param axis The axis or axes along which the norm is taken; omitted or `null`, all of them.
 * @param keepDims Whether reduced axes stay with length $1$ (default false).
 * @param ord The order $p$ of the norm (default 2).
 * @returns The norms: a number when every axis is reduced without `keepDims`, else a tensor.
 *
 * @example The 2-, 1- and max-norms of a vector
 * const v = tensor([3, -4])
 * print('2-norm =', norm(v))
 * print('1-norm =', norm(v, null, false, 1))
 * print('max-norm =', norm(v, null, false, Infinity))
 *
 * @example The norm of each row
 * print('row norms =', norm(tensor([[3, 4], [5, 12]]), 1))
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

/**
 * The parameters of `argmax` and `argmin`: `axis`, the one axis searched (undefined for a flat index over every
 * element), and `keepDims`, whether it stays with length $1$.
 */
type ArgParams = { axis: number | undefined; keepDims: boolean }

/**
 * An index reduction: a number without `axis`, else an int32 tensor; traced inputs give traced results. Called as
 * `(x, axis, keepDims)`.
 */
export interface ArgReduction {
  (x: number | Tensor): number
  (x: Tensor, axis: number, keepDims?: boolean): Tensor
  (x: Traced, axis?: number, keepDims?: boolean): Traced
  (x: Value, axis?: number, keepDims?: boolean): Value
}

/**
 * Index of the extreme element in each group, first occurrence winning; a NaN counts as the extreme. A primitive with
 * zero derivative (piecewise constant), so it can be batched and appear inside differentiated code. A number gives
 * index 0; complex input throws `DTypeError` and an empty group `AifnError`.
 *
 * @param name The reduction's name (`'argmax'` or `'argmin'`), for its primitive id and error messages.
 * @param better Whether its first argument beats its second (`>` for the maximum, `<` for the minimum).
 * @returns The index reduction.
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
 * `axis`. The first occurrence wins ties, and a NaN counts as the largest. Piecewise constant: its derivative is zero,
 * and inside `vmap` it is batched. Complex input throws `DTypeError`.
 *
 * @param x The values: a number (index 0), a tensor or a traced value.
 * @param axis The one axis to search along (negative counts from the end); omitted, every element, in row-major order.
 * @param keepDims Whether the searched axis stays with length $1$ (default false); ignored without `axis`.
 * @returns The index as a number without `axis`, else an int32 tensor of indices along `axis`.
 *
 * @example A flat index, and one per row
 * const x = tensor([[1, 5, 2], [7, 0, 7]])
 * print('flat index =', argmax(x))
 * print('per row =', argmax(x, 1))
 */
export const argmax: ArgReduction = argExtreme('argmax', (a, b) => a > b)

/**
 * Index of the smallest element; see `argmax`. The first occurrence wins ties, and a NaN counts as the smallest.
 *
 * @param x The values: a number (index 0), a tensor or a traced value.
 * @param axis The one axis to search along (negative counts from the end); omitted, every element, in row-major order.
 * @param keepDims Whether the searched axis stays with length $1$ (default false); ignored without `axis`.
 * @returns The index as a number without `axis`, else an int32 tensor of indices along `axis`.
 *
 * @example The smallest entry of each column
 * const x = tensor([[1, 5, 2], [7, 0, 3]])
 * print('flat index =', argmin(x))
 * print('per column =', argmin(x, 0))
 */
export const argmin: ArgReduction = argExtreme('argmin', (a, b) => a < b)

// ── Cumulative sum ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The parameters of `cumsum`: the `axis` summed along, and `reverse`, whether the sums run from its end. */
type CumsumParams = { axis: number; reverse: boolean }

/**
 * Running sums along one axis (from the end with `reverse`), float64 (complex128 for complex input).
 *
 * @param x The tensor to sum, of rank at least 1; not modified.
 * @param axis The axis to sum along (negative counts from the end).
 * @param reverse Whether each sum runs from the end of the axis to the position, rather than from its start.
 * @returns A new contiguous tensor shaped like `x`.
 */
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
 * Running sums along `axis` (default the last): $y_k = \sum_{j \le k} x_j$, or $\sum_{j \ge k} x_j$ with
 * `reverse`. float64, or complex128 for complex input. Linear; its adjoint is the reverse cumulative sum. A number
 * or a rank-0 tensor throws `ShapeError`.
 *
 * @param x The values: a tensor (rank at least 1) or a traced value.
 * @param axis The axis to sum along (negative counts from the end; default the last).
 * @param reverse Whether the sums run from the end of the axis (default false).
 * @returns The running sums, shaped like `x`.
 *
 * @example Running totals, forwards and backwards
 * const x = tensor([1, 2, 3, 4])
 * print('cumsum =', cumsum(x))
 * print('reverse =', cumsum(x, -1, true))
 *
 * @example Down the columns of a matrix
 * print('down the columns =', cumsum(tensor([[1, 2], [3, 4]]), 0))
 */
export function cumsum<X extends Value>(x: X, axis = -1, reverse = false): TensorResult<X> {
  return cumsumOp([x], { axis, reverse }) as TensorResult<X>
}
