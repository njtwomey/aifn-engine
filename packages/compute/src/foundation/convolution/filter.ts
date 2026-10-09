/**
 * The linear constant-coefficient difference equation $\sum_k a_k y_{t-k} = \sum_k b_k x_{t-k}$ as one primitive,
 * `linearFilter` (`scipy.signal.lfilter`). A recursive (IIR) filter is not a convolution: its output feeds back.
 *
 * The derivative rules follow Forgione and Piga (2021), "dynoNet: a neural network architecture for learning dynamical
 * systems", Int. J. Adapt. Control Signal Process. 35(4). Write $A$ and $B$ for the polynomials with coefficients $a$
 * and $b$, $\bar{y}$ for the cotangent of the output and $^*$ for the complex conjugate (the $\reals^2$ convention of
 * design K §8.1). With $v$ the anticausal filtering of $\bar{y}$ by $1/A^*$ (reverse, filter, reverse), the cotangent
 * $\bar{x}$ is the anticausal filtering of $\bar{y}$ by $B^* / A^*$, $\bar{b}_k = \sum_t v_t\, x^*_{t-k}$ and
 * $\bar{a}_k = -\sum_t v_t\, y^*_{t-k}$. The jvp filters $\dot{x}$ and the forcing
 * $\sum_k \dot{b}_k x_{t-k} - \sum_k \dot{a}_k y_{t-k}$ through the same recursion. Every rule is written with
 * `linearFilter` itself, so derivatives of every order exist.
 */

import {
  astype,
  avalOf,
  batchByLoop,
  concat,
  conj,
  definePrimitive,
  fromData,
  isTensor,
  isTraced,
  mul,
  neg,
  add,
  projectReal,
  reshape,
  shapeOfValue,
  slice,
  sum,
  tensor,
  zeros,
  type Op,
  type Raw,
  type SliceSpec,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Parameters of the `linearFilter` primitive: `axis`, the time axis of the signal (non-negative). */
type FilterParams = { axis: number }

/**
 * A raw input as a tensor, a number becoming a vector of one value.
 *
 * @param v The raw input of the impl.
 * @returns `v` as a tensor.
 */
const asVector = (v: Raw): Tensor => (typeof v === 'number' ? fromData(Float64Array.of(v), [1]) : v)

/**
 * The recursion along `axis` of $x$, on raw tensors, in direct form with the coefficients normalised by $a_0$:
 * $y_t = \sum_k (b_k / a_0)\, x_{t-k} - \sum_{k \ge 1} (a_k / a_0)\, y_{t-k}$, starting at rest. Complex when any
 * input is. Throws `DomainError` when $a_0 = 0$.
 *
 * @param bRaw The feedforward coefficients $b$, a vector (a number is one coefficient).
 * @param aRaw The feedback coefficients $a$, a vector with $a_0 \neq 0$.
 * @param xRaw The signals, of any shape; every line along `axis` is filtered independently.
 * @param axis The time axis of `xRaw`, non-negative.
 * @returns The outputs $y$, with the shape of `xRaw`; complex128 when any input is.
 */
function filterRaw(bRaw: Raw, aRaw: Raw, xRaw: Raw, axis: number): Tensor {
  const b = asVector(bRaw)
  const a = asVector(aRaw)
  const x = asVector(xRaw)
  const complex = b.dtype === 'complex128' || a.dtype === 'complex128' || x.dtype === 'complex128'
  const dtype = complex ? 'complex128' : 'float64'
  const B = astype(b, dtype).data as Float64Array
  const A = astype(a, dtype).data as Float64Array
  const X = astype(x, dtype).data as Float64Array
  const nb = b.shape[0]
  const na = a.shape[0]
  const shape = x.shape
  const n = shape[axis]
  let inner = 1
  for (let k = axis + 1; k < shape.length; k++) inner *= shape[k]
  const outer = n === 0 ? 0 : X.length / (complex ? 2 : 1) / (n * inner)
  const Y = new Float64Array(X.length)
  if (!complex) {
    const a0 = A[0]
    if (a0 === 0) throw new DomainError('linearFilter', 'linearFilter: a[0] must be nonzero')
    const bn = B.map((v) => v / a0)
    const an = A.map((v) => v / a0)
    for (let o = 0; o < outer; o++)
      for (let i = 0; i < inner; i++) {
        const base = o * n * inner + i
        for (let t = 0; t < n; t++) {
          let acc = 0
          const kb = Math.min(t, nb - 1)
          for (let k = 0; k <= kb; k++) acc += bn[k] * X[base + (t - k) * inner]
          const ka = Math.min(t, na - 1)
          for (let k = 1; k <= ka; k++) acc -= an[k] * Y[base + (t - k) * inner]
          Y[base + t * inner] = acc
        }
      }
    return fromData(Y, [...shape])
  }
  // Complex: divide the coefficients by a₀ once (Smith-free: a₀ ≠ 0 is checked), then the same loops on pairs.
  const ar = A[0]
  const ai = A[1]
  const d = ar * ar + ai * ai
  if (d === 0) throw new DomainError('linearFilter', 'linearFilter: a[0] must be nonzero')
  const normalise = (C: Float64Array) => {
    const out = new Float64Array(C.length)
    for (let k = 0; k < C.length; k += 2) {
      out[k] = (C[k] * ar + C[k + 1] * ai) / d
      out[k + 1] = (C[k + 1] * ar - C[k] * ai) / d
    }
    return out
  }
  const bn = normalise(B)
  const an = normalise(A)
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      const base = o * n * inner + i
      for (let t = 0; t < n; t++) {
        let re = 0
        let im = 0
        const kb = Math.min(t, nb - 1)
        for (let k = 0; k <= kb; k++) {
          const j = 2 * (base + (t - k) * inner)
          re += bn[2 * k] * X[j] - bn[2 * k + 1] * X[j + 1]
          im += bn[2 * k] * X[j + 1] + bn[2 * k + 1] * X[j]
        }
        const ka = Math.min(t, na - 1)
        for (let k = 1; k <= ka; k++) {
          const j = 2 * (base + (t - k) * inner)
          re -= an[2 * k] * Y[j] - an[2 * k + 1] * Y[j + 1]
          im -= an[2 * k] * Y[j + 1] + an[2 * k + 1] * Y[j]
        }
        const j = 2 * (base + t * inner)
        Y[j] = re
        Y[j + 1] = im
      }
    }
  return fromData(Y, [...shape], 'complex128')
}

/**
 * A slice spec selecting a range along one axis and everything along the others.
 *
 * @param rank The rank of the value to slice.
 * @param axis The axis the range applies to.
 * @param range The `[start, stop, step]` spec for that axis.
 * @returns One spec per axis: `range` at `axis`, null (all) elsewhere.
 */
function along(rank: number, axis: number, range: SliceSpec): SliceSpec[] {
  return Array.from({ length: rank }, (_, k) => (k === axis ? range : null))
}

/**
 * A value reversed along one axis (time reversal, for the anticausal filtering of the derivative rules).
 *
 * @param x The value.
 * @param axis The axis to reverse.
 * @returns `x` with its order along `axis` reversed (a slice, so differentiable).
 */
function reverse(x: Value, axis: number): Value {
  return slice(x, ...along(shapeOfValue(x).length, axis, [null, null, -1]))
}

/**
 * The lagged inner products $c_k = \sum_t u^*_{t-k}\, v_t$ for $k = 0, \dots, K - 1$, summed over every line along
 * `axis`: the cotangent of a coefficient vector. Lags at or beyond the signal's length give 0.
 *
 * @param u The signal that is lagged and conjugated ($x$ or $y$ in the derivative rules).
 * @param v The signal it is multiplied with, of the same shape (the filtered cotangent).
 * @param K The number of lags, the length of the coefficient vector.
 * @param axis The time axis of `u` and `v`.
 * @returns The vector $c$ of length $K$, complex when `u` or `v` is.
 */
function laggedProducts(u: Value, v: Value, K: number, axis: number): Value {
  const shape = shapeOfValue(u)
  const rank = shape.length
  const n = shape[axis]
  const all = Array.from({ length: rank }, (_, k) => k)
  const complex = avalOf(u).dtype === 'complex128' || avalOf(v).dtype === 'complex128'
  const terms: Value[] = []
  for (let k = 0; k < K; k++) {
    if (k >= n) {
      terms.push(zeros([1], complex ? 'complex128' : 'float64'))
      continue
    }
    const head = slice(u, ...along(rank, axis, [0, n - k]))
    const tail = slice(v, ...along(rank, axis, [k, n]))
    terms.push(reshape(sum(mul(conj(head), tail), all, true), [1]))
  }
  return concat(terms, 0)
}

/**
 * The difference-equation primitive, with inputs $(b, a, x)$. Its derivative rules are those of the file comment: the
 * vjp filters the time-reversed cotangent by $1/A^*$ and by $B^* / A^*$, and the jvp filters
 * $\dot{x}$, $\dot{b}$ against $x$ and $-\dot{a}$ against $y$ through the same recursion. Batched along the signal's
 * other axes when only $x$ is batched, and by a loop otherwise.
 */
const linearFilterOp: Op<FilterParams> = definePrimitive<FilterParams>({
  id: 'foundation/convolution/linearFilter',
  arity: 3,
  impl: ([b, a, x], { axis }) => filterRaw(b, a, x, axis),
  vjp: (ct, [b, a, x], y, { axis }, needed) => {
    const nb = shapeOfValue(b)[0]
    const na = shapeOfValue(a)[0]
    const ca = conj(a)
    // v = A⁻ᴴ ḡ: the cotangent filtered by 1/conj(A) backwards in time.
    const needV = needed[0] || needed[1]
    const v = needV ? reverse(linearFilterOp([tensor([1]), ca, reverse(ct, axis)], { axis }), axis) : null
    const bBar = needed[0] ? projectReal(laggedProducts(x, v!, nb, axis), avalOf(b).dtype) : null
    const aBar = needed[1] ? projectReal(neg(laggedProducts(y, v!, na, axis)), avalOf(a).dtype) : null
    const xBar = needed[2]
      ? projectReal(reverse(linearFilterOp([conj(b), ca, reverse(ct, axis)], { axis }), axis), avalOf(x).dtype)
      : null
    return [bBar, aBar, xBar]
  },
  jvp: ([db, da, dx], [b, a, x], y, params) => {
    let out: Value | null = null
    const plus = (term: Value) => (out = out === null ? term : add(out, term))
    if (dx !== null) plus(linearFilterOp([b, a, dx], params))
    if (db !== null) plus(linearFilterOp([db, a, x], params))
    if (da !== null) plus(neg(linearFilterOp([da, a, y], params)))
    return out
  },
  shape: ([b, a, x]) => ({
    shape: x.number ? [1] : [...x.shape],
    dtype: [b, a, x].some((v) => v.dtype === 'complex128') ? 'complex128' : 'float64',
    number: false,
  }),
  batch: (values, axes, params, size) => {
    const [bAxis, aAxis, xAxis] = axes
    // Coefficients shared across the batch: the batch axis of x is one more line of the recursion.
    if (bAxis === null && aAxis === null && xAxis !== null) {
      const axis = params.axis + (xAxis <= params.axis ? 1 : 0)
      return [linearFilterOp(values, { axis }), xAxis]
    }
    return batchByLoop(linearFilterOp, values, axes, params, size)
  },
  doc: {
    note: 'difference-equations',
    summary: 'The output of the difference equation with feedforward coefficients b and feedback coefficients a.',
    formula: '\\sum_{k} a_k y_{t-k} = \\sum_{k} b_k x_{t-k}',
  },
  test: {
    secondOrder: true,
    complex: true,
    cases: (draw) => [
      { inputs: [draw([3]), tensor([1, -0.5, 0.2]), draw([7])], params: { axis: 0 } },
      { inputs: [draw([2]), tensor([1.5, 0.4]), draw([2, 5])], params: { axis: 1 } },
      { inputs: [draw([2]), tensor([1, 0.3]), draw([4, 3])], params: { axis: 0 } },
    ],
  },
})

/** Options for `linearFilter`. */
export type LinearFilterOptions = {
  /** The time axis of `x` (default $-1$, the last); negative values count from the end. */
  axis?: number
  /**
   * Initial conditions of the transposed direct form II state, as `scipy.signal.lfilter`'s `zi`: `x`'s shape with the
   * time axis of length $\max(\lvert a \rvert, \lvert b \rvert) - 1$, for the coefficients normalised by $a_0$.
   * Omitted: the filter starts at rest.
   */
  zi?: Value
}

/**
 * A coefficient vector as a rank-1 value: numbers and plain arrays become tensors, and tensors and traced values pass
 * through. Throws `ShapeError` when it is not rank-1 or is empty.
 *
 * @param c The coefficients: a number, an array of numbers, a tensor or a traced value.
 * @param where The caller's name for error messages.
 * @returns The coefficients as a non-empty rank-1 value.
 */
function coefficients(c: Value | VectorLike, where: string): Value {
  if (typeof c === 'number') return tensor([c])
  const v: Value = isTraced(c) || isTensor(c) ? (c as Value) : tensor(Array.from(c as ArrayLike<number>))
  if (shapeOfValue(v).length !== 1) throw new ShapeError(where, `${where}: coefficients must be a vector`)
  if (shapeOfValue(v)[0] === 0) throw new ShapeError(where, `${where}: coefficients must not be empty`)
  return v
}

/**
 * The output $y$ of the difference equation $\sum_{k \ge 0} a_k y_{t-k} = \sum_{k \ge 0} b_k x_{t-k}$ along `axis`
 * of $x$, every other axis a batch of independent signals (`scipy.signal.lfilter`). With `zi`, the initial state $d$
 * (of the equation normalised by $a_0$) adds the forcing $d_t$ for $t < K$, with
 * $K = \max(\lvert a \rvert, \lvert b \rvert) - 1$, so $y$ is `linearFilter(b, a, x)` plus
 * `linearFilter([a[0]], a, d)`. Real or complex; differentiable in $b$, $a$, $x$ and `zi` (reverse and forward, every
 * order) and batched by `vmap`. Throws `DomainError` when $a_0 = 0$, and
 * `ShapeError` for empty or non-vector coefficients, an $x$ without a time axis, an axis out of range or a `zi` of the
 * wrong shape.
 *
 * @param b The feedforward (numerator) coefficients $b_0, b_1, \dots$: a number, an array of numbers or a vector.
 * @param a The feedback (denominator) coefficients $a_0, a_1, \dots$, in the same forms, with $a_0 \neq 0$. `[1]` makes
 *   it an FIR filter.
 * @param x The signal: any shape with a time axis (a number is a signal of one sample).
 * @param options The time `axis` and the initial state `zi`; see `LinearFilterOptions`.
 * @returns The filtered signal $y$, with the shape of $x$; complex when any input is.
 *
 * @example The impulse response of a leaky integrator
 * // y[t] = 0.9 y[t - 1] + x[t]
 * print('y =', linearFilter([1], [1, -0.9], tensor([1, 0, 0, 0, 0])))
 *
 * @example Each row filtered from its own initial state
 * const x = tensor([[1, 1, 1], [0, 0, 0]])
 * const zi = tensor([[0], [2]])
 * print('y =', linearFilter([1], [1, -0.5], x, { zi }))
 *
 * @example The gradient in the feedback coefficients
 * const x = tensor([1, 0, 0])
 * print('d sum(y) / da =', grad((a) => sum(linearFilter([1], a, x)))(tensor([1, -0.5])))
 */
export function linearFilter(
  b: Value | VectorLike,
  a: Value | VectorLike,
  x: Value,
  options?: LinearFilterOptions,
): Value
export function linearFilter(
  b: Value | VectorLike,
  a: Value | VectorLike,
  x: Value,
  options: LinearFilterOptions = {},
) {
  const bv = coefficients(b, 'linearFilter b')
  const av = coefficients(a, 'linearFilter a')
  const xv = typeof x === 'number' ? tensor([x]) : x
  const rank = shapeOfValue(xv).length
  if (rank === 0) throw new ShapeError('linearFilter', 'linearFilter: x must have a time axis')
  const axis = (options.axis ?? -1) < 0 ? rank + (options.axis ?? -1) : options.axis!
  if (axis < 0 || axis >= rank) throw new ShapeError('linearFilter', `linearFilter: axis ${options.axis} out of range`)
  const y = linearFilterOp([bv, av, xv], { axis })
  if (options.zi === undefined) return y
  const K = Math.max(shapeOfValue(bv)[0], shapeOfValue(av)[0]) - 1
  const zShape = shapeOfValue(options.zi)
  const xShape = shapeOfValue(xv)
  if (zShape.length !== rank || zShape.some((d, k) => d !== (k === axis ? K : xShape[k])))
    throw new ShapeError('linearFilter', `linearFilter: zi must have x's shape with ${K} along the time axis`)
  const n = xShape[axis]
  if (K === 0) return y
  // The forcing d: zi padded (or cut) to the signal's length along the time axis.
  const d =
    n >= K
      ? concat([options.zi, zeros(xShape.map((s, k) => (k === axis ? n - K : s)))], axis)
      : slice(options.zi, ...along(rank, axis, [0, n]))
  return add(y, linearFilterOp([slice(av, [0, 1]), av, d], { axis }))
}
