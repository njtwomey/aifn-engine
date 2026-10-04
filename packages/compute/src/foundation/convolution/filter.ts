/**
 * The linear constant-coefficient difference equation Σₖ aₖ y[t−k] = Σₖ bₖ x[t−k] as one primitive, `linearFilter`
 * (scipy.signal's `lfilter`). A recursive (IIR) filter is not a convolution: its output feeds back. The derivative
 * rules follow Forgione & Piga (2021), "dynoNet: a neural network architecture for learning dynamical systems", Int. J.
 * Adapt. Control Signal Process. 35(4): with v the anticausal filtering of the cotangent ḡ by 1/Ā (reverse, filter,
 * reverse), x̄ = the anticausal filtering of ḡ by B̄/Ā, b̄ₖ = Σₜ vₜ x̄[t−k] and āₖ = −Σₜ vₜ ȳ[t−k] (bars on signals are
 * complex conjugates, the ℝ² convention of design K §8.1). The jvp filters ẋ and the forcing Σ ḃₖ x[t−k] − Σ ȧₖ y[t−k]
 * through the same recursion. Every rule is written with `linearFilter` itself, so derivatives of every order exist.
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

type FilterParams = { axis: number }

const asVector = (v: Raw): Tensor => (typeof v === 'number' ? fromData(Float64Array.of(v), [1]) : v)

/** The recursion along `axis` of x, on raw tensors: direct form, coefficients normalised by a₀. */
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

/** A slice spec selecting `range` along `axis` of a rank-`rank` value (null elsewhere). */
function along(rank: number, axis: number, range: SliceSpec): SliceSpec[] {
  return Array.from({ length: rank }, (_, k) => (k === axis ? range : null))
}

/** x reversed along `axis`. */
function reverse(x: Value, axis: number): Value {
  return slice(x, ...along(shapeOfValue(x).length, axis, [null, null, -1]))
}

/**
 * The lagged inner products cₖ = Σₜ conj(u[t−k]) v[t], k = 0 … K−1, summed over every line along `axis` (a [K]
 * vector): the cotangent of a coefficient vector.
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
  /** The time axis of x (default −1, the last). */
  axis?: number
  /**
   * Initial conditions of the transposed direct form II state, as `scipy.signal.lfilter`'s `zi`: x's shape with the
   * time axis of length max(|a|, |b|) − 1, for the coefficients normalised by a₀. Omitted: the filter starts at rest.
   */
  zi?: Value
}

/** A coefficient vector as a rank-1 value (numbers and plain arrays become tensors). */
function coefficients(c: Value | VectorLike, where: string): Value {
  if (typeof c === 'number') return tensor([c])
  const v: Value = isTraced(c) || isTensor(c) ? (c as Value) : tensor(Array.from(c as ArrayLike<number>))
  if (shapeOfValue(v).length !== 1) throw new ShapeError(where, `${where}: coefficients must be a vector`)
  if (shapeOfValue(v)[0] === 0) throw new ShapeError(where, `${where}: coefficients must not be empty`)
  return v
}

/**
 * The output y of the difference equation Σₖ₌₀ aₖ y[t−k] = Σₖ₌₀ bₖ x[t−k] along `axis` of x, every other axis a batch
 * of independent signals (`scipy.signal.lfilter`). a₀ must be nonzero. With `zi`, the initial state d (of the
 * normalised equation) adds the forcing d[t], t < K − 1, so y = lfilter(b, a, x) + lfilter([a₀], a, d). Real or
 * complex; differentiable in b, a, x and zi (reverse and forward, every order) and batched by `vmap`.
 *
 * @example linearFilter([1], [1, -0.9], x) // the leaky integrator y[t] = 0.9 y[t−1] + x[t]
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
