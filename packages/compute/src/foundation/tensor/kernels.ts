/**
 * Raw computational kernels on plain tensors: elementwise maps with broadcasting, reductions and batched matrix
 * products. Primitives call these for their forward values. Contiguous float64 inputs take tight loops over `data`.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  allocate,
  complexPartView,
  flatData,
  forEachOffset,
  forEachOffset2,
  fromData,
  isContiguous,
  normaliseAxes,
  promote,
  scalarDType,
  showShape,
  size,
  sizeOf,
  type Axes,
  type DType,
  type Tensor,
} from './core'
import { broadcastShapes, broadcastView, permuteView } from './views'

/**
 * Apply `f` to every element, writing a new tensor of dtype `dtype`. A contiguous input takes a tight loop; any other
 * is walked through its strides.
 *
 * @param x The tensor, read and not modified.
 * @param f The scalar function.
 * @param dtype The dtype of the result.
 * @returns A new contiguous tensor of the shape of `x`.
 */
export function unaryKernel(x: Tensor, f: (v: number) => number, dtype: DType): Tensor {
  const n = size(x)
  const out = allocate(dtype, n)
  const src = x.data
  if (isContiguous(x)) {
    const o = x.offset
    for (let k = 0; k < n; k++) out[k] = f(src[o + k])
  } else {
    forEachOffset(x.shape, x.strides, x.offset, (off, k) => {
      out[k] = f(src[off])
    })
  }
  return fromData(out, x.shape)
}

/**
 * The arithmetic operations with dedicated loops. A loop that calls a closure per element is fast only while one
 * closure reaches it; shared by every elementwise primitive, the call becomes megamorphic (about 9 ns per element), so
 * the four operations every figure uses most are written out (design K §3.5, T5).
 */
export type Arithmetic = 'add' | 'sub' | 'mul' | 'div'

/**
 * `out[at + j] = f(a[ia + j * sa], b[ib + j * sb])` for $0 \le j < n$: one run of a binary kernel, with stride 0 for a
 * repeated operand. `op` selects a dedicated loop for the arithmetic operations.
 *
 * @param op The arithmetic operation with a dedicated loop, or undefined to call `f` per element.
 * @param f The scalar function, used only when `op` is undefined.
 * @param out The output array: entries `at` to `at + n - 1` are written.
 * @param at The first index of `out` written.
 * @param a The first operand's storage.
 * @param ia The index in `a` of the run's first element.
 * @param sa The step between the run's elements in `a` (0 repeats one element).
 * @param b The second operand's storage.
 * @param ib The index in `b` of the run's first element.
 * @param sb The step between the run's elements in `b` (0 repeats one element).
 * @param n The length $n$ of the run.
 */
function binaryRun(
  op: Arithmetic | undefined,
  f: (x: number, y: number) => number,
  out: { [k: number]: number },
  at: number,
  a: ArrayLike<number>,
  ia: number,
  sa: number,
  b: ArrayLike<number>,
  ib: number,
  sb: number,
  n: number,
): void {
  switch (op) {
    case 'add':
      for (let j = 0; j < n; j++) out[at + j] = a[ia + j * sa] + b[ib + j * sb]
      return
    case 'sub':
      for (let j = 0; j < n; j++) out[at + j] = a[ia + j * sa] - b[ib + j * sb]
      return
    case 'mul':
      for (let j = 0; j < n; j++) out[at + j] = a[ia + j * sa] * b[ib + j * sb]
      return
    case 'div':
      for (let j = 0; j < n; j++) out[at + j] = a[ia + j * sa] / b[ib + j * sb]
      return
    default:
      for (let j = 0; j < n; j++) out[at + j] = f(a[ia + j * sa], b[ib + j * sb])
  }
}

/**
 * `f(x, c)` for every element `x` of a tensor and a constant `c` (the path of `binaryKernel` for a non-contiguous
 * tensor and a number).
 *
 * @param a The tensor, read through its strides.
 * @param c The constant second argument.
 * @param f The scalar function.
 * @param dtype The dtype of the result.
 * @returns A new contiguous tensor of the shape of `a`.
 */
function withRightScalar(a: Tensor, c: number, f: (x: number, y: number) => number, dtype: DType): Tensor {
  const n = size(a)
  const out = allocate(dtype, n)
  const src = a.data
  if (isContiguous(a)) {
    const o = a.offset
    for (let k = 0; k < n; k++) out[k] = f(src[o + k], c)
  } else forEachOffset(a.shape, a.strides, a.offset, (off, k) => (out[k] = f(src[off], c)))
  return fromData(out, a.shape)
}

/**
 * `f(c, y)` for a constant `c` and every element `y` of a tensor (the path of `binaryKernel` for a number and a
 * non-contiguous tensor).
 *
 * @param c The constant first argument.
 * @param b The tensor, read through its strides.
 * @param f The scalar function.
 * @param dtype The dtype of the result.
 * @returns A new contiguous tensor of the shape of `b`.
 */
function withLeftScalar(c: number, b: Tensor, f: (x: number, y: number) => number, dtype: DType): Tensor {
  const n = size(b)
  const out = allocate(dtype, n)
  const src = b.data
  if (isContiguous(b)) {
    const o = b.offset
    for (let k = 0; k < n; k++) out[k] = f(c, src[o + k])
  } else forEachOffset(b.shape, b.strides, b.offset, (off, k) => (out[k] = f(c, src[off])))
  return fromData(out, b.shape)
}

/**
 * Whether two shapes are equal.
 *
 * @param a The first shape.
 * @param b The second shape.
 * @returns True when they have the same rank and the same length on every axis.
 */
const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((d, k) => d === b[k])

/**
 * The length $m$ of the trailing block that `small` repeats when broadcast to `shape`, or 0 when it is not such a row:
 * `small`'s shape without its leading 1s must equal the last axes of `shape` (`[n, d] + [d]`, `[b, n, d] + [1, n, d]`).
 *
 * @param small The shape of the operand that is broadcast.
 * @param shape The broadcast shape.
 * @returns $m$, the number of elements of `small`, when it repeats as a row; otherwise 0.
 */
function rowLength(small: readonly number[], shape: readonly number[]): number {
  let lead = 0
  while (lead < small.length && small[lead] === 1) lead++
  const tail = small.length - lead
  if (tail > shape.length) return 0
  let m = 1
  for (let k = 0; k < tail; k++) {
    if (small[lead + k] !== shape[shape.length - tail + k]) return 0
    m *= small[lead + k]
  }
  return m
}

/**
 * Apply `f` elementwise to two broadcast operands, writing dtype `dtype`. A plain number broadcasts as a scalar.
 * Contiguous operands take tight loops: same shapes, a scalar, and a row repeated over leading axes (`[n, d] + [d]`,
 * design K §3.5); other layouts walk their strides. `op` names an arithmetic operation with a dedicated loop.
 *
 * @param a The first operand: a tensor (read, not modified), or a number broadcast as a scalar.
 * @param b The second operand, likewise.
 * @param f The scalar function of one element of each.
 * @param dtype The dtype of the result: the kernel does not promote (see `binaryDType`).
 * @param op An arithmetic operation with a dedicated loop; left out, `f` is called per element.
 * @returns A new contiguous tensor of the broadcast shape (rank 0 for two numbers).
 */
export function binaryKernel(
  a: Tensor | number,
  b: Tensor | number,
  f: (x: number, y: number) => number,
  dtype: DType,
  op?: Arithmetic,
): Tensor {
  if (typeof a === 'number' && typeof b === 'number') return fromData(allocate(dtype, 1).fill(f(a, b)), [])
  if (typeof b === 'number') {
    if (!isContiguous(a as Tensor)) return withRightScalar(a as Tensor, b, f, dtype)
    const t = a as Tensor
    const out = allocate(dtype, size(t))
    binaryRun(op, f, out, 0, t.data, t.offset, 1, [b], 0, 0, out.length)
    return fromData(out, t.shape)
  }
  if (typeof a === 'number') {
    if (!isContiguous(b)) return withLeftScalar(a, b, f, dtype)
    const out = allocate(dtype, size(b))
    binaryRun(op, f, out, 0, [a], 0, 0, b.data, b.offset, 1, out.length)
    return fromData(out, b.shape)
  }
  const shape = broadcastShapes(a.shape, b.shape)
  const n = sizeOf(shape)
  const out = allocate(dtype, n)
  const da = a.data
  const db = b.data
  const oa = a.offset
  const ob = b.offset
  const ca = isContiguous(a)
  const cb = isContiguous(b)
  const fullA = sameShape(a.shape, shape)
  const fullB = sameShape(b.shape, shape)
  if (ca && cb && fullA && fullB) {
    binaryRun(op, f, out, 0, da, oa, 1, db, ob, 1, n)
    return fromData(out, shape)
  }
  if (ca && cb && fullA && n > 0) {
    const m = rowLength(b.shape, shape)
    if (m > 0) {
      for (let k = 0; k < n; k += m) binaryRun(op, f, out, k, da, oa + k, 1, db, ob, 1, m)
      return fromData(out, shape)
    }
  }
  if (ca && cb && fullB && n > 0) {
    const m = rowLength(a.shape, shape)
    if (m > 0) {
      for (let k = 0; k < n; k += m) binaryRun(op, f, out, k, da, oa, 1, db, ob + k, 1, m)
      return fromData(out, shape)
    }
  }
  const va = broadcastView(a, shape)
  const vb = broadcastView(b, shape)
  forEachOffset2(shape, va.strides, va.offset, vb.strides, vb.offset, (i, j, k) => {
    out[k] = f(da[i], db[j])
  })
  return fromData(out, shape)
}

/**
 * Apply `f` elementwise to three broadcast operands, writing dtype `dtype`. Plain numbers broadcast as scalars; three
 * numbers give a rank-0 tensor. Operands are read in place through their (broadcast) strides, never copied.
 *
 * @param a The first operand: a tensor, or a number broadcast as a scalar.
 * @param b The second operand, likewise.
 * @param c The third operand, likewise.
 * @param f The scalar function of one element of each.
 * @param dtype The dtype of the result.
 * @returns A new contiguous tensor of the broadcast shape.
 */
export function ternaryKernel(
  a: Tensor | number,
  b: Tensor | number,
  c: Tensor | number,
  f: (x: number, y: number, z: number) => number,
  dtype: DType,
): Tensor {
  const operands = [a, b, c]
  const shape = broadcastShapes(...operands.map((v) => (typeof v === 'number' ? [] : v.shape)))
  const n = sizeOf(shape)
  const out = allocate(dtype, n)
  const rank = shape.length
  // Each tensor operand as (data, offset, strides over `shape`); a number has stride 0 into a one-element array.
  const data: ArrayLike<number>[] = []
  const offsets: number[] = []
  const strides: (readonly number[])[] = []
  let dense = true
  for (const v of operands) {
    if (typeof v === 'number') {
      data.push([v])
      offsets.push(0)
      strides.push(new Array<number>(rank).fill(0))
      dense = false
    } else {
      const view = broadcastView(v, shape)
      data.push(v.data)
      offsets.push(view.offset)
      strides.push(view.strides)
      if (!(sameShape(v.shape, shape) && isContiguous(v))) dense = false
    }
  }
  const [d0, d1, d2] = data
  if (dense) {
    const [o0, o1, o2] = offsets
    for (let k = 0; k < n; k++) out[k] = f(d0[o0 + k], d1[o1 + k], d2[o2 + k])
    return fromData(out, shape)
  }
  if (n === 0) return fromData(out, shape)
  if (rank === 0) {
    out[0] = f(d0[offsets[0]], d1[offsets[1]], d2[offsets[2]])
    return fromData(out, shape)
  }
  // An odometer over the leading axes and a tight loop over the last, as `forEachOffset`, with three offsets.
  const last = rank - 1
  const inner = shape[last]
  const [s0, s1, s2] = strides
  const index = new Array<number>(rank).fill(0)
  let [b0, b1, b2] = offsets
  for (let k = 0; k < n;) {
    for (let j = 0, i0 = b0, i1 = b1, i2 = b2; j < inner; j++, k++, i0 += s0[last], i1 += s1[last], i2 += s2[last])
      out[k] = f(d0[i0], d1[i1], d2[i2])
    let axis = last - 1
    while (axis >= 0) {
      index[axis]++
      b0 += s0[axis]
      b1 += s1[axis]
      b2 += s2[axis]
      if (index[axis] < shape[axis]) break
      b0 -= s0[axis] * shape[axis]
      b1 -= s1[axis] * shape[axis]
      b2 -= s2[axis] * shape[axis]
      index[axis] = 0
      axis--
    }
    if (axis < 0) break
  }
  return fromData(out, shape)
}

/**
 * The dtype of a binary operation's result, before any operation-specific rule (e.g. division gives floats): two
 * tensors promote, a number next to a tensor is a weak scalar (`weakType`), and two numbers give float64.
 *
 * @param a The first operand.
 * @param b The second operand.
 * @returns The promoted dtype.
 */
export function binaryDType(a: Tensor | number, b: Tensor | number): DType {
  if (typeof a === 'number' && typeof b === 'number') return 'float64'
  if (typeof a === 'number') return scalarDType(a, (b as Tensor).dtype)
  if (typeof b === 'number') return scalarDType(b, a.dtype)
  return promote(a.dtype, b.dtype)
}

/**
 * Shape bookkeeping for a reduction over `axes` (sorted, normalised).
 *
 * @param shape The shape of the tensor reduced.
 * @param axes The reduced axes, as non-negative indices.
 * @param keepDims Keep each reduced axis with length 1 instead of dropping it.
 * @returns The shape of the result.
 */
export function reducedShape(shape: readonly number[], axes: readonly number[], keepDims: boolean): number[] {
  return keepDims ? shape.map((d, k) => (axes.includes(k) ? 1 : d)) : shape.filter((_, k) => !axes.includes(k))
}

/**
 * The reduction of one output group: `values[start … start + width)` are its elements in row-major order of the reduced
 * axes. Receiving a range rather than a copy keeps a reduction with many small groups (per-row statistics of a tall
 * matrix) free of per-group allocation (design K §3.5).
 */
export type GroupReducer = (values: ArrayLike<number>, start: number, width: number) => number

/**
 * Reduce over `axes`: for every output position, `fn` folds that group's elements (a range of one array, see
 * `GroupReducer`). An input whose reduced axes are already innermost and contiguous is read in place; otherwise its
 * elements are copied once into float64, with the reduced axes moved last. Returns a tensor of dtype `dtype`.
 *
 * @param x The real tensor to reduce.
 * @param axis The axis or axes to reduce (negative counts from the end); null or undefined reduces every axis.
 *   Repeated or out-of-range axes throw `ShapeError`.
 * @param keepDims Keep each reduced axis with length 1 instead of dropping it.
 * @param fn The reduction of one group's elements.
 * @param dtype The dtype of the result.
 * @returns One value of `fn` per group, in the reduced shape.
 */
export function reduceKernel(
  x: Tensor,
  axis: Axes | null | undefined,
  keepDims: boolean,
  fn: GroupReducer,
  dtype: DType = 'float64',
): Tensor {
  const axes = normaliseAxes(axis, x.shape.length, 'reduce')
  const outShape = reducedShape(x.shape, axes, keepDims)
  const kept = x.shape.map((_, k) => k).filter((k) => !axes.includes(k))
  const groups = sizeOf(kept.map((k) => x.shape[k]))
  const width = sizeOf(axes.map((k) => x.shape[k]))
  // Move the kept axes first: in row-major order, each run of `width` elements then belongs to one output.
  const moved = permuteView(x, [...kept, ...axes])
  let values: ArrayLike<number>
  let base = 0
  if (isContiguous(moved)) {
    values = x.data
    base = moved.offset
  } else {
    const copy = new Float64Array(groups * width)
    const src = x.data
    forEachOffset(moved.shape, moved.strides, moved.offset, (off, k) => {
      copy[k] = src[off]
    })
    values = copy
  }
  const out = allocate(dtype, groups)
  for (let g = 0; g < groups; g++) out[g] = fn(values, base + g * width, width)
  return fromData(out, outShape)
}

/** Below this many terms a sum is accumulated left to right; above it, by pairwise halving. */
const PAIRWISE_BLOCK = 128

/**
 * The sum of `values[start … start + width)`, by pairwise summation above 128 terms (rounding error
 * $O(\varepsilon \log n)$ rather than $O(\varepsilon n)$, as NumPy's `sum`; Higham, "Accuracy and Stability of
 * Numerical Algorithms", 2nd ed., §4.2). It is a `GroupReducer`.
 *
 * @param values The array holding the terms.
 * @param start The index of the first term.
 * @param width The number of terms.
 * @returns Their sum.
 */
export function pairwiseSum(values: ArrayLike<number>, start: number, width: number): number {
  if (width <= PAIRWISE_BLOCK) {
    let s = 0
    for (let i = start, end = start + width; i < end; i++) s += values[i]
    return s
  }
  const half = Math.floor(width / 2)
  return pairwiseSum(values, start, half) + pairwiseSum(values, start + half, width - half)
}

/**
 * Sum `x` down to `shape`, the inverse of broadcasting `shape` up to `x.shape`: leading axes are summed away and axes
 * where `shape` has length 1 are summed with the length kept. A shape that does not broadcast to `x.shape` throws
 * `ShapeError`.
 *
 * @param x The tensor to sum; a complex128 one has its real and imaginary parts summed separately.
 * @param shape The target shape.
 * @returns A tensor of shape `shape`: int32 for an int32 `x`, complex128 for a complex one, float64 otherwise.
 */
export function sumToKernel(x: Tensor, shape: readonly number[]): Tensor {
  const lead = x.shape.length - shape.length
  if (lead < 0)
    throw new ShapeError('sumTo', `sumTo: cannot reduce ${showShape(x.shape)} to ${showShape(shape)}`, [x.shape, shape])
  const axes: number[] = []
  for (let k = 0; k < x.shape.length; k++) {
    if (k < lead) axes.push(k)
    else if (shape[k - lead] === 1 && x.shape[k] !== 1) axes.push(k)
    else if (shape[k - lead] !== x.shape[k]) {
      throw new ShapeError('sumTo', `sumTo: cannot reduce ${showShape(x.shape)} to ${showShape(shape)}`, [
        x.shape,
        shape,
      ])
    }
  }
  if (x.dtype === 'complex128') {
    const [re, im] = splitComplex(x)
    return joinComplex(sumToKernel(re, shape), sumToKernel(im!, shape))
  }
  const summed = reduceKernel(x, axes, true, pairwiseSum, x.dtype === 'int32' ? 'int32' : 'float64')
  return fromData(summed.data, shape)
}

/**
 * Batched matrix product for operands of rank $\ge 2$: the last two axes multiply as matrices
 * ($[\dots, m, k] \times [\dots, k, n]$) and the leading (batch) axes broadcast. Complex operands are multiplied
 * through the real products of their parts. A lower rank, or inner dimensions that differ, throws `ShapeError`.
 *
 * @param a The left operand, of shape $[\dots, m, k]$.
 * @param b The right operand, of shape $[\dots, k, n]$.
 * @returns The product, of shape $[\dots, m, n]$ with the broadcast batch axes, and the promoted dtype.
 */
export function matmulKernel(a: Tensor, b: Tensor): Tensor {
  if (a.dtype === 'complex128' || b.dtype === 'complex128') {
    // (Ar + iAi)(Br + iBi) = (ArBr − AiBi) + i(ArBi + AiBr), on the zero-copy float64 views of the parts.
    const [ar, ai] = splitComplex(a)
    const [br, bi] = splitComplex(b)
    const re =
      ai && bi
        ? binaryKernel(matmulKernel(ar, br), matmulKernel(ai, bi), (x, y) => x - y, 'float64', 'sub')
        : matmulKernel(ar, br)
    const terms = [ai ? matmulKernel(ai, br) : null, bi ? matmulKernel(ar, bi) : null].filter((t) => t !== null)
    const im =
      terms.length === 2
        ? binaryKernel(terms[0], terms[1], (x, y) => x + y, 'float64', 'add')
        : terms.length === 1
          ? terms[0]
          : null
    return joinComplex(re, im ?? binaryKernel(re, 0, (x, y) => x * y, 'float64', 'mul'))
  }
  const ra = a.shape.length
  const rb = b.shape.length
  if (ra < 2 || rb < 2) throw new ShapeError('matmul', 'matmul: kernel needs rank ≥ 2 operands')
  const [m, k] = a.shape.slice(-2)
  const [k2, n] = b.shape.slice(-2)
  if (k !== k2)
    throw new ShapeError(
      'matmul',
      `matmul: shapes ${showShape(a.shape)} and ${showShape(b.shape)} do not align (${k} ≠ ${k2})`,
      [a.shape, b.shape],
    )
  const batch = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2))
  const va = broadcastView(a, [...batch, m, k])
  const vb = broadcastView(b, [...batch, k, n])
  const dtype = promote(a.dtype, b.dtype)
  const nb = sizeOf(batch)
  const out = allocate(dtype, nb * m * n)
  // Per-batch offsets of each operand's matrix.
  const offA: number[] = []
  const offB: number[] = []
  forEachOffset2(batch, va.strides.slice(0, -2), va.offset, vb.strides.slice(0, -2), vb.offset, (i, j) => {
    offA.push(i)
    offB.push(j)
  })
  const [sai, sak] = va.strides.slice(-2)
  const [sbk, sbj] = vb.strides.slice(-2)
  const da = a.data
  const db = b.data
  const row = new Float64Array(n)
  for (let p = 0; p < nb; p++) {
    const oa = offA[p]
    const ob = offB[p]
    const base = p * m * n
    for (let i = 0; i < m; i++) {
      // i-k-j order: stream a row of B per element of A's row, which is cache-friendly for row-major B.
      row.fill(0)
      for (let q = 0; q < k; q++) {
        const aiq = da[oa + i * sai + q * sak]
        const bq = ob + q * sbk
        for (let j = 0; j < n; j++) row[j] += aiq * db[bq + j * sbj]
      }
      for (let j = 0; j < n; j++) out[base + i * n + j] = row[j]
    }
  }
  return fromData(out, [...batch, m, n])
}

// ── Complex kernels ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The real and imaginary parts of a tensor as float64 views: for complex128 the zero-copy views of its storage
 * (`complexPartView`), for a real tensor the tensor itself and null (a zero imaginary part).
 *
 * @param z The tensor to split.
 * @returns The pair `[re, im]`, where `im` is null for a real tensor.
 *
 * @example Split a complex vector without copying, and a real one
 * const [re, im] = splitComplex(complex(tensor([1, 2]), tensor([3, 4])))
 * print('re =', re)
 * print('im =', im)
 * print('a real tensor:', splitComplex(tensor([5, 6])))
 */
export function splitComplex(z: Tensor): [Tensor, Tensor | null] {
  if (z.dtype !== 'complex128') return [z, null]
  return [complexPartView(z, 0), complexPartView(z, 1)]
}

/**
 * A complex128 tensor from real and imaginary parts of one shape (any real dtypes, any strides). Copies.
 *
 * @param re The real parts.
 * @param im The imaginary parts, of the shape of `re` (not checked).
 * @returns A new contiguous complex128 tensor of the shape of `re`.
 *
 * @example Join two real vectors, and split them again
 * const z = joinComplex(tensor([1, 2]), tensor([3, 4]))
 * print('z as (re, im) pairs:', z)
 * print('dtype:', z.dtype)
 * print('parts:', splitComplex(z))
 */
export function joinComplex(re: Tensor, im: Tensor): Tensor {
  const n = size(re)
  const out = new Float64Array(2 * n)
  const fr = flatData(re, 'float64')
  const fi = flatData(im, 'float64')
  for (let k = 0; k < n; k++) {
    out[2 * k] = fr[k]
    out[2 * k + 1] = fi[k]
  }
  return fromData(out, re.shape, 'complex128')
}

/**
 * A complex scalar rule: `z` holds the arguments as (re, im) pairs in order (`z[0]`, `z[1]` the first; a real argument
 * has imaginary part 0), and the rule writes the result's real part to `out[0]` and imaginary part to `out[1]`.
 */
export type ComplexRule = (out: Float64Array, z: Float64Array) => void

/**
 * Apply a complex scalar rule elementwise to broadcast operands (tensors of any dtype, or JS numbers, which are real).
 * The result is complex128, or float64 with `real` (the rule then writes only `out[0]`, e.g. $\lvert z \rvert$).
 * Operands are read in place through their broadcast strides, two slots per complex element.
 *
 * @param args The operands, broadcast together: complex128 tensors are read as (re, im) pairs, other tensors and
 *   numbers as real values.
 * @param rule The complex scalar rule, given one (re, im) pair per operand.
 * @param real Whether the result is real: float64, from the real part the rule writes (default complex128).
 * @returns A new contiguous tensor of the broadcast shape.
 *
 * @example Conjugate a complex vector, and take its modulus
 * const z = complex(tensor([3, 5]), tensor([4, 12]))
 * const conjugate = (o, w) => {
 *   o[0] = w[0]
 *   o[1] = -w[1]
 * }
 * print('conj as (re, im) pairs:', complexKernel([z], conjugate))
 * print('modulus:', complexKernel([z], (o, w) => (o[0] = Math.hypot(w[0], w[1])), true))
 */
export function complexKernel(args: readonly (Tensor | number)[], rule: ComplexRule, real = false): Tensor {
  const shape = broadcastShapes(...args.map((v) => (typeof v === 'number' ? [] : v.shape)))
  const n = sizeOf(shape)
  const rank = shape.length
  const m = args.length
  const data: ArrayLike<number>[] = []
  const pos: number[] = []
  const strides: (readonly number[])[] = []
  const pair: boolean[] = []
  for (const v of args) {
    if (typeof v === 'number') {
      data.push([v, 0])
      pos.push(0)
      strides.push(new Array<number>(rank).fill(0))
      pair.push(true)
    } else {
      const b = broadcastView(v, shape)
      data.push(v.data)
      pos.push(b.offset)
      strides.push(b.strides)
      pair.push(v.dtype === 'complex128')
    }
  }
  const out = new Float64Array(real ? n : 2 * n)
  const z = new Float64Array(2 * m)
  const r = new Float64Array(2)
  const index = new Array<number>(rank).fill(0)
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < m; i++) {
      const d = data[i]
      const o = pos[i]
      if (pair[i]) {
        z[2 * i] = d[2 * o]
        z[2 * i + 1] = d[2 * o + 1]
      } else {
        z[2 * i] = d[o]
        z[2 * i + 1] = 0
      }
    }
    rule(r, z)
    if (real) out[k] = r[0]
    else {
      out[2 * k] = r[0]
      out[2 * k + 1] = r[1]
    }
    // Advance the odometer over the output's axes, moving every operand's position with it.
    for (let a = rank - 1; a >= 0; a--) {
      index[a]++
      for (let i = 0; i < m; i++) pos[i] += strides[i][a]
      if (index[a] < shape[a]) break
      for (let i = 0; i < m; i++) pos[i] -= strides[i][a] * shape[a]
      index[a] = 0
    }
  }
  return fromData(out, shape, real ? 'float64' : 'complex128')
}
