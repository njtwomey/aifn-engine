/**
 * Products: batched matrix multiplication, dot and outer products, and a small `einsum`. Each is a primitive (or a
 * composition of primitives) with its derivative rule: for C = AB the cotangents are Ḡ Bᵀ and Aᵀ Ḡ (Giles, 2008,
 * "Collected matrix derivative results for forward and reverse mode algorithmic differentiation", §2.2).
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { allocate, flatData, fromData, isTensor, promote, readonlyData, showShape, sizeOf, type Tensor } from './core'
import * as dense from './dense'
import { mul } from './elementwise'
import { resultType } from './dtype'
import { complexKernel, matmulKernel, splitComplex } from './kernels'
import { conj } from './complex'
import { batchToFront, definePrimitive, sumLike, type Op, type Raw, type Result2 } from './primitive'
import { sum } from './reduce'
import { batchByLoop, broadcastTo, expandDims, permute, reshape, shapeOfValue, squeeze } from './structure'
import { scatterAdd } from './gather'
import { avalOf, type Value } from './trace'
import { broadcastShapes } from './views'

/** Swap the last two axes. */
function swapLast(x: Value): Value {
  const rank = shapeOfValue(x).length
  const axes = Array.from({ length: rank }, (_, k) => k)
  ;[axes[rank - 2], axes[rank - 1]] = [axes[rank - 1], axes[rank - 2]]
  return permute(x, axes)
}

function tensorInput(x: Raw, where: string): Tensor {
  if (!isTensor(x)) throw new AifnError(where, `${where}: expected a tensor, got a number`)
  return x
}

// A worked example of a multilinear primitive (design K §4.2): C = AB is linear in A and in B separately. The author
// writes the transpose in each argument (Ā = C̄ Bᵀ, B̄ = Aᵀ C̄; Giles, 2008, §2.2), the shape rule and the batching rule;
// the vjp and the jvp (Ȧ B + A Ḃ) are derived. For complex operands the ℝ² adjoint conjugates: Ā = C̄ Bᴴ, B̄ = Aᴴ C̄
// (conj is the identity on real values, so real matrices pay nothing).
const matmulOp: Op<undefined> = definePrimitive<undefined>({
  id: 'foundation/tensor/matmul',
  dtype: 'same',
  arity: 2,
  impl: ([a, b]) => matmulKernel(tensorInput(a, 'matmul'), tensorInput(b, 'matmul')),
  linear: 'multilinear',
  transpose: (ct, [a, b], which) =>
    which === 0
      ? sumLike(matmulOp([ct, swapLast(conj(b))], undefined), a)
      : sumLike(matmulOp([swapLast(conj(a)), ct], undefined), b),
  shape: ([a, b]) => {
    const [m, k] = a.shape.slice(-2)
    const [k2, n] = b.shape.slice(-2)
    if (k !== k2)
      throw new ShapeError('matmul', `matmul: shapes ${showShape(a.shape)} and ${showShape(b.shape)} do not align`)
    return {
      shape: [...broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2)), m, n],
      dtype: promote(a.dtype, b.dtype),
      number: false,
    }
  },
  // Batch axes of matmul broadcast like elementwise ones, so a batch axis moved to the front (and padded to the rank
  // of the larger example) is just one more batch axis.
  batch: (values, axes) => {
    const ranks = values.map((v, i) => avalOf(v).shape.length - (axes[i] === null ? 0 : 1))
    const rank = Math.max(...ranks)
    const moved = values.map((v, i) => {
      const axis = axes[i]
      return axis === null ? v : batchToFront(v, axis, rank - ranks[i])
    })
    return [matmulOp(moved, undefined), 0]
  },
  doc: { note: 'matrix-multiplication', summary: 'Batched matrix product with broadcast batch axes.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [{ inputs: [draw([2, 3]), draw([3, 4])] }, { inputs: [draw([2, 2, 3]), draw([3, 2])] }],
  },
})

/**
 * The fast path of `matmul` for concrete contiguous float64 operands, a matrix times a matrix or a vector: the
 * `dense` kernels on zero-copy views (`readonlyData`), without the views and the primitive dispatch that dominate for
 * the small matrices of filters and other per-step recursions. Concrete tensors are constants to every transform, so
 * the result is the value the primitive gives. Null for any other operands (the general path handles them).
 */
function matmulDirect(a: Tensor, b: Tensor): Tensor | null {
  const rb = b.shape.length
  if (a.dtype !== 'float64' || b.dtype !== 'float64' || a.shape.length !== 2 || rb < 1 || rb > 2) return null
  const [m, k] = a.shape
  if (b.shape[0] !== k) return null
  const ad = readonlyData(a)
  const bd = readonlyData(b)
  if (ad === null || bd === null) return null
  return rb === 1
    ? fromData(dense.matVec(ad, bd, m, k), [m])
    : fromData(dense.matMul(ad, bd, m, k, b.shape[1]), [m, b.shape[1]])
}

/**
 * Matrix product with NumPy's `matmul` rules. Rank-2 operands multiply as matrices ([m, k] × [k, n] → [m, n]). Higher
 * ranks are stacks of matrices whose leading (batch) axes broadcast. A rank-1 left operand is a row vector and a
 * rank-1 right operand a column vector, and the added axis is removed from the result (vector · vector gives a rank-0
 * tensor; use `dot` for a number).
 */
export function matmul<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, Tensor> {
  if (isTensor(a) && isTensor(b)) {
    const direct = matmulDirect(a, b)
    if (direct !== null) return direct as Result2<A, B, Tensor>
  }
  const ra = shapeOfValue(a).length
  const rb = shapeOfValue(b).length
  if (ra === 0 || rb === 0) throw new ShapeError('matmul', 'matmul: operands must have rank ≥ 1 (use mul for scalars)')
  const left = ra === 1 ? expandDims(a, 0) : a
  const right = rb === 1 ? expandDims(b, 1) : b
  let out = matmulOp([left, right], undefined)
  if (rb === 1) out = squeeze(out, -1)
  if (ra === 1) out = squeeze(out, -2 + (rb === 1 ? 1 : 0))
  return out as Result2<A, B, Tensor>
}

/** The dot product Σᵢ aᵢbᵢ of two vectors of equal length, as a number. */
export function dot<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, number> {
  const sa = shapeOfValue(a)
  const sb = shapeOfValue(b)
  if (sa.length !== 1 || sb.length !== 1 || sa[0] !== sb[0]) {
    throw new ShapeError(
      'dot',
      `dot: expected two vectors of equal length, got ${showShape(sa)} and ${showShape(sb)}`,
      [sa, sb],
    )
  }
  return sum(mul(a, b)) as Result2<A, B, number>
}

/** The outer product abᵀ of two vectors (other ranks are flattened first), shape [len a, len b]. */
export function outer<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, Tensor> {
  return mul(reshape(a, [-1, 1]), reshape(b, [1, -1])) as Result2<A, B, Tensor>
}

// ── einsum ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A parsed einsum specification: the labels of each operand and of the output. */
type EinsumSpec = { inputs: string[]; output: string }

/** Parse "ij,jk->ik" (explicit) or "ij,jk" (implicit: labels used once, in alphabetical order). */
function parseEinsum(spec: string, count: number): EinsumSpec {
  const clean = spec.replace(/\s+/g, '')
  if (clean.includes('.')) throw new AifnError('einsum', `einsum: ellipsis ("...") is not supported in "${spec}"`)
  const [lhs, rhs] = clean.split('->')
  if (clean.split('->').length > 2) throw new AifnError('einsum', `einsum: more than one "->" in "${spec}"`)
  const inputs = lhs.split(',')
  if (inputs.length !== count)
    throw new ShapeError('einsum', `einsum: "${spec}" names ${inputs.length} operands, got ${count}`)
  for (const labels of [...inputs, rhs ?? '']) {
    if (!/^[a-zA-Z]*$/.test(labels))
      throw new AifnError('einsum', `einsum: labels must be letters, got "${labels}" in "${spec}"`)
  }
  let output = rhs
  if (output === undefined) {
    const counts = new Map<string, number>()
    for (const c of inputs.join('')) counts.set(c, (counts.get(c) ?? 0) + 1)
    output = [...counts.keys()]
      .filter((c) => counts.get(c) === 1)
      .sort()
      .join('')
  }
  if (new Set(output).size !== output.length)
    throw new AifnError('einsum', `einsum: output "${output}" repeats a label`)
  for (const c of output) {
    if (!inputs.some((labels) => labels.includes(c)))
      throw new ShapeError('einsum', `einsum: output label "${c}" is in no operand`)
  }
  return { inputs, output }
}

/**
 * Forward einsum on any dtypes: einsum is linear in each operand, so a complex operand splits into its real and
 * imaginary parts, einsum(…, x + iy, …) = einsum(…, x, …) + i·einsum(…, y, …), down to real kernels.
 */
function einsumAny(spec: EinsumSpec, ts: Tensor[]): Tensor {
  const j = ts.findIndex((t) => t.dtype === 'complex128')
  if (j < 0) return einsumKernel(spec, ts)
  const [re, im] = splitComplex(ts[j])
  const a = einsumAny(
    spec,
    ts.map((t, k) => (k === j ? re : t)),
  )
  const b = einsumAny(
    spec,
    ts.map((t, k) => (k === j ? im! : t)),
  )
  return complexKernel([a, b], (o, z) => {
    o[0] = z[0] - z[3]
    o[1] = z[1] + z[2]
  })
}

/** Forward einsum: a loop over every combination of labels, output labels outermost (real dtypes). */
function einsumKernel(spec: EinsumSpec, ts: Tensor[]): Tensor {
  const sizes = new Map<string, number>()
  spec.inputs.forEach((labels, i) => {
    const t = ts[i]
    if (labels.length !== t.shape.length) {
      throw new ShapeError('einsum', `einsum: operand ${i} has rank ${t.shape.length} but labels "${labels}"`)
    }
    ;[...labels].forEach((c, k) => {
      const d = t.shape[k]
      const known = sizes.get(c)
      if (known !== undefined && known !== d)
        throw new ShapeError('einsum', `einsum: label "${c}" has lengths ${known} and ${d}`)
      sizes.set(c, d)
    })
  })
  const summed = [...sizes.keys()].filter((c) => !spec.output.includes(c))
  const order = [...spec.output, ...summed]
  const shape = order.map((c) => sizes.get(c)!)
  // Each operand's stride per label; a label repeated within an operand (a diagonal) adds its strides.
  const strides = ts.map((t, i) =>
    order.map((c) => [...spec.inputs[i]].reduce((s, l, k) => (l === c ? s + t.strides[k] : s), 0)),
  )
  const outShape = [...spec.output].map((c) => sizes.get(c)!)
  const out = allocate(ts.map((t) => t.dtype).reduce(promote), sizeOf(outShape))
  const inner = sizeOf(summed.map((c) => sizes.get(c)!))
  const index = new Array<number>(order.length).fill(0)
  const offsets = ts.map((t) => t.offset)
  const total = sizeOf(shape)
  // Walk all label combinations in row-major order of `order`; each run of `inner` steps accumulates one output.
  for (let k = 0; k < total; k++) {
    let term = 1
    for (let i = 0; i < ts.length; i++) term *= ts[i].data[offsets[i]]
    out[Math.floor(k / inner)] += term
    for (let a = order.length - 1; a >= 0; a--) {
      index[a]++
      for (let i = 0; i < ts.length; i++) offsets[i] += strides[i][a]
      if (index[a] < shape[a]) break
      for (let i = 0; i < ts.length; i++) offsets[i] -= strides[i][a] * shape[a]
      index[a] = 0
    }
  }
  return fromData(out, outShape)
}

/** The length of each label, checked to agree wherever the label occurs. */
function labelSizes(spec: EinsumSpec, shapes: readonly (readonly number[])[]): Map<string, number> {
  const sizes = new Map<string, number>()
  spec.inputs.forEach((labels, i) => {
    const shape = shapes[i]
    if (labels.length !== shape.length) {
      throw new ShapeError('einsum', `einsum: operand ${i} has rank ${shape.length} but labels "${labels}"`)
    }
    ;[...labels].forEach((c, k) => {
      const known = sizes.get(c)
      if (known !== undefined && known !== shape[k])
        throw new ShapeError('einsum', `einsum: label "${c}" has lengths ${known} and ${shape[k]}`)
      sizes.set(c, shape[k])
    })
  })
  return sizes
}

/**
 * The flat row-major positions, in an array of shape `shape` labelled `labels` (with repeats), of the elements whose
 * repeated labels agree, listed in row-major order of the distinct labels `unique`: where a diagonal sits.
 */
function diagonalPositions(labels: string, unique: string, shape: readonly number[]): Int32Array {
  const sizes = [...unique].map((c) => shape[labels.indexOf(c)])
  const strides = new Array<number>(unique.length).fill(0)
  let stride = 1
  for (let k = labels.length - 1; k >= 0; k--) {
    strides[unique.indexOf(labels[k])] += stride
    stride *= shape[k]
  }
  const out = new Int32Array(sizeOf(sizes))
  const index = new Array<number>(unique.length).fill(0)
  for (let n = 0; n < out.length; n++) {
    let off = 0
    for (let a = 0; a < unique.length; a++) off += index[a] * strides[a]
    out[n] = off
    for (let a = unique.length - 1; a >= 0; a--) {
      if (++index[a] < sizes[a]) break
      index[a] = 0
    }
  }
  return out
}

// einsum is multilinear: linear in each operand with the others fixed. The transpose in operand i contracts the output
// cotangent with the other operands over the labels operand i shares with them; labels only operand i has were summed
// away, so the cotangent is constant along them (broadcast back); a label repeated in operand i (a diagonal) puts the
// cotangent on that diagonal and zeros elsewhere (a scatterAdd). The jvp Σᵢ einsum(…, ẋᵢ, …) is derived.
const einsumOp: Op<EinsumSpec> = definePrimitive<EinsumSpec>({
  id: 'foundation/tensor/einsum',
  dtype: 'same',
  impl: (xs, spec) =>
    einsumAny(
      spec,
      xs.map((x) => tensorInput(x, 'einsum')),
    ),
  linear: 'multilinear',
  transpose: (ct, xs, i, spec) => {
    const labels = spec.inputs[i]
    const unique = [...new Set(labels)].join('')
    const others = spec.inputs.filter((_, j) => j !== i)
    const reached = [...unique].filter((c) => spec.output.includes(c) || others.some((o) => o.includes(c))).join('')
    const g = avalOf(ct).number ? reshape(ct, []) : ct
    // The ℝ² adjoint contracts with the conjugates of the other operands.
    const operands = [g, ...xs.filter((_, j) => j !== i).map((x) => conj(x))]
    let out: Value = einsumOp(operands, { inputs: [spec.output, ...others], output: reached })
    const shape = shapeOfValue(xs[i])
    const sizes = [...unique].map((c) => shape[labels.indexOf(c)])
    if (reached.length !== unique.length) {
      const partial = [...unique].map((c, k) => (reached.includes(c) ? sizes[k] : 1))
      out = broadcastTo(reshape(out, partial), sizes)
    }
    if (unique.length === labels.length) return out
    return scatterAdd(out, diagonalPositions(labels, unique, shape), shape)
  },
  shape: (avals, spec) => {
    const sizes = labelSizes(
      spec,
      avals.map((a) => a.shape),
    )
    return {
      shape: [...spec.output].map((c) => sizes.get(c)!),
      dtype: avals.map((a) => a.dtype).reduce(promote),
      number: false,
    }
  },
  // A fresh label for the batch axis, on every batched operand and on the output.
  batch: (xs, axes, spec, size) => {
    const used = new Set([...spec.inputs.join(''), ...spec.output])
    const free = [...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'].find((c) => !used.has(c))
    if (free === undefined) return batchByLoop(einsumOp, xs, axes, spec, size)
    const moved = xs.map((x, i) => {
      const b = axes[i]
      return b === null ? x : batchToFront(x, b)
    })
    const inputs = spec.inputs.map((labels, i) => (axes[i] === null ? labels : free + labels))
    return [einsumOp(moved, { inputs, output: free + spec.output }), 0]
  },
  doc: { note: 'matrix-multiplication', summary: 'Einstein summation over letter-labelled axes.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 3]), draw([3, 4])], params: { inputs: ['ij', 'jk'], output: 'ik' } },
      { inputs: [draw([2, 3])], params: { inputs: ['ij'], output: 'ji' } },
      { inputs: [draw([3]), draw([3])], params: { inputs: ['i', 'i'], output: '' } },
      { inputs: [draw([3, 3])], params: { inputs: ['ii'], output: 'i' } },
    ],
  },
})

/**
 * Einstein summation over letter-labelled axes, e.g. `einsum('ij,jk->ik', a, b)` (matrix product),
 * `einsum('ij->ji', a)`, `einsum('i,i->', a, b)`, `einsum('bij,bjk->bik', a, b)`, `einsum('ii->', a)` (trace) or
 * `einsum('ii->i', a)` (diagonal). Without `->` the output is the labels used once, in alphabetical order. Labels
 * must have equal lengths wherever they occur (no broadcasting) and ellipses are not supported. The loop runs over
 * every label combination, so it suits small contractions; use `matmul` for large products.
 */
export function einsum(spec: string, ...operands: Tensor[]): Tensor
export function einsum(spec: string, ...operands: Value[]): Value
export function einsum(spec: string, ...operands: Value[]): Value {
  const parsed = parseEinsum(spec, operands.length)
  return einsumOp(operands, parsed)
}

// ── Linear combinations ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Σᵢ cᵢ xᵢ over inputs of one shape with fixed real coefficients, in one pass: the update of an explicit Runge–Kutta
 * stage (x + h Σ aᵢⱼ kⱼ) or of an optimiser step, which as `add`s and `mul`s would cost two primitives per term. Linear,
 * so its derivatives are derived from the transpose, cᵢ·ct in input i.
 */
const linearCombinationOp: Op<readonly number[]> = definePrimitive<readonly number[]>({
  id: 'foundation/tensor/linearCombination',
  arity: 'variadic',
  // Accumulated in float64; the result is float32 only when every tensor input is (the `float` rule).
  dtype: 'float',
  impl: (xs, c) => {
    if (xs.length === 0 || xs.length !== c.length)
      throw new ShapeError('linearCombination', `linearCombination: ${xs.length} inputs for ${c.length} coefficients`)
    if (xs.every((x) => typeof x === 'number')) return (xs as number[]).reduce((s, x, i) => s + c[i] * x, 0)
    const shape = isTensor(xs[0]) ? xs[0].shape : (xs.find(isTensor) as Tensor).shape
    const n = sizeOf(shape)
    const out = new Float64Array(n)
    xs.forEach((x, i) => {
      const ci = c[i]
      if (ci === 0) return
      if (typeof x === 'number') {
        for (let k = 0; k < n; k++) out[k] += ci * x
        return
      }
      if (x.dtype === 'complex128')
        throw new AifnError('linearCombination', 'linearCombination: complex inputs are not supported')
      if (sizeOf(x.shape) !== n || x.shape.length !== shape.length || x.shape.some((d, k) => d !== shape[k]))
        throw new ShapeError(
          'linearCombination',
          `linearCombination: shape ${showShape(x.shape)} differs from ${showShape(shape)}`,
          [x.shape, shape],
        )
      const v = readonlyData(x) ?? flatData(x)
      for (let k = 0; k < n; k++) out[k] += ci * (v[k] as number)
    })
    const tensors = xs.filter(isTensor)
    const single = tensors.length > 0 && tensors.every((x) => x.dtype === 'float32')
    return single ? fromData(Float32Array.from(out), [...shape]) : fromData(out, [...shape])
  },
  linear: 'linear',
  transpose: (ct, xs, which, c) => {
    const term = mul(c[which], ct)
    return avalOf(xs[which]).number ? sum(term) : term
  },
  shape: (avals, c) => {
    const tensorAval = avals.find((a) => !a.number)
    if (avals.length !== c.length)
      throw new ShapeError(
        'linearCombination',
        `linearCombination: ${avals.length} inputs for ${c.length} coefficients`,
      )
    const tensorDTypes = avals.filter((a) => !a.number).map((a) => a.dtype)
    return tensorAval
      ? {
          shape: [...tensorAval.shape],
          dtype: tensorDTypes.every((d) => d === 'float32')
            ? 'float32'
            : resultType('float', tensorDTypes.reduce(promote)),
          number: false,
        }
      : { shape: [], dtype: 'float64', number: true }
  },
  // Every input is brought to [size, ...example shape]: batched ones move their batch axis first, and both they and
  // unbatched ones are broadcast up from a number example (which the forward rule broadcasts against tensors).
  batch: (xs, axes, c, size) => {
    const examples = xs.map((x, i) => shapeOfValue(x).filter((_, k) => axes[i] === null || k !== axes[i]))
    const example = examples.find((s, i) => s.length > 0 || (axes[i] === null && !avalOf(xs[i]).number)) ?? []
    const shape = [size, ...example]
    const moved = xs.map((x, i) => {
      const b = axes[i]
      const front = b !== null ? batchToFront(x, b) : reshape(x, [1, ...examples[i]])
      const pad = example.length - examples[i].length
      const lead = b !== null ? size : 1
      return broadcastTo(pad > 0 ? reshape(front, [lead, ...new Array<number>(pad).fill(1)]) : front, shape)
    })
    return [linearCombinationOp(moved, c), 0]
  },
  doc: { summary: 'The linear combination Σ cᵢ xᵢ of same-shaped inputs with fixed coefficients.' },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([3]), draw([3]), draw([3])], params: [1, -0.5, 2] },
      { inputs: [draw([2, 2]), draw([2, 2])], params: [0.25, 3] },
    ],
  },
})

/**
 * Σᵢ cᵢ xᵢ for same-shaped real inputs (numbers, tensors or traced values) and fixed coefficients c, in one primitive.
 *
 * @example linearCombination([x, k1, k2], [1, h / 2, h / 2]) // x + h(k1 + k2)/2
 */
export function linearCombination(xs: readonly Value[], coefficients: readonly number[]): Value {
  return linearCombinationOp(xs, coefficients)
}
