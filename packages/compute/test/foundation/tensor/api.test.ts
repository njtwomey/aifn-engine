import { describe, expect, it } from 'vitest'
import {
  abs,
  add,
  allclose,
  complex,
  arange,
  argmax,
  astype,
  broadcastShapes,
  broadcastTo,
  concat,
  copy,
  div,
  einsum,
  equal,
  equalTo,
  exp,
  expandDims,
  eye,
  fromData,
  fromRows,
  full,
  get,
  greater,
  isContiguous,
  isTraced,
  item,
  less,
  linspace,
  logsumexp,
  map,
  map2,
  matmul,
  max,
  mean,
  mul,
  neg,
  ones,
  reshape,
  scalar,
  set,
  shapeOf,
  slice,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  toRows,
  transpose,
  where,
  zeros,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { grad, traceGraph } from 'aifn-compute/foundation/autodiff'

describe('constructors', () => {
  it('tensor infers shapes from nested arrays and checks raggedness', () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(t.shape).toEqual([2, 3])
    expect(t.strides).toEqual([3, 1])
    expect(t.dtype).toBe('float64')
    expect(() => tensor([[1, 2], [3]] as number[][])).toThrow(/ragged/)
    expect(tensor([1, 2, 3, 4, 5, 6], [3, 2]).shape).toEqual([3, 2])
    expect(() => tensor([1, 2, 3], [2, 2])).toThrow(/do not fill/)
    expect(tensor(new Float32Array([1, 2]), undefined, 'float32').dtype).toBe('float32')
    expect(tensor(7).shape).toEqual([])
  })
  it('zeros, ones, full, eye, scalar', () => {
    expect(toArray(zeros([2]))).toEqual([0, 0])
    expect(toArray(ones([1, 2], 'int32'))).toEqual([[1, 1]])
    expect(toArray(full([2], 2.5))).toEqual([2.5, 2.5])
    expect(toArray(eye(2, 3, 1))).toEqual([
      [0, 1, 0],
      [0, 0, 1],
    ])
    expect(item(scalar(4))).toBe(4)
  })
  it('arange and linspace as numpy', () => {
    expect(toFlat(arange(4))).toEqual([0, 1, 2, 3])
    expect(toFlat(arange(3, 0, -1))).toEqual([3, 2, 1])
    expect(toFlat(linspace(0, 1, 5))).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(linspace(0, 1, 0).shape).toEqual([0])
    expect(toFlat(linspace(3, 3, 1))).toEqual([3])
  })
  it('fromRows and fromData', () => {
    expect(
      fromRows([
        [1, 2],
        [3, 4],
      ]).shape,
    ).toEqual([2, 2])
    expect(() => fromRows([[1, 2], [3]])).toThrow(/row 1/)
    const data = new Float64Array([1, 2, 3, 4])
    expect(fromData(data, [2, 2]).data).toBe(data)
  })
})

describe('converters', () => {
  const m = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ])
  it('round-trip through plain arrays, also from views', () => {
    expect(toRows(m)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(toRows(transpose(m))).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ])
    expect(toFlat(slice(m, null, [null, null, -1]))).toEqual([3, 2, 1, 6, 5, 4])
    expect(toArray(scalar(2))).toBe(2)
    expect(shapeOf([[[1], [2]]])).toEqual([1, 2, 1])
  })
  it('item needs exactly one element', () => {
    expect(item(reshape(tensor([5]), [1, 1]))).toBe(5)
    expect(() => item(m)).toThrow(/6 elements/)
  })
})

describe('indexing and views', () => {
  const m = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ])
  it('get reads with negative indices and checks bounds', () => {
    expect(get(m, 1, -1)).toBe(6)
    expect(get(transpose(m), 2, 0)).toBe(3)
    expect(() => get(m, 2, 0)).toThrow(/out of range/)
    expect(() => get(m, 0)).toThrow(/1 indices/)
  })
  it('set returns a copy and leaves the original unchanged', () => {
    const n = set(m, [0, -1], 9)
    expect(toArray(n)).toEqual([
      [1, 2, 9],
      [4, 5, 6],
    ])
    expect(get(m, 0, 2)).toBe(3)
    expect(get(set(transpose(m), [2, 1], 0), 2, 1)).toBe(0)
  })
  it('views report contiguity; squeeze and expandDims', () => {
    expect(isContiguous(m)).toBe(true)
    expect(isContiguous(transpose(m))).toBe(false)
    expect(isContiguous(slice(m, 1))).toBe(true)
    expect(expandDims(m, -1).shape).toEqual([2, 3, 1])
    expect(squeeze(expandDims(expandDims(m, 0), 3)).shape).toEqual([2, 3])
    expect(() => squeeze(m, 0)).toThrow(/length 2/)
  })
  it('slice errors and clamping', () => {
    expect(() => slice(m, 5)).toThrow(/out of range/)
    expect(() => slice(m, [0, 2, 0])).toThrow(/step/)
    expect(slice(m, [-10, 10]).shape).toEqual([2, 3])
  })
  it('reshape infers -1 and rejects bad sizes', () => {
    expect(reshape(m, [-1]).shape).toEqual([6])
    expect(() => reshape(m, [4, -1])).toThrow(/cannot reshape/)
    expect(() => reshape(m, [-1, -1])).toThrow(/only one/)
  })
  it('concat and stack check shapes and promote dtypes', () => {
    expect(() => concat([m, tensor([1, 2])])).toThrow(/does not match/)
    expect(() => stack([m, transpose(m)])).toThrow(/differs/)
    expect(concat([ones([2], 'int32'), ones([1], 'float32')]).dtype).toBe('float64')
    expect(stack([ones([2], 'int32'), ones([2], 'int32')]).dtype).toBe('int32')
  })
})

describe('elementwise operations', () => {
  it('numbers in, numbers out; tensors in, tensors out', () => {
    expect(add(2, 3)).toBe(5)
    expect(exp(0)).toBe(1)
    expect(typeof sqrt(4)).toBe('number')
    expect(toFlat(add(tensor([1, 2]), 1))).toEqual([2, 3])
    expect(toFlat(sub(1, tensor([1, 2])))).toEqual([0, -1])
  })
  it('dtypes follow NumPy promotion with weak scalars', () => {
    const i = tensor([1, 2], undefined, 'int32')
    expect(add(i, i).dtype).toBe('int32')
    expect(add(i, 1).dtype).toBe('int32')
    expect(add(i, 0.5).dtype).toBe('float64')
    expect(div(i, i).dtype).toBe('float64')
    expect(exp(i).dtype).toBe('float64')
    expect(neg(i).dtype).toBe('int32')
    expect(add(tensor([1], undefined, 'float32'), 1).dtype).toBe('float32')
    expect(add(tensor([1], undefined, 'float32'), i).dtype).toBe('float64')
    expect(abs(astype(tensor([-1.7]), 'int32')).dtype).toBe('int32')
    expect(toFlat(astype(tensor([-1.7, 2.9]), 'int32'))).toEqual([-1, 2])
  })
  it('broadcast shapes and incompatible shapes', () => {
    expect(broadcastShapes([3, 1, 4], [2, 1], [])).toEqual([3, 2, 4])
    expect(() => broadcastShapes([2, 3], [4])).toThrow(/incompatible/)
    expect(() => add(ones([2, 3]), ones([4]))).toThrow(/incompatible/)
    expect(() => broadcastTo(ones([3]), [2, 4])).toThrow(/cannot broadcast/)
  })
  it('comparisons give bool masks; where selects', () => {
    const x = tensor([-1, 0, 2])
    expect(toFlat(greater(x, 0))).toEqual([0, 0, 1])
    expect(greater(x, 0).dtype).toBe('bool')
    expect(greater(x, 0).data).toBeInstanceOf(Uint8Array)
    expect(less(1, 2)).toBe(1)
    expect(toFlat(equalTo(x, tensor([NaN, 0, 2])))).toEqual([0, 1, 1])
    expect(toFlat(where(greater(x, 0), x, 0))).toEqual([0, 0, 2])
    expect(where(1, 5, 6)).toBe(5)
    // A bool condition promotes nothing: int choices stay int.
    expect(where(greater(x, 0), tensor([1, 2, 3], [3], 'int32'), 0).dtype).toBe('int32')
  })
  it('map and map2 apply arbitrary functions', () => {
    expect(toFlat(map(tensor([1.2, 2.7]), Math.floor))).toEqual([1, 2])
    expect(map(2, (v) => v * v)).toBe(4)
    expect(toFlat(map2(tensor([1, 2]), tensor([[10], [20]]), (a, b) => a + b))).toEqual([11, 12, 21, 22])
  })
  it('non-contiguous operands take the strided path', () => {
    const m = reshape(arange(6), [2, 3])
    expect(toRows(add(transpose(m), transpose(m)))).toEqual([
      [0, 6],
      [2, 8],
      [4, 10],
    ])
  })
})

describe('reductions', () => {
  it('reducing everything gives a number; axes and keepDims give tensors', () => {
    const m = reshape(arange(6), [2, 3])
    expect(sum(m)).toBe(15)
    expect(mean(m)).toBe(2.5)
    expect(toFlat(sum(m, 0))).toEqual([3, 5, 7])
    expect(sum(m, null, true).shape).toEqual([1, 1])
    expect(sum(m, [0, 1]).shape).toEqual([])
    expect(() => sum(m, 2)).toThrow(/out of range/)
    expect(() => sum(m, [0, 0])).toThrow(/repeated/)
  })
  it('max propagates NaN; logsumexp handles infinities without overflow', () => {
    expect(max(tensor([1, NaN, 3]))).toBeNaN()
    expect(logsumexp(tensor([1000, 1000]))).toBeCloseTo(1000 + Math.LN2, 12)
    expect(logsumexp(tensor([-Infinity, -Infinity]))).toBe(-Infinity)
    expect(logsumexp(tensor([-Infinity, 0]))).toBe(0)
  })
  it('argmax takes the first of ties and a NaN', () => {
    expect(argmax(tensor([1, 3, 3]))).toBe(1)
    expect(argmax(tensor([1, NaN, 3]))).toBe(1)
  })
})

describe('products', () => {
  it('matmul checks alignment; einsum checks its spec', () => {
    expect(() => matmul(ones([2, 3]), ones([2, 3]))).toThrow(/do not align/)
    expect(() => einsum('i...,i', ones([2]), ones([2]))).toThrow(/ellipsis/)
    expect(() => einsum('ij,jk->ik', ones([2, 3]), ones([2, 3]))).toThrow(/lengths 3 and 2/)
    expect(() => einsum('ij->k', ones([2, 3]))).toThrow(/in no operand/)
    expect(() => einsum('ij', ones([2, 3]), ones([3]))).toThrow(/names 1 operands/)
    expect(() => einsum('ijk', ones([2, 3]))).toThrow(/rank 2/)
  })
  it('matmul of int32 stays int32', () => {
    expect(matmul(eye(2, 2, 0, 'int32'), eye(2, 2, 0, 'int32')).dtype).toBe('int32')
  })
  it('matmul fast path handles a 64×64 product exactly on integers', () => {
    const a = reshape(arange(64 * 64), [64, 64])
    const c = matmul(a, eye(64))
    expect(equal(c, a)).toBe(true)
  })
})

describe('comparison helpers', () => {
  it('allclose and equal', () => {
    expect(allclose(tensor([1, 2]), tensor([1 + 1e-9, 2]))).toBe(true)
    expect(allclose(tensor([1, NaN]), tensor([1, NaN]))).toBe(false)
    expect(allclose(tensor([1, NaN]), tensor([1, NaN]), { equalNan: true })).toBe(true)
    expect(allclose(tensor([Infinity]), tensor([Infinity]))).toBe(true)
    expect(allclose(tensor([[1, 1]]), 1)).toBe(true)
    expect(equal(tensor([1, 2]), tensor([1, 2], undefined, 'int32'))).toBe(true)
    expect(equal(tensor([1, 2]), tensor([[1, 2]]))).toBe(false)
    expect(equal(copy(transpose(eye(2))), eye(2))).toBe(true)
  })
})

describe('interpreters', () => {
  it('primitives on raw values run directly and record nothing', () => {
    expect(isTraced(mul(tensor([1]), 2))).toBe(false)
    const g = traceGraph((x: Tensor) => sum(mul(x, x)), tensor([1, 2]))
    expect(g.nodes.map((n) => n.op)).toEqual(['input', 'mul', 'sum'])
    expect(g.value).toBe(5)
    expect(toFlat(g.grad as Tensor)).toEqual([2, 4])
  })
  it('nested transforms keep their levels apart (no perturbation confusion)', () => {
    // d/dx [x · d/dy (x + y)] = d/dx [x · 1] = 1 (Siskind and Pearlmutter, 2005).
    const inner = (x: number) => grad((y: number) => add(x, y) as number)(1)
    expect(grad((x: number) => mul(x, inner(x)) as number)(3)).toBe(1)
  })
  it('comparisons and argmax are never recorded', () => {
    const g = traceGraph(
      (x: Tensor) => {
        greater(x, 1)
        argmax(x)
        return sum(x)
      },
      tensor([1, 2]),
    )
    expect(g.nodes.map((n) => n.op)).toEqual(['input', 'sum'])
    const z: Tensor = zeros([1])
    expect(z.shape).toEqual([1])
  })
})

describe('equal across real and complex (review 2026-10-01)', () => {
  it('compares a real tensor with a complex one as complex, imaginary parts included', () => {
    // Before: the real tensor's two values were compared with the first (re, im) pair of the complex one.
    expect(equal(tensor([1, 2]), complex(tensor([1, 3]), tensor([2, 4])))).toBe(false)
    expect(equal(tensor([1, 3]), complex(tensor([1, 3]), tensor([0, 0])))).toBe(true)
    expect(equal(complex(scalar(1), scalar(5)), 1)).toBe(false)
    expect(equal(complex(scalar(1), scalar(0)), 1)).toBe(true)
  })
})
