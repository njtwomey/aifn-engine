import { ShapeError } from 'aifn-compute/foundation/errors'
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  dense,
  float64Data,
  fromData,
  gather,
  matmul,
  readonlyData,
  scatterAdd,
  slice,
  sum,
  take,
  tensor,
  toFlat,
  toRows,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { checkGradient } from './check-gradient'

describe('dense kernels', () => {
  it('convert vector and matrix arguments, reading strides', () => {
    expect(Array.from(dense.toF64([1, 2, 3], 'test'))).toEqual([1, 2, 3])
    expect(Array.from(dense.toF64(tensor([1, 2, 3]), 'test'))).toEqual([1, 2, 3])
    expect(() => dense.toF64(tensor([[1, 2]]), 'test')).toThrow(/expected a vector/)
    const t = transpose(tensor([1, 2, 3, 4, 5, 6], [2, 3]))
    const m = dense.toMatrixF64(t, 'test')
    expect([m.m, m.n]).toEqual([3, 2])
    expect(Array.from(m.data)).toEqual([1, 4, 2, 5, 3, 6])
    expect(
      Array.from(
        dense.toMatrixF64(
          [
            [1, 2],
            [3, 4],
          ],
          'test',
          2,
          2,
        ).data,
      ),
    ).toEqual([1, 2, 3, 4])
    expect(() => dense.toMatrixF64([[1, 2], [3]], 'test')).toThrow(/ragged/)
    expect(() => dense.toMatrixF64([[1, 2]], 'test', 2, 2)).toThrow(/2×2/)
  })

  it('shares storage only for contiguous float64 tensors', () => {
    const a = tensor([1, 2, 3, 4], [2, 2])
    expect(dense.data(a)).toBe(a.data)
    expect(Array.from(dense.data(transpose(a)))).toEqual([1, 3, 2, 4])
  })

  it('computes products, norms and updates', () => {
    const A = Float64Array.of(1, 2, 3, 4, 5, 6) // 2×3
    expect(dense.dot([1, 2, 3], [4, 5, 6])).toBe(32)
    expect(Array.from(dense.matVec(A, [1, 0, -1], 2, 3))).toEqual([-2, -2])
    expect(Array.from(dense.matTVec(A, [1, -1], 2, 3))).toEqual([-3, -3, -3])
    expect(Array.from(dense.matMul(A, Float64Array.of(1, 0, 0, 1, 1, 1), 2, 3, 2))).toEqual([4, 5, 10, 11])
    expect(Array.from(dense.gram(A, 2, 3))).toEqual([17, 22, 27, 22, 29, 36, 27, 36, 45])
    expect(Array.from(dense.axpy(2, [1, 1], [3, 4]))).toEqual([5, 6])
    expect(Array.from(dense.identity(2))).toEqual([1, 0, 0, 1])
    expect(dense.norm([3e300, 4e300])).toBeCloseTo(5e300, -290)
    expect(dense.norm1(Float64Array.of(1, -2, 3, 4), 2)).toBe(6)
    expect(dense.allFinite([1, NaN])).toBe(false)
  })
})

describe('gather, scatterAdd and take', () => {
  const x = tensor([10, 11, 12, 13, 14, 15], [3, 2])

  it('gather reads flat indices and scatterAdd is its adjoint', () => {
    const g = gather(x, Int32Array.of(5, 0, 0), [3]) as Tensor
    expect(toFlat(g)).toEqual([15, 10, 10])
    const s = scatterAdd(tensor([1, 2, 3]), Int32Array.of(5, 0, 0), [3, 2]) as Tensor
    expect(toRows(s)).toEqual([
      [5, 0],
      [0, 0],
      [0, 1],
    ])
    expect(() => gather(x, Int32Array.of(6), [1])).toThrow(ShapeError)
  })

  it('take reads rows along the first axis, keeping the index shape', () => {
    expect(toRows(take(x, [2, 0]) as Tensor)).toEqual([
      [14, 15],
      [10, 11],
    ])
    expect((take(x, tensor([[1], [2]])) as Tensor).shape).toEqual([2, 1, 2])
    expect(() => take(x, [3])).toThrow(ShapeError)
  })

  it('differentiates through gather, scatterAdd and take (repeats add up)', () => {
    checkGradient((v) => take(v, [2, 0, 2]), [x])
    checkGradient((v) => gather(v, Int32Array.of(1, 1, 4), [3]), [x])
    checkGradient((v) => scatterAdd(v, Int32Array.of(0, 3, 0), [2, 2]), [tensor([1, 2, 3])])
  })
})

describe('raw access and small-matrix kernels', () => {
  it('readonlyData is a zero-copy view of contiguous tensors, null for strided ones', () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(readonlyData(t)).toBe(t.data)
    const row = slice(t, [1, 2]) as Tensor
    const view = readonlyData(row)!
    expect(Array.from(view)).toEqual([4, 5, 6])
    expect(view.buffer).toBe(t.data.buffer)
    expect(readonlyData(transpose(t) as Tensor)).toBeNull()
    expect(Array.from(dense.data(transpose(t) as Tensor))).toEqual([1, 4, 2, 5, 3, 6])
    expect(float64Data(row)).toEqual(Float64Array.from([4, 5, 6]))
  })
  it('dense transpose, sandwich and symmetrise', () => {
    const a = Float64Array.from([1, 2, 3, 4, 5, 6]) // 2×3
    expect(Array.from(dense.transpose(a, 2, 3))).toEqual([1, 4, 2, 5, 3, 6])
    const b = Float64Array.from([2, 0, 1, 0, 1, 0, 1, 0, 3]) // 3×3
    const ref = toFlat(
      matmul(
        matmul(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          fromData(b, [3, 3]),
        ),
        transpose(fromData(a, [2, 3])),
      ),
    )
    expect(Array.from(dense.sandwich(a, b, 2, 3))).toEqual(ref)
    expect(Array.from(dense.symmetrise(Float64Array.from([1, 2, 4, 3]), 2))).toEqual([1, 3, 3, 3])
  })
  it('matmul fast path: same values as the primitive, NaN from 0·∞, strided operands through the general path', () => {
    const a = tensor([
      [0, 1],
      [2, 3],
    ])
    const b = tensor([
      [Infinity, 1],
      [1, 1],
    ])
    expect(toFlat(matmul(a, b))[0]).toBeNaN()
    expect(toFlat(matmul(a, tensor([1, -1])))).toEqual([-1, -1])
    expect(toFlat(matmul(transpose(a), a))).toEqual(
      toFlat(
        matmul(
          tensor([
            [0, 2],
            [1, 3],
          ]),
          a,
        ),
      ),
    )
    // Traced operands still differentiate: d/dx Σ (x A) = row sums of A.
    expect(toFlat(grad((x: Value) => sum(matmul(x, a)))(tensor([[1, 1]])) as Tensor)).toEqual([1, 5])
  })
})
