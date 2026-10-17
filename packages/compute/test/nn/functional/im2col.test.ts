/**
 * aifn-compute/nn/functional `im2col`: the column matrix times the flattened kernels equals `conv2d` for several
 * strides and paddings, and the convolutional-layer worked example (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream, units } from 'aifn-compute/foundation/random'
import { matmul, reshape, sum, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { conv2d, im2col } from 'aifn-compute/nn/functional'

const flat = (v: Value) => toFlat(v as Tensor)

describe('im2col', () => {
  it.each([
    [1, 0],
    [1, 1],
    [2, 1],
  ])('times the kernels equals conv2d (stride %i, padding %i)', (stride, padding) => {
    const s = stream('fast-convolution-check')
    const [C, H, W, O, k] = [3, 6, 5, 2, 3]
    const x = tensor(
      Array.from(units(s, C * H * W), (u) => 2 * u - 1),
      [C, H, W],
    )
    const w = tensor(
      Array.from(units(s, O * C * k * k), (u) => 2 * u - 1),
      [O, C, k, k],
    )
    const product = flat(matmul(reshape(w, [O, C * k * k]), im2col(x, k, { stride, padding })))
    const engine = flat(conv2d(x, w, { stride, padding }))
    expect(product.length).toBe(engine.length)
    product.forEach((v, i) => expect(Math.abs(v - engine[i])).toBeLessThan(1e-12))
  })

  it('gives the worked example: a 4 x 4 image, a 3 x 3 vertical-edge kernel, a 9 x 4 matrix', () => {
    const x = tensor([
      [
        [1, 2, 0, 1],
        [0, 1, 3, 1],
        [2, 1, 0, 0],
        [1, 0, 1, 2],
      ],
    ])
    const cols = im2col(x, 3) as Tensor
    expect(cols.shape).toEqual([9, 4])
    expect(toRows(cols)[0]).toEqual([1, 2, 0, 1])
    expect(toRows(cols)[4]).toEqual([1, 3, 1, 0])
    expect(flat(matmul(tensor([[1, 0, -1, 1, 0, -1, 1, 0, -1]]), cols))).toEqual([0, 2, -1, -1])
  })

  it('keeps a batch axis and differentiates in x', () => {
    const x = tensor([
      [
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      ],
    ])
    expect((im2col(x, 2) as Tensor).shape).toEqual([1, 4, 4])
    // Each pixel appears in as many windows as cover it: the corners once, the centre four times.
    expect(flat(grad((v: Value) => sum(im2col(v, 2)))(x))).toEqual([1, 2, 1, 2, 4, 2, 1, 2, 1])
  })

  it('throws for an input of rank 2', () => {
    expect(() =>
      im2col(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        1,
      ),
    ).toThrow(AifnError)
  })
})
