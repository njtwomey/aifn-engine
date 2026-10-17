/**
 * aifn-compute/nn/functional `deformableConv2d`: zero offsets give the ordinary cross-correlation, offsets reproduce
 * dilation, masks scale the taps, and the gradient in the offsets agrees with finite differences (the site's
 * self-check, on the tensor form).
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { conv } from 'aifn-compute/foundation/convolution'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream, units } from 'aifn-compute/foundation/random'
import { fromData, mul, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { deformableConv2d } from 'aifn-compute/nn/functional'

const C = 2
const H = 6
const W = 7
const O = 3
const k = 3
const K = k * k
const s = stream('deformable-check')
const x = tensor(
  Array.from(units(s, C * H * W), (u) => 2 * u - 1),
  [1, C, H, W],
)
const w = tensor(
  Array.from(units(s, O * C * K), (u) => 2 * u - 1),
  [O, C, k, k],
)
const flat = (v: Value) => toFlat(v as Tensor)
const zero = () => fromData(new Float64Array(2 * K * H * W), [1, 2 * K, H, W])
const maxDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))

describe('deformableConv2d', () => {
  it('with zero offsets is the same-padded cross-correlation', () => {
    const engine = flat(conv(x, w, { padding: 'same', flip: false }))
    expect(maxDiff(flat(deformableConv2d(x, w, zero(), { padding: 1 })), engine)).toBeLessThan(1e-12)
  })

  it('with zero offsets and dilation 2 is the dilated cross-correlation', () => {
    const engine = flat(conv(x, w, { padding: 2, dilation: 2, flip: false }))
    expect(maxDiff(flat(deformableConv2d(x, w, zero(), { padding: 2, dilation: 2 })), engine)).toBeLessThan(1e-12)
  })

  it('reproduces dilation 2 with offsets p_n', () => {
    const offsets = new Float64Array(2 * K * H * W)
    for (let n = 0; n < K; n++)
      for (let q = 0; q < H * W; q++) {
        offsets[2 * n * H * W + q] = Math.floor(n / k) - 1
        offsets[(2 * n + 1) * H * W + q] = (n % k) - 1
      }
    const engine = flat(conv(x, w, { padding: 2, dilation: 2, flip: false }))
    const ours = flat(deformableConv2d(x, w, fromData(offsets, [1, 2 * K, H, W]), { padding: 1 }))
    expect(maxDiff(ours, engine)).toBeLessThan(1e-12)
  })

  it('halves the output with masks of 0.5', () => {
    const engine = flat(conv(x, w, { padding: 'same', flip: false }))
    const mask = fromData(new Float64Array(K * H * W).fill(0.5), [1, K, H, W])
    const ours = flat(deformableConv2d(x, w, zero(), { padding: 1, mask }))
    expect(
      maxDiff(
        ours,
        engine.map((v) => v / 2),
      ),
    ).toBeLessThan(1e-12)
  })

  it('takes one image without a batch axis', () => {
    const offsets = fromData(new Float64Array(2 * K * H * W), [2 * K, H, W])
    const single = deformableConv2d(tensor(flat(x), [C, H, W]), w, offsets, { padding: 1 }) as Tensor
    expect(single.shape).toEqual([O, H, W])
  })

  it('differentiates in the offsets, the input and the kernels', () => {
    const offsets = fromData(
      Float64Array.from(units(stream('offsets'), 2 * K * H * W), (u) => 1.6 * u - 0.8),
      [1, 2 * K, H, W],
    )
    const weights = tensor(
      Array.from(units(stream('weights'), O * H * W), (u) => u),
      [1, O, H, W],
    )
    const f = (o: Value) => sum(mul(deformableConv2d(x, w, o, { padding: 1 }), weights))
    const g = flat(grad(f)(offsets))
    const base = flat(offsets)
    for (const at of [3, 50, 200, 700]) {
      const h = 1e-6
      const bumped = Float64Array.from(base)
      bumped[at] += h
      const fd = ((f(fromData(bumped, [1, 2 * K, H, W])) as number) - (f(offsets) as number)) / h
      expect(g[at]).toBeCloseTo(fd, 4)
    }
    const gx = flat(grad((v: Value) => sum(mul(deformableConv2d(v, w, offsets, { padding: 1 }), weights)))(x))
    expect(gx.length).toBe(C * H * W)
    const gw = flat(grad((v: Value) => sum(mul(deformableConv2d(x, v, offsets, { padding: 1 }), weights)))(w))
    expect(gw.length).toBe(O * C * K)
  })

  it('throws for offsets of the wrong shape', () => {
    expect(() => deformableConv2d(x, w, fromData(new Float64Array(9), [1, 9, 1, 1]))).toThrow(AifnError)
  })
})
