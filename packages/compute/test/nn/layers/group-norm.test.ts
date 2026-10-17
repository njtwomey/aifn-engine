/**
 * aifn-compute/nn/layers group and instance normalisation: the [2, 4, 2, 2] worked example of the
 * normalisation-in-convolutional-networks note, the limits G = 1 (layer norm over C, H, W) and G = C (instance norm),
 * the per-channel affine, the layers, and gradients (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { mul, reshape, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { batchNorm, GroupNorm, groupNorm, InstanceNorm, instanceNorm, layerNorm } from 'aifn-compute/nn/layers'

const X = tensor(
  [
    [1, 3, 5, 7],
    [2, 2, 4, 4],
    [0, 4, 0, 4],
    [6, 6, 6, 10],
    [9, 11, 13, 15],
    [0, 0, 2, 2],
    [3, 3, 5, 5],
    [2, 4, 6, 8],
  ].flat(),
  [2, 4, 2, 2],
)
const flat = (v: Value) => toFlat(v as Tensor)
const close = (a: number[], b: number[], tol = 1e-9) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol))
}

describe('groupNorm and instanceNorm', () => {
  it('give the worked example x[0, 0, 0, 0] = 1 by hand (eps 0)', () => {
    expect(flat(instanceNorm(X, undefined, undefined, 0))[0]).toBeCloseTo(-3 / Math.sqrt(5), 12)
    expect(flat(groupNorm(X, 2, undefined, undefined, 0))[0]).toBeCloseTo(-2.5 / Math.sqrt(3.25), 12)
    expect(flat(groupNorm(X, 1, undefined, undefined, 0))[0]).toBeCloseTo(-3 / Math.sqrt(6.75), 12)
    expect(flat(batchNorm(X, undefined, undefined, { eps: 0 }))[0]).toBeCloseTo(-7 / Math.sqrt(21), 12)
    expect(-3 / Math.sqrt(5)).toBeCloseTo(-1.342, 3)
    expect(-2.5 / Math.sqrt(3.25)).toBeCloseTo(-1.387, 3)
    expect(-3 / Math.sqrt(6.75)).toBeCloseTo(-1.155, 3)
  })

  it('with G = 1 is layerNorm over each example’s (C, H, W)', () => {
    close(flat(groupNorm(X, 1)), flat(reshape(layerNorm(reshape(X, [2, 16])), [2, 4, 2, 2])))
  })

  it('with G = C is layerNorm over each [H W] row, and instanceNorm', () => {
    const rows = flat(reshape(layerNorm(reshape(X, [8, 4])), [2, 4, 2, 2]))
    close(flat(groupNorm(X, 4)), rows)
    close(flat(instanceNorm(X)), rows)
  })

  it('applies the affine per channel', () => {
    const y = flat(groupNorm(X, 4, tensor([2, 1, 1, 1]), tensor([0.5, 0, 0, 0]), 0))
    expect(y[0]).toBeCloseTo(2 * (-3 / Math.sqrt(5)) + 0.5, 12)
    expect(y[4]).toBeCloseTo(flat(instanceNorm(X, undefined, undefined, 0))[4], 12)
  })

  it('is differentiable in x, gamma and beta', () => {
    const w = tensor(
      Array.from({ length: 32 }, (_, i) => Math.sin(i)),
      [2, 4, 2, 2],
    )
    const f = ([x, g, b]: Value[]) => sum(mul(groupNorm(x, 2, g, b), w))
    expect(gradCheck(f, [X, tensor([1, 2, 0.5, 1]), tensor([0, 1, 0, -1])]).ok).toBe(true)
  })

  it('throws for channels not divisible by the groups, or a rank below 3', () => {
    expect(() => groupNorm(X, 3)).toThrow(AifnError)
    expect(() => groupNorm(tensor([[1, 2]]), 1)).toThrow(AifnError)
    expect(() => instanceNorm(tensor([[1, 2]]))).toThrow(AifnError)
  })
})

describe('GroupNorm and InstanceNorm layers', () => {
  it('start as the functional forms without scale or shift', () => {
    const gn = GroupNorm(2, 4)
    close(flat(gn.apply(gn.init(stream(0)), X)), flat(groupNorm(X, 2)))
    const inorm = InstanceNorm(4)
    close(flat(inorm.apply(inorm.init(stream(0)), X)), flat(instanceNorm(X)))
    expect(() => GroupNorm(3, 4)).toThrow(AifnError)
  })
})
