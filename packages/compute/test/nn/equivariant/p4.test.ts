/** p4 group convolution (aifn-compute/nn/equivariant), against the site's self-check (group-convolution). */
import { describe, expect, it } from 'vitest'
import { orientationPool, p4Act, p4GroupConv, p4Lift, rot90 } from 'aifn-compute/nn/equivariant'
import { conv2d } from 'aifn-compute/nn/functional'
import { normals, stream } from 'aifn-compute/foundation/random'
import { abs, max, sub, tensor, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

const err = (a: Value, b: Value) => max(abs(sub(a, b))) as number

describe('rot90', () => {
  it('as numpy.rot90: [[1, 2], [3, 4]] once [[2, 4], [1, 3]], twice [[4, 3], [2, 1]], four times the identity', () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(toRows(rot90(x, 1) as Tensor)).toEqual([
      [2, 4],
      [1, 3],
    ])
    expect(toRows(rot90(x, 2) as Tensor)).toEqual([
      [4, 3],
      [2, 1],
    ])
    expect(toRows(rot90(x, 4) as Tensor)).toEqual([
      [1, 2],
      [3, 4],
    ])
    expect(toRows(rot90(x, -1) as Tensor)).toEqual(toRows(rot90(x, 3) as Tensor))
  })
  it('throws for an odd number of turns of a map that is not square', () => {
    expect(() => rot90(tensor([[1, 2, 3]]), 1)).toThrow()
  })
})

describe('equivariance', () => {
  const x = normals(stream('group-conv-check'), [2, 3, 7, 7])
  const w1 = normals(stream('w1'), [5, 3, 3, 3])
  const w2 = normals(stream('w2'), [4, 5, 4, 3, 3])
  for (const padding of [0, 1]) {
    const lift = (v: Value) => p4Lift(v, w1, { padding })
    const net = (v: Value) => p4GroupConv(lift(v), w2, { padding })
    it(`lifting, two layers and orientation pooling, padding ${padding}, to 1e-12`, () => {
      for (const k of [1, 2, 3]) {
        expect(err(lift(rot90(x, k)), p4Act(lift(x), k))).toBeLessThan(1e-12)
        expect(err(net(rot90(x, k)), p4Act(net(x), k))).toBeLessThan(1e-12)
        expect(err(orientationPool(net(rot90(x, k))), rot90(orientationPool(net(x)), k))).toBeLessThan(1e-12)
      }
    })
  }
  it('a plain convolution with an asymmetric filter breaks it (error > 0.1)', () => {
    expect(err(conv2d(rot90(x, 1), w1), rot90(conv2d(x, w1), 1))).toBeGreaterThan(0.1)
  })
  it('p4Act is a group action: three single turns equal one triple turn', () => {
    const f = p4Lift(x, w1)
    expect(err(p4Act(p4Act(p4Act(f, 1), 1), 1), p4Act(f, 3))).toBeLessThan(1e-12)
  })
  it('throws for filters without an orientation axis of 4', () => {
    expect(() => p4GroupConv(p4Lift(x, w1), normals(stream(0), [4, 5, 3, 3, 3]))).toThrow()
  })
})
