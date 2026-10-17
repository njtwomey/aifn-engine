/**
 * aifn-methods/neural/operator `deepOnet` (the DeepONet note's check): the prediction is the coefficients times the
 * basis plus a bias that starts at 0, of shape $[N, Q]$, and the loss is the mean squared error on aligned data.
 * (The note's training probe takes seconds and depends on the seed, so it is not a unit test.)
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, matmul, mean, square, sub, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { deepOnet, deepOnetLoss } from 'aifn-methods/neural/operator'

describe('DeepONet', () => {
  const net = deepOnet({ sensors: 5, latent: 3 })
  const p = net.init(stream('check/deeponet'))
  const u = fromData(
    Float64Array.from({ length: 10 }, (_, i) => Math.sin(i)),
    [2, 5],
  )
  const y = fromData(Float64Array.from([0, 0.5, 1]), [3, 1])

  it('predicts b(u) t(y)T + b0, shape [N, Q]', () => {
    const out = net.apply(p, u, y) as Tensor
    expect(out.shape).toEqual([2, 3])
    expect(toFlat(p.bias)).toEqual([0])
    const direct = toFlat(matmul(net.coefficients(p, u), transpose(net.basis(p, y))) as Tensor)
    toFlat(out).forEach((v, i) => expect(Math.abs(v - direct[i])).toBeLessThan(1e-12))
    expect(net.label).toBe('DeepONet(m = 5, p = 3)')
  })

  it('scores aligned batches by the mean squared error', () => {
    const G = fromData(Float64Array.of(1, 2, 3, 4, 5, 6), [2, 3])
    const num = (v: unknown) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
    const loss = num(deepOnetLoss(net, y)(p, { u, G }))
    expect(loss).toBeCloseTo(num(mean(square(sub(net.apply(p, u, y), G)))), 12)
  })
})
