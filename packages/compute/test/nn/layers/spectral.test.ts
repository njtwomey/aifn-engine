/**
 * aifn-compute/nn/layers `SpectralConv1d` (the Fourier neural operator note's check): the same weights on a
 * band-limited input at 16 and 64 points give the same function, the output has the input's grid and the output
 * channels, too few points for the modes throw, and the layer differentiates in its weights.
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { mul, sum, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { SpectralConv1d, type SpectralConvParams } from 'aifn-compute/nn/layers'

const signal = (n: number) =>
  tensor([
    Array.from({ length: n }, (_, j) => {
      const x = j / n
      return [Math.sin(2 * Math.PI * x) + 0.5 * Math.cos(4 * Math.PI * x), 1 + 0.25 * Math.sin(6 * Math.PI * x)]
    }),
  ])

describe('SpectralConv1d', () => {
  const layer = SpectralConv1d(2, 3, 4)
  const p = layer.init(stream(0))

  it('gives the same function at 16 and 64 points for a band-limited input', () => {
    const coarse = toFlat(layer.apply(p, signal(16)) as Tensor)
    const fine = layer.apply(p, signal(64)) as Tensor
    expect(fine.shape).toEqual([1, 64, 3])
    const f = toFlat(fine)
    let worst = 0
    for (let j = 0; j < 16; j++)
      for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(coarse[j * 3 + c] - f[4 * j * 3 + c]))
    expect(worst).toBeLessThan(1e-12)
  })

  it('draws weights in [0, 1/(in out)) and refuses too few points for its modes', () => {
    expect(p.real.shape).toEqual([4, 2, 3])
    expect(toFlat(p.imag).every((w) => w >= 0 && w < 1 / 6)).toBe(true)
    expect(() => layer.apply(p, signal(4))).toThrow(/points/)
  })

  it('differentiates in its weights', () => {
    const small = SpectralConv1d(1, 1, 2)
    const q = small.init(stream(1))
    const x = tensor([[[1], [0.5], [-1], [0.25]]])
    const w = tensor([[[1], [-2], [0.5], [3]]])
    const loss = (r: SpectralConvParams) => sum(mul(small.apply(r, x), w))
    expect(gradCheck(loss, q, { rtol: 1e-5, atol: 1e-7 }).ok).toBe(true)
  })
})
