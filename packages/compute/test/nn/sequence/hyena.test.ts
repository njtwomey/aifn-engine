/**
 * aifn-compute/nn/sequence Hyena: the long-convolution note's worked example by FFT and directly
 * (`causalConvolution`'s two methods, also per channel), the order-2 operator by hand, the positional encoding's
 * shape, and a 512-tap implicit filter convolved both ways.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { causalConvolution, hyenaOperator, hyenaPositionalEncoding, implicitFilter } from 'aifn-compute/nn/sequence'

const flat = (v: Value) => toFlat(v as Tensor)
const maxError = (got: number[], want: number[]) => Math.max(...got.map((g, i) => Math.abs(g - want[i])))

describe('Hyena', () => {
  const v = tensor([1, 2, 0, -1])
  const h = tensor([1, 0.5, 0.25, 0])

  it('convolves causally by FFT and directly: (1, 2.5, 1.25, -0.5)', () => {
    const want = [1, 2.5, 1.25, -0.5]
    expect(maxError(flat(causalConvolution(h, v, { method: 'fft' })), want)).toBeLessThan(1e-12)
    expect(maxError(flat(causalConvolution(h, v)), want)).toBeLessThan(1e-12)
  })

  it('convolves each channel with its own kernel by FFT, as directly', () => {
    const K = tensor([
      [1, 2],
      [0.5, 1],
      [0.25, 0],
    ])
    const u = tensor([
      [1, 1],
      [1, 0],
      [0, 0],
      [2, -1],
    ])
    expect(maxError(flat(causalConvolution(K, u, { method: 'fft' })), flat(causalConvolution(K, u)))).toBeLessThan(
      1e-12,
    )
  })

  it('runs the order-2 operator: y = (0.5, -1, 1.25, -3.5)', () => {
    const gates = tensor([
      [1, 0, 1, 1],
      [0.5, 1, 1, 2],
    ])
    const filters = tensor([
      [1, 0.5, 0.25, 0],
      [1, -1, 0, 0],
    ])
    const z = toRows(hyenaOperator(v, gates, filters) as Tensor)
    expect(z[0]).toEqual([1, 2, 0, -1])
    expect(maxError(z[2], [0.5, -1, 1.25, -3.5])).toBeLessThan(1e-12)
  })

  it('encodes positions in 2K + 1 columns with rho_0 = 1', () => {
    const pe = hyenaPositionalEncoding(8, 3)
    expect(pe.shape).toEqual([8, 7])
    expect(toRows(pe)[5][1]).toBe(1)
  })

  it('convolves a 512-tap implicit filter by FFT as directly, under its decaying window', () => {
    const options = { features: 8, width: 32, depth: 2, frequency: 10, decay: 6, bias: 0.01 }
    const f = implicitFilter(512, options, stream(1))
    const u = tensor(Array.from({ length: 512 }, (_, t) => Math.sin(0.05 * t) + (t % 7) / 7))
    const a = flat(causalConvolution(f.filter, u, { method: 'fft' }))
    const b = flat(causalConvolution(f.filter, u))
    expect(maxError(a, b)).toBeLessThan(1e-9)
    expect(Math.abs(toFlat(f.window)[511] - (Math.exp((-6 * 511) / 512) + 0.01))).toBeLessThan(1e-12)
  })
})
