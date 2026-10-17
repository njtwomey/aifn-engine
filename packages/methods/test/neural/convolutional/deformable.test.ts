/**
 * aifn-methods/neural/convolutional `deformableConvolutionAt`: the worked example of the deformable-convolution note (a
 * 5 x 5 ridge, a mean kernel at (2, 2)) with its bilinear taps and offset gradient, checked by a finite difference
 * (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { deformableConvolutionAt } from 'aifn-methods/neural/convolutional'

const X = [0, 0, 1, 4, 8, 0, 1, 4, 8, 4, 1, 4, 8, 4, 1, 4, 8, 4, 1, 0, 8, 4, 1, 0, 0]
const kernel = new Array(9).fill(1 / 9)
const offsets: [number, number][] = [
  [0.4, 0.8],
  [0, 0.5],
  [0, 0],
  [0.5, 0],
  [0, 0],
  [-0.5, 0],
  [0, 0],
  [0, -0.5],
  [-0.8, -0.4],
]

describe('deformableConvolutionAt', () => {
  it('gives 42/9 on the regular grid', () => {
    const regular = deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], new Array(9).fill([0, 0]))
    expect(regular.value).toBeCloseTo(42 / 9, 12)
    expect(regular.taps[0].regular).toEqual([1, 1])
  })

  it('gives 57.84/9 on the bent grid, with tap (1.4, 1.8) reading 4.92', () => {
    const d = deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], offsets)
    expect(d.value).toBeCloseTo(57.84 / 9, 12)
    expect(d.taps[0].position[0]).toBeCloseTo(1.4, 12)
    expect(d.taps[0].position[1]).toBeCloseTo(1.8, 12)
    expect(d.taps[0].sample.value).toBeCloseTo(4.92, 12)
    d.taps[0].sample.taps.forEach((t, i) => expect(t.weight).toBeCloseTo([0.12, 0.48, 0.08, 0.32][i], 12))
    expect(d.taps[8].sample.value).toBeCloseTo(4.92, 12)
    expect(d.offsetGradient[0][0]).toBeCloseTo(3.8 / 9, 12)
    expect(d.offsetGradient[0][1]).toBeCloseTo(3.4 / 9, 12)
  })

  it('has the offset gradient of a finite difference, and halves with masks of 0.5', () => {
    const d = deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], offsets)
    const h = 1e-6
    const bumped = offsets.map((o, n): [number, number] => (n === 0 ? [o[0] + h, o[1]] : o))
    const fd = (deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], bumped).value - d.value) / h
    expect(Math.abs(fd - 3.8 / 9)).toBeLessThan(1e-5)
    const masked = deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], offsets, { mask: new Array(9).fill(0.5) })
    expect(masked.value).toBeCloseTo(d.value / 2, 12)
  })

  it('throws for an even kernel or the wrong number of offsets', () => {
    expect(() => deformableConvolutionAt(X, 5, 5, new Array(4).fill(1), 2, [2, 2], new Array(4).fill([0, 0]))).toThrow(
      AifnError,
    )
    expect(() => deformableConvolutionAt(X, 5, 5, kernel, 3, [2, 2], offsets.slice(1))).toThrow(AifnError)
  })
})
