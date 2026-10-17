/**
 * aifn-compute/signal/image resampling: `bilinearSample` at pixels, cell centres and outside the image, exact with
 * its gradient on a bilinear function; `upsampleBilinear` against torch's `interpolate(align_corners=False)` (the
 * site's self-checks).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { bilinearSample, upsampleBilinear } from 'aifn-compute/signal/image'

describe('bilinearSample', () => {
  const img = [
    [0, 1],
    [2, 3],
  ]
  it('reads the pixel at an integer position, the mean at a cell centre, zero outside', () => {
    expect(bilinearSample(img, 1, 0).value).toBe(2)
    expect(bilinearSample(img, 0.5, 0.5).value).toBe(1.5)
    expect(bilinearSample(img, -1.5, 0).value).toBe(0)
    expect(bilinearSample(img, 0.5, 0.5).taps.map((t) => t.weight)).toEqual([0.25, 0.25, 0.25, 0.25])
  })

  it('is exact, with its gradient, on a bilinear function', () => {
    const f = (r: number, c: number) => 0.3 + 1.7 * r - 0.4 * c + 0.25 * r * c
    const im = Array.from({ length: 4 }, (_, r) => Array.from({ length: 5 }, (_, c) => f(r, c)))
    const s = bilinearSample(im, 1.37, 2.81)
    expect(s.value).toBeCloseTo(f(1.37, 2.81), 12)
    expect(s.gradient[0]).toBeCloseTo(1.7 + 0.25 * 2.81, 12)
    expect(s.gradient[1]).toBeCloseTo(-0.4 + 0.25 * 1.37, 12)
  })

  it('gives the deformable-convolution worked example: (1.4, 1.8) on the ridge reads 4.92', () => {
    const X = [0, 0, 1, 4, 8, 0, 1, 4, 8, 4, 1, 4, 8, 4, 1, 4, 8, 4, 1, 0, 8, 4, 1, 0, 0]
    const rows = Array.from({ length: 5 }, (_, r) => X.slice(5 * r, 5 * r + 5))
    const s = bilinearSample(rows, 1.4, 1.8)
    expect(s.value).toBeCloseTo(4.92, 12)
    s.taps.map((t) => t.weight).forEach((w, i) => expect(w).toBeCloseTo([0.12, 0.48, 0.08, 0.32][i], 12))
  })
})

describe('upsampleBilinear', () => {
  it('matches torch with half-pixel centres', () => {
    const rows = toRows(
      upsampleBilinear(
        [
          [1, 2],
          [3, 4],
        ],
        [4, 4],
      ),
    )
    expect(rows[0]).toEqual([1, 1.25, 1.75, 2])
    expect(rows[1]).toEqual([1.5, 1.75, 2.25, 2.5])
  })

  it('keeps a constant image constant', () => {
    const out = toFlat(
      upsampleBilinear(
        [
          [7, 7],
          [7, 7],
        ],
        [3, 5],
      ),
    )
    expect(out.length).toBe(15)
    out.forEach((v) => expect(v).toBeCloseTo(7, 12))
  })

  it('throws for an empty output', () => {
    expect(() => upsampleBilinear([[1]], [0, 2])).toThrow(AifnError)
  })
})
