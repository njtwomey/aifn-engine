/**
 * Student noise for Noisy Student: jitter copies rows, and inverted dropout zeroes or doubles features at rate 0.5 and
 * keeps their mean (the site's self-check), with either mask draw.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { featureDropout, inputJitter } from 'aifn-methods/learning/semi-supervised'

const data = dataset(
  tensor([
    [1, 2],
    [3, 4],
  ]),
  tensor([0, 1]),
)

describe('inputJitter', () => {
  it('makes 3 copies of 2 rows, keeping the labels, and copies exactly at sd 0', () => {
    const j = inputJitter(0.1, 3)(data, stream(1))
    expect(toFlat(j.y)).toEqual([0, 1, 0, 1, 0, 1])
    expect(j.x.shape).toEqual([6, 2])
    expect(toFlat(inputJitter(0, 2)(data, stream(1)).x)).toEqual([1, 2, 3, 4, 1, 2, 3, 4])
    expect(() => inputJitter(-1, 1)).toThrow(/sd/)
  })
})

describe('featureDropout', () => {
  for (const mask of ['bernoulli', 'uniform'] as const)
    it(`zeroes or doubles every feature at rate 0.5 and keeps the mean 1 (${mask} masks)`, () => {
      const f = featureDropout(0.5, 200, { mask })(data, stream(1))
      const fx = toFlat(f.x)
      expect(fx.length).toBe(800)
      expect(fx.every((v, i) => v === 0 || v === 2 * [1, 2, 3, 4][i % 4])).toBe(true)
      const meanFirst = fx.filter((_, i) => i % 4 === 0).reduce((a, b) => a + b, 0) / 200
      expect(Math.abs(meanFirst - 1)).toBeLessThan(0.15)
    })

  it('rate 0 returns the data, and a rate of 1 throws', () => {
    expect(featureDropout(0, 3)(data, stream(1))).toBe(data)
    expect(() => featureDropout(1, 3)).toThrow(/rate/)
  })
})
