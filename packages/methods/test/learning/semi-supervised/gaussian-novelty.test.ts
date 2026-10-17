/**
 * Gaussian novelty scores on the open-set note's worked example (the site's self-check): means $\mp 1.2$, pooled
 * variance 0.32, scores 16.53, 5.28, 4.50, 1.53, and the p-value of 4.5, $2(1 - \Phi(\sqrt{4.5})) = 0.0339$.
 */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'
import { gaussianNoveltyScores } from 'aifn-methods/learning/semi-supervised'

describe('gaussianNoveltyScores', () => {
  const x = [[-1.6], [-0.8], [0.8], [1.6], [3.5], [2.5], [0], [1.9]]
  const r = gaussianNoveltyScores(x, [0, 0, 1, 1, -1, -1, -1, -1], 2)

  it('pools the variance of the labelled points about their class means', () => {
    expect(toFlat(r.means)).toEqual([-1.2, 1.2].map((m) => expect.closeTo(m, 12)))
    expect(r.variance).toBeCloseTo(0.32, 12)
  })

  it('scores each point by its nearest class, with chi-squared p-values', () => {
    expect(Math.abs(r.score[4] - 5.29 / 0.32)).toBeLessThan(1e-9)
    expect(Math.abs(r.score[5] - 1.69 / 0.32)).toBeLessThan(1e-9)
    expect(Math.abs(r.score[6] - 4.5)).toBeLessThan(1e-9)
    expect(Math.abs(r.score[7] - 0.49 / 0.32)).toBeLessThan(1e-9)
    expect(Array.from(r.nearest.slice(4))).toEqual([1, 1, 0, 1])
    expect(r.pValue[6]).toBeCloseTo(0.03389, 4)
    expect(r.pValue[6]).toBeCloseTo(2 * (1 - normalCdf(Math.sqrt(4.5))), 9)
  })

  it('refuses a class with no label and too few labels', () => {
    expect(() => gaussianNoveltyScores(x, [0, 0, -1, -1, -1, -1, -1, -1], 2)).toThrow(/every class/)
    expect(() => gaussianNoveltyScores([[0], [1]], [0, 1], 2)).toThrow(/more labelled/)
  })
})
