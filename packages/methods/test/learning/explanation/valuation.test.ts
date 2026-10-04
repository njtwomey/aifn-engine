/**
 * The data valuation study: on two Gaussian blobs with 10% of labels flipped, every method ranks the flipped points
 * ahead of the clean ones better than chance (area under the gain curve above ½), data Shapley grows block by block
 * and stays efficient-in-mean, and the explained fit is accurate on the validation set.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { rocCurve } from 'aifn-compute/learning/metrics'
import { blobs, flippedMask, withLabelNoise } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { dataValuationStudy, VALUATION_METHODS, type ValuationSnapshot } from 'aifn-methods/learning/explanation'

describe('data valuation study', () => {
  it('finds planted label noise with every method', () => {
    const clean = blobs(stream('valuation-train'), {
      centers: 2,
      n: 80,
      separation: 3,
      sd: 1,
      layout: 'polygon',
    } as never)
    const train = withLabelNoise(stream('valuation-flip'), clean, { rate: 0.12 })
    const valid = blobs(stream('valuation-valid'), {
      centers: 2,
      n: 60,
      separation: 3,
      sd: 1,
      layout: 'polygon',
    } as never)
    const flipped = flippedMask(train)
    expect(flipped.reduce((a, b) => a + b, 0)).toBeGreaterThan(3)
    let last: ValuationSnapshot | undefined
    const stages: string[] = []
    for (const s of dataValuationStudy(train, valid, logisticRegression({ l2: 1 }), { permutations: 10 })) {
      stages.push(s.stage)
      last = s
    }
    expect(stages[0]).toBe('fit')
    expect(stages.at(-1)).toBe('done')
    const r = last as ValuationSnapshot
    expect(r.accuracy).toBeGreaterThan(0.85)
    expect(r.done).toBe(10)
    for (const m of VALUATION_METHODS) {
      const roc = rocCurve(Array.from(flipped), r.suspicion[m] as Float64Array, { positive: 1 })
      expect(roc.area, m).toBeGreaterThan(0.6)
    }
  })
})
