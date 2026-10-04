/**
 * Perturbation curves, AOPC and AOPCR (MILLET's interpretability metric): on an additive predictor the curve removes
 * the right positions block by block; the most-relevant-first order of the true weights has the largest AOPC (it
 * removes the largest weights first), so its AOPCR is positive and the reversed order's negative.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { aopc, aopcr, perturbationCurve } from 'aifn-compute/learning/explain'

const w = [0.1, 3, -0.5, 2, 0.7, 0, 1.5, -1, 0.2, 0.9]
const predict = (kept: Uint8Array) => kept.reduce((s, k, i) => s + k * w[i], 0)

describe('perturbationCurve', () => {
  it('removes blocks in order up to the limit', () => {
    const order = [1, 3, 6, 9, 4, 8, 0, 5, 7, 2]
    const c = perturbationCurve(predict, order, { block: 2, until: 0.5 })
    expect(c.removed).toEqual([0, 2, 4, 5])
    const total = w.reduce((a, v) => a + v, 0)
    expect(c.values[0]).toBeCloseTo(total, 12)
    expect(c.values[1]).toBeCloseTo(total - 3 - 2, 12)
    expect(c.values[3]).toBeCloseTo(total - 3 - 2 - 1.5 - 0.9 - 0.7, 12)
  })
  it('the default removes all but one position, one at a time', () => {
    expect(perturbationCurve(predict, [...w.keys()]).removed.length).toBe(w.length)
  })
})

describe('AOPC and AOPCR', () => {
  it('the true MoRF order maximises AOPC; AOPCR is positive for it and negative reversed', () => {
    const order = [...w.keys()].sort((a, b) => w[b] - w[a])
    const best = aopc(perturbationCurve(predict, order))
    const reversed = aopc(perturbationCurve(predict, [...order].reverse()))
    expect(best).toBeGreaterThan(reversed)
    expect(aopcr(stream(1), predict, w).aopcr).toBeGreaterThan(0)
    expect(
      aopcr(
        stream(1),
        predict,
        w.map((v) => -v),
      ).aopcr,
    ).toBeLessThan(0)
  })
})
