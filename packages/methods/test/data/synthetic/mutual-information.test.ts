/** aifn-methods/data/synthetic: `gaussianPairLogRatios`, scored Gaussian pairs with a known mutual information. */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  correlationForMutualInformation,
  donskerVaradhanBound,
  infoNceBound,
  nwjBound,
} from 'aifn-compute/probability/information'
import { gaussianPairLogRatios } from 'aifn-methods/data/synthetic'

describe('gaussianPairLogRatios', () => {
  it('reports the true mutual information and an n × n score matrix', () => {
    const b = gaussianPairLogRatios(stream(1), { n: 5, rho: 0.6, dims: 2 })
    expect(b.scores.length).toBe(25)
    expect(b.mutualInformation).toBeCloseTo(-Math.log(1 - 0.36), 14)
    expect(gaussianPairLogRatios(stream(1), { n: 5, rho: 0.6, dims: 2 }).scores).toEqual(b.scores)
    expect(gaussianPairLogRatios(stream(1), { n: 4, rho: 0 }).scores.every((v) => v === 0)).toBe(true)
  })

  it('at the optimal critic NWJ is unbiased and InfoNCE stays below the truth and log n (200 batches)', () => {
    const R = 200
    const n = 64
    const expected = [
      { mi: 0.5, nwj: 0.5038194178606792, nce: 0.49486092037946117, dv: 0.5041510036805186 },
      { mi: 2, nwj: 1.9931070696803368, nce: 1.903991387748823, dv: 1.997673144767022 },
    ]
    for (const { mi, nwj: nwjWant, nce: nceWant, dv: dvWant } of expected) {
      const rho = correlationForMutualInformation(mi)
      let nwj = 0
      let nce = 0
      let dv = 0
      for (let r = 0; r < R; r++) {
        const b = gaussianPairLogRatios(child(stream('check'), r), { n, rho })
        nce += infoNceBound(b.scores, n) / R
        dv += donskerVaradhanBound(b.scores, n) / R
        nwj +=
          nwjBound(
            b.scores.map((v) => v + 1),
            n,
          ) / R
      }
      expect(Math.abs(nwj - mi)).toBeLessThan(0.1 * Math.max(1, mi))
      expect(nce).toBeLessThanOrEqual(Math.min(mi + 0.05, Math.log(n)))
      expect(nwj).toBeCloseTo(nwjWant, 9)
      expect(nce).toBeCloseTo(nceWant, 9)
      expect(dv).toBeCloseTo(dvWant, 9)
    }
  })

  it('refuses a correlation outside [0, 1)', () => {
    expect(() => gaussianPairLogRatios(stream(1), { n: 4, rho: 1 })).toThrow(/rho/)
  })
})
