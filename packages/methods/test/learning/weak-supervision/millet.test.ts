/**
 * MILLET (Early et al. 2024): replicate padding repeats the boundary values; a short run on WebTraffic learns above
 * chance and scores its interpretations; NDCG@n of an oracle interpretation (the planted mask itself) is 1.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { ndcg } from 'aifn-compute/learning/metrics'
import { webTraffic } from 'aifn-methods/data/synthetic'
import { milletRun, replicatePad, type MilletSnapshot } from 'aifn-methods/learning/weak-supervision'

describe('replicatePad', () => {
  it('repeats the first and last values', () => {
    const y = toFlat(replicatePad(fromData(Float64Array.of(1, 2, 3), [1, 1, 3]), 2, 1) as Tensor)
    expect(Array.from(y)).toEqual([1, 1, 1, 2, 3, 3])
  })
})

describe('milletRun', () => {
  it('learns WebTraffic above chance and scores its interpretations', () => {
    const classes = [0, 1, 5]
    const train = webTraffic(stream('train'), { perClass: 12, classes, samplesPerDay: 12 })
    const test = webTraffic(stream('test'), { perClass: 6, classes, samplesPerDay: 12 })
    let last: MilletSnapshot | undefined
    for (const s of milletRun({ train, test, pooling: 'conjunctive', steps: 120, evaluate: 6, stepSize: 0.02 }))
      last = s
    expect(last!.done).toBe(true)
    expect(last!.history.trainAccuracy.at(-1)!).toBeGreaterThan(0.6)
    expect(Number.isFinite(last!.scores!.aopcr)).toBe(true)
    expect(last!.scores!.ndcg).toBeGreaterThan(0)
    expect(last!.checkpoints[0].step).toBe(0)
  }, 60000)

  it('NDCG@n of the planted mask itself is 1', () => {
    const rel = [0, 0, 1, 1, 1, 0]
    expect(ndcg(rel, rel, { k: 3 })).toBeCloseTo(1, 12)
  })
})
