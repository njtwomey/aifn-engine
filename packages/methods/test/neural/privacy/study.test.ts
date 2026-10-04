/**
 * The DP-SGD study: snapshots stream per checkpoint; the non-private baseline (σ = 0) learns two blobs and reports
 * ε = ∞; private runs report finite ε that grows with the steps and shrinks as σ grows.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { blobs } from 'aifn-methods/data/synthetic'
import { privateTrainingStudy, type PrivateStudySnapshot } from 'aifn-methods/neural/privacy'

describe('privateTrainingStudy', () => {
  it('trains each noise multiplier and accounts ε', () => {
    const data = blobs(stream(1), { n: 200, centers: 2, sd: 1, separation: 4 })
    const test = blobs(stream(2), { n: 100, centers: 2, sd: 1, separation: 4 })
    let last: PrivateStudySnapshot | undefined
    for (const s of privateTrainingStudy(data, test, { noiseMultipliers: [0, 1, 3], steps: 60, checkpoints: 6 }))
      last = s
    expect(last!.done).toBe(180)
    const [base, one, three] = last!.runs
    expect(base.accuracy.at(-1)).toBeGreaterThan(0.9)
    expect(base.epsilon.at(-1)).toBe(Infinity)
    expect(one.epsilon.at(-1)).toBeGreaterThan(one.epsilon[1])
    expect(three.epsilon.at(-1)).toBeLessThan(one.epsilon.at(-1)!)
    expect(one.theta.length).toBe(one.at.length)
  })
})
