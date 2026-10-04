import { describe, expect, it } from 'vitest'
import { anscombe, coalMining, iris, oldFaithful } from 'aifn-methods/data/real'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { bocpd, detectChangepoints, poissonGamma, runLengthMass } from 'aifn-compute/inference/filtering'
import { run } from 'aifn-compute/foundation/trace'

describe('real data', () => {
  it('Iris, Old Faithful and Anscombe have their published sizes and summaries', () => {
    const d = iris()
    expect(d.x.shape).toEqual([150, 4])
    const x = toFlat(d.x)
    const sepal = x.filter((_, i) => i % 4 === 0)
    expect(sepal.reduce((a, b) => a + b, 0) / 150).toBeCloseTo(5.8433, 4)
    expect(oldFaithful().x.shape).toEqual([272, 2])
    for (const set of anscombe()) {
      const xs = toFlat(set.x)
      const ys = toFlat(set.y!)
      expect(xs.reduce((a, b) => a + b, 0) / 11).toBeCloseTo(9, 10)
      expect(ys.reduce((a, b) => a + b, 0) / 11).toBeCloseTo(7.5, 2)
    }
  })
})

describe('coal-mining disasters', () => {
  const d = coalMining()
  const counts = Array.from(toFlat(d.x))
  const years = Array.from(toFlat(d.t!))

  it('has 112 yearly counts from 1851 to 1962 summing to 191', () => {
    expect(d.x.shape).toEqual([112, 1])
    expect(years[0]).toBe(1851)
    expect(years.at(-1)).toBe(1962)
    expect(counts.reduce((a, b) => a + b, 0)).toBe(191)
    expect(counts.every((c) => Number.isInteger(c) && c >= 0)).toBe(true)
    expect(d.meta.task).toBe('sequence')
  })

  it('BOCPD with a Poisson–gamma model places the change in rate in 1892', () => {
    const det = detectChangepoints(poissonGamma({ shape: 1, rate: 1 }), counts, { hazard: 1 / 100 })
    const found = det.changepoints.map((i) => years[i])
    expect(found[0]).toBe(1892)
    // The rate falls from about three disasters a year to under one.
    const k = det.changepoints[0]
    const before = counts.slice(0, k).reduce((a, b) => a + b, 0) / k
    const after = counts.slice(k).reduce((a, b) => a + b, 0) / (112 - k)
    expect(before).toBeGreaterThan(2.5)
    expect(after).toBeLessThan(1.2)
    // The detection does not hinge on the hazard: 1/50 and 1/200 find the same first change.
    for (const hazard of [1 / 50, 1 / 200])
      expect(years[detectChangepoints(poissonGamma(), counts, { hazard }).changepoints[0]]).toBe(1892)
    // At the end of the series most posterior mass is on runs that began in 1892 or later.
    const last = run(bocpd(poissonGamma(), counts, { hazard: 1 / 100 }), undefined, 112)
    expect(last.terminated).toBe(true)
    expect(runLengthMass(last, 0, 112 - k + 1)).toBeGreaterThan(0.5)
    expect(runLengthMass(last, 0)).toBeCloseTo(1, 12)
  })
})
