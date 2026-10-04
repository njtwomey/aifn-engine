import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  bitsetAndCount,
  bitsetCount,
  bitsetJaccard,
  correlationModel,
  logisticModel,
  meanShiftQuality,
  selectorLanguage,
  subgroupDiscovery,
  wraccQuality,
} from 'aifn-compute/learning/subgroups'
import { plantedModelFlip, plantedSubgroups } from 'aifn-methods/data/synthetic'
import { titanic } from 'aifn-methods/data/real'

const numbers = (c: unknown) => c as number[]

describe('plantedSubgroups', () => {
  const d = plantedSubgroups(stream('planted'), { n: 2000 })
  const lang = selectorLanguage(d.table, { exclude: d.targets, bins: 6 })
  const main = lang.cover(d.planted[0].description)
  const y = numbers(d.table.outcome)

  it('has the planted rates', () => {
    const n = bitsetCount(main)
    const tp = bitsetAndCount(main, wraccQuality(y).target)
    expect(tp / n).toBeGreaterThan(0.6)
    expect(tp / n).toBeLessThan(0.8)
    expect(Object.values(d.table).every((c) => (c as unknown[]).length === 2000)).toBe(true)
  })

  it('subgroup discovery ranks the planted subgroup first, for the rate and for the cost', () => {
    const top = subgroupDiscovery(lang, wraccQuality(y), { beamWidth: 10, maxDepth: 2, k: 5 })
    expect(bitsetJaccard(top[0].cover, main)).toBeGreaterThan(0.8)
    const cost = subgroupDiscovery(lang, meanShiftQuality(numbers(d.table.cost)), { maxDepth: 2, k: 3 })
    expect(bitsetJaccard(cost[0].cover, main)).toBeGreaterThan(0.8)
  })
})

describe('plantedModelFlip', () => {
  const d = plantedModelFlip(stream('flip'), { n: 1500 })
  const lang = selectorLanguage(d.table, { exclude: d.targets, bins: 2 })
  it('flips the correlation and the classifier inside their subgroups', () => {
    const corr = correlationModel(numbers(d.table.x), numbers(d.table.y))
    const fits = corr.fit(lang.cover(d.planted[0].description))
    expect(fits.inside.rho).toBeLessThan(-0.5)
    expect(fits.outside.rho).toBeGreaterThan(0.5)
    const top = subgroupDiscovery(lang, corr, { beamWidth: 6, maxDepth: 2, k: 3, minSupport: 20 })
    expect(bitsetJaccard(top[0].cover, lang.cover(d.planted[0].description))).toBeGreaterThan(0.85)
    const cls = logisticModel(numbers(d.table.x), numbers(d.table.label))
    const f2 = cls.fit(lang.cover(d.planted[1].description))
    expect(f2.inside.slope).toBeLessThan(0)
    expect(f2.outside.slope).toBeGreaterThan(0)
  })
})

describe('titanic', () => {
  it('has the 2201 people of R’s table', () => {
    const t = titanic()
    const s = numbers(t.table.survived)
    expect(s.length).toBe(2201)
    expect(s.reduce((a, b) => a + b, 0)).toBe(711)
    expect((t.table.class as string[]).filter((c) => c === 'crew').length).toBe(885)
    expect((t.table.sex as string[]).filter((c) => c === 'female').length).toBe(470)
    const lang = selectorLanguage(t.table, { exclude: t.targets })
    const top = subgroupDiscovery(lang, wraccQuality(s), { maxDepth: 2, k: 1 })
    expect(top[0].description).toEqual([{ attribute: 'sex', op: '=', value: 'female' }])
  })
})
