import { describe, expect, it } from 'vitest'
import type { Table } from 'aifn-compute/foundation/contracts'
import { stream, standardNormals, units } from 'aifn-compute/foundation/random'
import {
  associationModel,
  bitset,
  bitsetCount,
  bitsetJaccard,
  correlationModel,
  cutPoints,
  logisticModel,
  regressionModel,
  selectorLanguage,
  subgroupDiscovery,
} from 'aifn-compute/learning/subgroups'
import { fixture } from '../../fixtures'

type Emm = {
  x: number[]
  y: number[]
  label: number[]
  a: number[]
  b: number[]
  inside: number[]
  rhoInside: number
  rhoOutside: number
  fisherZ: number
  slopeInside: number
  slopeOutside: number
  slopeT: number
  cook: number
  wald: number
  yuleInside: number
  yuleOutside: number
}

describe('exceptional-model classes against numpy, statsmodels', () => {
  const f = fixture<{ emm: Emm }>('learning/subgroups').emm
  const cover = bitset(f.x.length, (i) => f.inside[i] === 1)

  it('correlation: ρ inside and outside, and the Fisher z statistic', () => {
    const m = correlationModel(f.x, f.y)
    const fits = m.fit(cover)
    expect(fits.inside.rho).toBeCloseTo(f.rhoInside, 12)
    expect(fits.outside.rho).toBeCloseTo(f.rhoOutside, 12)
    expect(m.quality(cover)).toBeCloseTo(f.fisherZ, 10)
    expect(correlationModel(f.x, f.y, { measure: 'absolute' }).quality(cover)).toBeCloseTo(
      Math.abs(f.rhoInside - f.rhoOutside),
      12,
    )
  })

  it('regression: slopes, the t statistic of equal slopes, and Cook’s distance of the deleted subgroup', () => {
    const m = regressionModel(f.x, f.y)
    const fits = m.fit(cover)
    expect(fits.inside.slope).toBeCloseTo(f.slopeInside, 12)
    expect(fits.outside.slope).toBeCloseTo(f.slopeOutside, 12)
    expect(m.quality(cover)).toBeCloseTo(f.slopeT, 10)
    expect(regressionModel(f.x, f.y, { measure: 'cook' }).quality(cover)).toBeCloseTo(f.cook, 9)
  })

  it('classification: the Wald statistic of the logistic interaction', () => {
    expect(logisticModel(f.x, f.label).quality(cover)).toBeCloseTo(f.wald, 4)
  })

  it('association: Yule’s Q inside and outside', () => {
    const fits = associationModel(f.a, f.b).fit(cover)
    expect(fits.inside.q).toBeCloseTo(f.yuleInside, 12)
    expect(fits.outside.q).toBeCloseTo(f.yuleOutside, 12)
  })
})

describe('exceptional model mining finds a planted flip', () => {
  const n = 500
  const u = units(stream('emm'), 3 * n)
  const z = standardNormals(stream('emm/noise'), 2 * n)
  const group = Array.from({ length: n }, (_, i) => ['a', 'b', 'c'][Math.floor(3 * u[i])])
  const level = Array.from({ length: n }, (_, i) => 10 * u[n + i])
  // The planted threshold is the median cut, so the language can express it exactly.
  const cut = cutPoints(level, 2, 'equal-frequency')[0]
  const inside = group.map((g, i) => g === 'b' && level[i] >= cut)
  const x = Array.from({ length: n }, (_, i) => z[i])
  const y = x.map((v, i) => (inside[i] ? -0.9 : 0.9) * v + 0.4 * z[n + i])
  const label = x.map((v, i) => (u[2 * n + i] < 1 / (1 + Math.exp(-(inside[i] ? -3 : 3) * v)) ? 1 : 0))
  const table: Table = { group, level, x, y, label }
  const lang = selectorLanguage(table, { exclude: ['x', 'y', 'label'], bins: 2 })
  const truth = bitset(n, (i) => inside[i])

  for (const [name, measure] of [
    ['correlation', correlationModel(x, y)],
    ['regression', regressionModel(x, y)],
    ['classification', logisticModel(x, label)],
  ] as const)
    it(name, () => {
      const top = subgroupDiscovery(lang, measure, {
        strategy: 'beam',
        beamWidth: 6,
        maxDepth: 2,
        k: 3,
        minSupport: 20,
      })
      expect(bitsetJaccard(top[0].cover, truth)).toBeGreaterThan(0.85)
    })

  it('Cook’s distance of the planted subgroup exceeds that of random subsets of its size', () => {
    // Cook's distance grows with the rows deleted (deleting most of the ordinary rows moves the fit too), so it is
    // compared at a fixed size rather than searched without a size constraint.
    const m = regressionModel(x, y, { measure: 'cook' })
    const size = bitsetCount(truth)
    const r = units(stream('emm/random'), 20 * n)
    for (let k = 0; k < 20; k++) {
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => r[k * n + a] - r[k * n + b])
      const chosen = new Set(order.slice(0, size))
      expect(m.quality(bitset(n, (i) => chosen.has(i)))).toBeLessThan(m.quality(truth))
    }
  })
})
