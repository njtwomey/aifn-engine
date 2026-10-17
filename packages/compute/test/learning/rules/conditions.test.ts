/**
 * Conditions and covers: threshold and nominal conditions, masks and counts on the Titanic coding (1667 adult men, 338
 * of them survivors), rule text with level names, level-wise antecedent mining against brute force, and the decision
 * list applier.
 */
import { describe, expect, it } from 'vitest'
import {
  andMask,
  applyClassRules,
  coverMask,
  mineAntecedents,
  nominalConditions,
  ruleCounts,
  ruleText,
  thresholdConditions,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()

describe('conditions', () => {
  it('thresholds come in pairs at the inner quantile edges', () => {
    const rows = [[1], [2], [3], [4], [5], [6], [7], [8]]
    const c = thresholdConditions(rows, 4)
    expect(c).toEqual([
      { feature: 0, lower: -Infinity, upper: 2.75 },
      { feature: 0, lower: 2.75, upper: Infinity },
      { feature: 0, lower: -Infinity, upper: 4.5 },
      { feature: 0, lower: 4.5, upper: Infinity },
      { feature: 0, lower: -Infinity, upper: 6.25 },
      { feature: 0, lower: 6.25, upper: Infinity },
    ])
  })

  it('nominal levels are width-1 intervals, feature by feature', () => {
    const c = nominalConditions([4, 2, 2])
    expect(c).toHaveLength(8)
    expect(c[5]).toEqual({ feature: 1, lower: 0.5, upper: 1.5 })
    expect(nominalConditions([2], [3])).toEqual([
      { feature: 3, lower: -0.5, upper: 0.5 },
      { feature: 3, lower: 0.5, upper: 1.5 },
    ])
  })

  it('reproduce the Titanic counts: 1667 adult men, 338 survivors', () => {
    const c = nominalConditions([4, 2, 2])
    const adultMen = coverMask(t.rows, [c[4], c[7]])
    expect(adultMen.reduce((a, v) => a + v, 0)).toBe(1667)
    expect(andMask(coverMask(t.rows, [c[4]]), coverMask(t.rows, [c[7]]))).toEqual(adultMen)
    const k = ruleCounts(adultMen, t.y)
    expect(k).toEqual({ p: 338, n: 1329, P: 711, N: 1490 })
    // Within an active subset: the adults only.
    const adults = coverMask(t.rows, [c[7]])
    expect(ruleCounts(adultMen, t.y, adults)).toEqual({ p: 338, n: 1329, P: 654, N: 1438 })
  })

  it('print with names and levels', () => {
    const c = nominalConditions([4, 2, 2])
    expect(ruleText([c[0], c[5]], t)).toBe('class = 1st ∧ sex = female')
    expect(ruleText([{ feature: 0, lower: 0.5, upper: 2.5 }], t)).toBe('class ∈ {2nd, 3rd}')
    expect(ruleText([{ feature: 1, lower: -Infinity, upper: 0.25 }])).toBe('x2 ≤ 0.25')
    expect(ruleText([{ feature: 0, lower: 1, upper: 2 }], { digits: 1 })).toBe('1.0 < x1 ≤ 2.0')
    expect(ruleText([])).toBe('true')
  })
})

describe('mineAntecedents', () => {
  it('finds the frequent conjunctions of the Titanic table (8 singletons, 26 up to pairs, 36 up to triples)', () => {
    const items = nominalConditions([4, 2, 2])
    const two = mineAntecedents(t.rows, items, { minSupport: 0.01, maxLength: 2 })
    expect(two).toHaveLength(26)
    expect(two.filter((a) => a.conditions.length === 1)).toHaveLength(8)
    expect(mineAntecedents(t.rows, items, { minSupport: 0.01, maxLength: 3 })).toHaveLength(36)
  })

  it('equals brute force over every conjunction of distinct features', () => {
    const items = nominalConditions([4, 2, 2])
    const minSupport = 0.05
    const found = new Set(
      mineAntecedents(t.rows, items, { minSupport, maxLength: 3 }).map((a) => ruleText(a.conditions, t)),
    )
    const brute = new Set<string>()
    for (const c of [-1, 0, 1, 2, 3])
      for (const s of [-1, 0, 1])
        for (const g of [-1, 0, 1]) {
          const pick = (k: number, offset: number) => (k >= 0 ? [items[offset + k]] : [])
          const conds = [...pick(c, 0), ...pick(s, 4), ...pick(g, 6)]
          if (conds.length === 0) continue
          const m = coverMask(t.rows, conds)
          if (m.reduce((a, v) => a + v, 0) / t.rows.length >= minSupport) brute.add(ruleText(conds, t))
        }
    expect(found).toEqual(brute)
  })

  it('carries masks and supports', () => {
    const [first] = mineAntecedents(t.rows, nominalConditions([4, 2, 2]))
    expect(first.support).toBeCloseTo(325 / 2201, 12)
    expect(first.mask.reduce((a, v) => a + v, 0)).toBe(325)
  })
})

describe('applyClassRules', () => {
  it('gives the first matching rule or the default', () => {
    const rules = [
      { conditions: [{ feature: 0, lower: 2, upper: Infinity }], label: 1 },
      { conditions: [{ feature: 1, lower: 0.5, upper: Infinity }], label: 2 },
    ]
    expect(
      Array.from(
        applyClassRules(rules, 0, [
          [3, 1],
          [1, 1],
          [1, 0],
        ]),
      ),
    ).toEqual([1, 2, 0])
  })
})
