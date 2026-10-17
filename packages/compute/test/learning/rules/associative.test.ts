/**
 * Associative classification on the Titanic table (minimum support 1%, confidence 50%, at most three conditions): the
 * 33 class association rules equal a brute-force count, ranked by confidence then support, the top rule 2nd-class
 * children 24/24; CBA's M1 keeps 7 rules with default "survived" and 477 errors, through the totals 687, 550, 550,
 * 483, 483, 483, 477, 483, 481, 478, 478; the 2 x 2 chi-square (456.87 for sex = male -> died) and its maximum; CMAR
 * keeps 17 rules and gets 1708 of 2201 right.
 */
import { describe, expect, it } from 'vitest'
import {
  cbaBuild,
  cbaPredict,
  classAssociationRules,
  cmarBuild,
  cmarPredict,
  coverMask,
  mineAntecedents,
  nominalConditions,
  ruleChiSquare,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
const n = t.y.length
const items = nominalConditions([4, 2, 2])
const ants = mineAntecedents(t.rows, items, { minSupport: 0.01, maxLength: 3 })
const cars = classAssociationRules(ants, t.y, 2, { minSupport: 0.01, minConfidence: 0.5 })
const covers = (r: number, i: number) => cars[r].mask[i] === 1

describe('classAssociationRules', () => {
  it('finds the 33 CARs a brute-force enumeration finds', () => {
    let brute = 0
    for (const c of [-1, 0, 1, 2, 3])
      for (const s of [-1, 0, 1])
        for (const a of [-1, 0, 1]) {
          const pick = (k: number, offset: number) => (k >= 0 ? [items[offset + k]] : [])
          const conds = [...pick(c, 0), ...pick(s, 4), ...pick(a, 6)]
          if (conds.length === 0) continue
          const m = coverMask(t.rows, conds)
          let k = 0
          let k1 = 0
          for (let i = 0; i < n; i++)
            if (m[i]) {
              k++
              k1 += t.y[i]
            }
          if (k === 0) continue
          const best = Math.max(k1, k - k1)
          if (best / n >= 0.01 - 1e-12 && best / k >= 0.5 - 1e-12) brute++
        }
    expect(cars).toHaveLength(33)
    expect(brute).toBe(33)
  })

  it('ranks by confidence, then support; the top rule is 2nd-class children, 24 of 24', () => {
    for (let k = 1; k < cars.length; k++) {
      const [a, b] = [cars[k - 1], cars[k]]
      expect(a.confidence > b.confidence || (a.confidence === b.confidence && a.support >= b.support)).toBe(true)
    }
    expect(cars[0]).toMatchObject({ ruleCount: 24, condCount: 24, label: 1 })
  })
})

describe('CBA', () => {
  const cba = cbaBuild(cars, t.y, 2)

  it('keeps 7 rules, default survived, 477 errors', () => {
    expect(cba.list).toEqual([0, 1, 3, 4, 7, 8, 9])
    expect(cba.defaultLabel).toBe(1)
    expect(cba.errors).toBe(477)
    expect(cba.cut).toBe(9)
    expect(cba.trace).toHaveLength(26)
  })

  it('totals the errors after each kept rule as M1 does', () => {
    expect(cba.trace.filter((s) => s.kept).map((s) => s.totalErrors)).toEqual([
      687, 550, 550, 483, 483, 483, 477, 483, 481, 478, 478,
    ])
  })

  it('predictions reproduce the training errors', () => {
    const p = cbaPredict(cba, cars, covers, n)
    expect(p.filter((v, i) => v !== t.y[i])).toHaveLength(477)
  })
})

describe('ruleChiSquare', () => {
  it('equals the 2 x 2 formula: 456.87 for sex = male -> died', () => {
    const male = cars.find((r) => r.conditions.length === 1 && r.conditions[0].feature === 1 && r.label === 0)
    expect(male).toBeDefined()
    const s = ruleChiSquare(male!.condCount, male!.ruleCount, 1490, n)
    const [a, b, c, d] = [1364, 367, 126, 344]
    expect(s.chi2).toBeCloseTo((n * (a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d)), 6)
    expect(s.chi2).toBeCloseTo(456.874, 3)
    expect(s.maxChi2).toBeCloseTo(1252.384, 3)
  })

  it('reaches its maximum for a rule that matches its class exactly', () => {
    const s = ruleChiSquare(500, 500, 500, 2000)
    expect(s.chi2).toBeCloseTo(2000, 9)
    expect(s.maxChi2).toBeCloseTo(2000, 9)
  })
})

describe('CMAR', () => {
  it('prunes to 17 rules and gets 1708 of 2201 right', () => {
    const cm = cmarBuild(cars, t.y, 2, { coverage: 4 })
    expect(cm.rules).toHaveLength(17)
    expect([cm.prunedGeneral, cm.prunedCorrelation, cm.prunedCoverage, cm.defaultLabel]).toEqual([15, 1, 0, 0])
    const q = cmarPredict(cm, cars, covers, n, 2)
    expect(q.filter((v, i) => v === t.y[i])).toHaveLength(1708)
  })

  it('takes a delta larger than a byte, keeping every rule that still classifies a row', () => {
    const cm = cmarBuild(cars, t.y, 2, { coverage: 300 })
    expect(cm.prunedCoverage).toBe(0)
    expect(cm.rules).toHaveLength(17)
  })
})
