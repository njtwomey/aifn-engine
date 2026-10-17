/**
 * RuleFit's terms: the 8-row worked example (x = 1..8, y = 1,1,1,1,5,5,5,9, four stump rules) has supports 1/2, 1/2,
 * 7/8, 1/8 and covariances z = (-1.25, 1.25, -0.6875, 0.6875); repeated and constant rules are dropped; a tree of t
 * leaves gives 2(t - 1) rules, each the conjunction of its path; the winsorised linear terms are scaled to standard
 * deviation 0.4.
 */
import { describe, expect, it } from 'vitest'
import { ensembleRules, linearTerms, ruleFeatures, type EnsembleRule } from 'aifn-compute/learning/rules'

const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
const y = [1, 1, 1, 1, 5, 5, 5, 9]
const R = (lower: number, upper: number): EnsembleRule => ({
  conditions: [{ feature: 0, lower, upper }],
  tree: 0,
  node: 0,
})
const stumps = [R(-Infinity, 4.5), R(4.5, Infinity), R(-Infinity, 7.5), R(7.5, Infinity)]

describe('ruleFeatures', () => {
  it('supports and covariances of the worked example', () => {
    const cols = ruleFeatures(rows, stumps)
    expect(cols.support).toEqual([0.5, 0.5, 0.875, 0.125])
    const z = cols.columns.map((c, k) => c.reduce((a, r, i) => a + (r - cols.support[k]) * y[i], 0) / 8)
    expect(z).toEqual([-1.25, 1.25, -0.6875, 0.6875])
    // Unpenalised one-rule effects z / (s(1 - s)): r1 -5, r4 44/7.
    expect(z[0] / 0.25).toBe(-5)
    expect(z[3] / (0.125 * 0.875)).toBeCloseTo(44 / 7, 12)
  })

  it('merges equal coverage and drops constant rules', () => {
    const cols = ruleFeatures(rows, [...stumps, R(-Infinity, 4.6), R(0, Infinity), R(9, Infinity)])
    expect(cols.rules).toHaveLength(4)
    expect(cols.merged).toBe(1)
    expect(cols.constant).toBe(2)
  })
})

describe('ensembleRules', () => {
  const node = (id: number, parent: number | null, children: number[], feature = -1, threshold = 0) => ({
    id,
    parent,
    children,
    feature,
    threshold,
  })
  // x0 <= 0.5 | x0 > 0.5 then x1 <= 2 | x1 > 2 then x0 <= 0.8 | x0 > 0.8: four leaves.
  const tree = {
    root: 0,
    nodes: [
      node(0, null, [1, 2], 0, 0.5),
      node(1, 0, []),
      node(2, 0, [3, 4], 1, 2),
      node(3, 2, []),
      node(4, 2, [5, 6], 0, 0.8),
      node(5, 4, []),
      node(6, 4, []),
    ],
  }

  it('gives 2(t - 1) rules per tree, each its path with one interval per feature', () => {
    const rules = ensembleRules([tree, tree])
    expect(rules).toHaveLength(2 * 2 * (4 - 1))
    expect(rules.slice(0, 6).map((r) => r.node)).toEqual([1, 2, 3, 4, 5, 6])
    expect(rules[4].conditions).toEqual([
      { feature: 0, lower: 0.5, upper: 0.8 },
      { feature: 1, lower: 2, upper: Infinity },
    ])
    expect(rules[6].tree).toBe(1)
  })
})

describe('linearTerms', () => {
  it('winsorises to the beta quantiles and scales to standard deviation 0.4', () => {
    const data = Array.from({ length: 50 }, (_, i) => [i, (i * 7) % 11, i === 49 ? 1000 : i / 3])
    const lt = linearTerms(data, 0.025)
    lt.sd.forEach((s, j) => expect(s * lt.scale[j]).toBeCloseTo(0.4, 12))
    // The 2.5% and 97.5% linear quantiles of 0..49 are 1.225 and 47.775.
    expect(lt.lower[0]).toBeCloseTo(1.225, 12)
    expect(lt.upper[0]).toBeCloseTo(47.775, 12)
    expect(lt.upper[2]).toBeLessThan(1000)
    expect(linearTerms([[1], [1], [1]]).scale).toEqual([0])
  })
})
