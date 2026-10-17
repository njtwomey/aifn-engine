/**
 * CORELS on the Titanic table: the certified optimum equals brute force over short lists (8 single-condition
 * antecedents at lambda = 0.01, and 26 antecedents of up to two conditions, where the optimum is "3rd class died,
 * adult men died, else survived", 461 errors); switching the extra bounds off leaves the optimum unchanged but
 * searches far longer (496 steps against 27); best-first order reaches the same certificate.
 */
import { describe, expect, it } from 'vitest'
import { corels, mineAntecedents, nominalConditions, ruleText, type Antecedent } from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
const n = t.y.length

/** The regularised risk of a list: misclassified rows over n (majority labels) plus lambda per rule. */
function risk(list: number[], ants: readonly Antecedent[], lambda: number): number {
  const counts = Array.from({ length: list.length + 1 }, () => [0, 0])
  for (let i = 0; i < n; i++) {
    const k = list.findIndex((j) => ants[j].mask[i])
    counts[k < 0 ? list.length : k][t.y[i]]++
  }
  return counts.reduce((a, [c0, c1]) => a + Math.min(c0, c1), 0) / n + lambda * list.length
}

/** The least risk over every list of at most `L` distinct antecedents. */
function brute(ants: readonly Antecedent[], lambda: number, L: number): number {
  let best = risk([], ants, lambda)
  const rec = (list: number[]) => {
    if (list.length >= L) return
    for (let j = 0; j < ants.length; j++)
      if (!list.includes(j)) {
        const l = [...list, j]
        best = Math.min(best, risk(l, ants, lambda))
        rec(l)
      }
  }
  rec([])
  return best
}

const off = { lookahead: false, support: false, accurateSupport: false, permutation: false, equivalentPoints: false }

describe('corels with 8 single-condition antecedents', () => {
  const ants = mineAntecedents(t.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength: 1 })
  const r = corels(ants, t.y, { lambda: 0.01 })

  it('certifies the brute-force optimum', () => {
    expect(ants).toHaveLength(8)
    expect(r.certified).toBe(true)
    expect(r.bestObjective).toBeCloseTo(brute(ants, 0.01, 4), 12)
    expect(r.bestObjective).toBeCloseTo(493 / n + 0.01, 12)
    expect(r.best.map((j) => ruleText(ants[j].conditions, t))).toEqual(['sex = male'])
    expect(r.labels).toEqual([0])
    expect(r.defaultLabel).toBe(1)
    expect([r.nodes.length, r.steps.length]).toEqual([37, 27])
  })

  it('finds the same optimum with the extra bounds off, in far more steps', () => {
    const plain = corels(ants, t.y, { lambda: 0.01, bounds: off, maxNodes: 20000 })
    expect(plain.bestObjective).toBe(r.bestObjective)
    expect(plain.certified).toBe(true)
    expect([plain.nodes.length, plain.steps.length]).toEqual([2097, 496])
  })

  it('records a status for every generated prefix', () => {
    const counts: Record<string, number> = {}
    for (const node of r.nodes) counts[node.status] = (counts[node.status] ?? 0) + 1
    expect(counts).toEqual({ expanded: 5, lookahead: 5, bound: 5, 'equivalent-points': 16, permutation: 6 })
  })
})

describe('corels with 26 antecedents of up to two conditions', () => {
  const ants = mineAntecedents(t.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength: 2 })

  it('certifies 3rd class died, adult men died, else survived (461 errors)', () => {
    const r = corels(ants, t.y, { lambda: 0.01 })
    expect(r.certified).toBe(true)
    expect(r.best.map((j) => ruleText(ants[j].conditions, t))).toEqual(['class = 3rd', 'sex = male ∧ age = adult'])
    expect(r.labels).toEqual([0, 0])
    expect(r.defaultLabel).toBe(1)
    expect(r.bestObjective).toBeCloseTo(461 / n + 0.02, 12)
    expect(r.bestObjective).toBeCloseTo(brute(ants, 0.01, 2), 12)
    expect(r.steps).toHaveLength(171)
  })

  it('reaches the same certificate best first', () => {
    const r = corels(ants, t.y, { lambda: 0.01, order: 'best' })
    expect(r.certified).toBe(true)
    expect(r.bestObjective).toBeCloseTo(461 / n + 0.02, 12)
    expect(r.steps).toHaveLength(437)
  })

  it('gives no certificate when the node budget runs out', () => {
    const r = corels(ants, t.y, { lambda: 0.01, bounds: off, maxNodes: 50 })
    expect(r.certified).toBe(false)
  })
})
