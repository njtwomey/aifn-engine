/**
 * Boolean rules by column generation on the Titanic table coded as 12 binary features (24 distinct (row, label)
 * pairs): for budgets 4 and 6 the Hamming loss equals brute force over rule sets of up to three clauses, the lower
 * bound holds and the LP is solved to optimality; C = 4 gives 477 (bound 472, LP 471.67), C = 6 gives 461 errors
 * certified by the bound; the first pricing step has reduced cost -234 = 20 - 254 at LP objective 711; `dnfPredict`
 * reproduces the errors; incompatible features given as data give the same run as the callback.
 */
import { describe, expect, it } from 'vitest'
import {
  aggregateBinary,
  booleanRuleColumnGeneration,
  dnfCover,
  dnfPredict,
  type DnfClause,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
// Features: class = v (0-3), class != v (4-7), sex male, sex female, age child, age adult.
const group = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
const bin = t.rows.map(([c, s, a]) =>
  [0, 1, 2, 3]
    .map((v) => (c === v ? 1 : 0))
    .concat(
      [0, 1, 2, 3].map((v) => (c !== v ? 1 : 0)),
      [s === 0 ? 1 : 0, s === 1 ? 1 : 0, a === 0 ? 1 : 0, a === 1 ? 1 : 0],
    ),
)
const compatible = (a: number, b: number) => group[a] !== group[b] || (a >= 4 && a < 8 && b >= 4 && b < 8)
const data = aggregateBinary(bin, t.y)

/** The least Hamming loss over rule sets of at most three clauses of at most three features within the budget. */
function brute(C: number): number {
  const clauses: DnfClause[] = []
  const grow = (c: DnfClause, from: number) => {
    if (c.length) clauses.push(c)
    if (c.length === 3) return
    for (let j = from; j < 12; j++) if (c.every((k) => compatible(k, j))) grow([...c, j], j + 1)
  }
  grow([], 0)
  const hamming = (dnf: DnfClause[]) => {
    const cv = dnfCover(data, dnf)
    return data.rows.reduce((a, _, i) => a + data.count[i] * (data.y[i] ? (cv.any[i] ? 0 : 1) : cv.hits[i]), 0)
  }
  let best = hamming([])
  for (let a = 0; a < clauses.length; a++) {
    const ca = 1 + clauses[a].length
    if (ca > C) continue
    best = Math.min(best, hamming([clauses[a]]))
    for (let b = a + 1; b < clauses.length; b++) {
      const cb = ca + 1 + clauses[b].length
      if (cb > C) continue
      best = Math.min(best, hamming([clauses[a], clauses[b]]))
      for (let e = b + 1; e < clauses.length; e++)
        if (cb + 1 + clauses[e].length <= C) best = Math.min(best, hamming([clauses[a], clauses[b], clauses[e]]))
    }
  }
  return best
}

describe('booleanRuleColumnGeneration on the Titanic table', () => {
  it('aggregates 2201 rows into 24 distinct (row, label) pairs', () => {
    expect(data.rows).toHaveLength(24)
    expect(data.count.reduce((a, b) => a + b)).toBe(2201)
  })

  it('matches brute force within each budget, with a valid bound and an optimal LP', () => {
    for (const C of [4, 6]) {
      const r = booleanRuleColumnGeneration(data, { budget: C, maxDegree: 3, compatible })
      const best = brute(C)
      expect(r.hammingLoss).toBe(best)
      expect(r.lowerBound).toBeLessThanOrEqual(best)
      expect(r.lpOptimal).toBe(true)
    }
  })

  it('C = 4: 477, bound 472, LP 471.67', () => {
    const r = booleanRuleColumnGeneration(data, { budget: 4, maxDegree: 3, compatible })
    expect(r.hammingLoss).toBe(477)
    expect(r.lowerBound).toBe(472)
    expect(r.trace[r.trace.length - 1].lpObjective).toBeCloseTo(471.6667, 3)
    expect(r.selected.map((k) => r.clauses[k])).toEqual([[6, 9]])
  })

  it('C = 6: 461 errors, certified by the bound, reproduced by dnfPredict', () => {
    const r = booleanRuleColumnGeneration(data, { budget: 6, maxDegree: 3, compatible })
    expect(r.errors).toBe(461)
    expect(r.lowerBound).toBe(461)
    const dnf = r.selected.map((k) => r.clauses[k])
    expect(dnf).toEqual([
      [6, 9],
      [6, 10],
    ])
    expect(dnfPredict(dnf, bin).filter((v, i) => v !== t.y[i])).toHaveLength(461)
    expect(r.trace[0].reducedCost).toBe(-234)
    expect(r.trace[0].lpObjective).toBe(711)
    expect(r.trace[0].clause).toEqual([6, 9])
    expect(r.trace.every((s) => s.maxMu >= 0 && s.lambda >= -1e-9)).toBe(true)
  })

  it('takes the incompatible pairs as plain data', () => {
    const incompatible: [number, number][] = []
    for (let a = 0; a < 12; a++) for (let b = a + 1; b < 12; b++) if (!compatible(a, b)) incompatible.push([a, b])
    expect(incompatible).toHaveLength(24)
    const viaData = booleanRuleColumnGeneration(data, { budget: 6, incompatible })
    const viaCallback = booleanRuleColumnGeneration(data, { budget: 6, compatible })
    expect(viaData.clauses).toEqual(viaCallback.clauses)
    expect(viaData.hammingLoss).toBe(viaCallback.hammingLoss)
  })
})
