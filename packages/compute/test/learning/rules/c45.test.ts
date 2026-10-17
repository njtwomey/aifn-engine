/**
 * C4.5rules: Quinlan's pessimistic errors (U25%(0, 6) = 0.206, U25%(0, 9) = 0.143, U25%(0, 1) = 0.750, and the
 * exact U25%(1, 16) = 0.1596 where C4.5 prints 0.157), the bound's definition, rule generalisation, and the depth-3
 * entropy tree of the Titanic table (inlined: the tree is grown by `aifn-methods`) turned into six rules at the
 * tree's accuracy 1740/2201.
 */
import { describe, expect, it } from 'vitest'
import type { Predicate } from 'aifn-compute/learning/explain'
import { logChoose } from 'aifn-compute/numerics/special'
import {
  applyClassRules,
  c45Rules,
  generaliseRule,
  pessimisticError,
  ruleError,
  ruleText,
  type TreeRulesStep,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()

/** $P(\text{Binomial}(N, q) \le E)$. */
const binomialCdf = (E: number, N: number, q: number) => {
  let s = 0
  for (let k = 0; k <= E; k++) s += Math.exp(logChoose(N, k) + k * Math.log(q) + (N - k) * Math.log(1 - q))
  return s
}

describe('pessimisticError', () => {
  it('gives Quinlan’s values', () => {
    expect(pessimisticError(0, 6)).toBeCloseTo(0.206, 3)
    expect(pessimisticError(0, 9)).toBeCloseTo(0.143, 3)
    expect(pessimisticError(0, 1)).toBeCloseTo(0.75, 12)
    expect(pessimisticError(1, 16)).toBeCloseTo(0.15961, 5)
  })

  it('is the q at which E or fewer errors have probability CF', () => {
    for (const [E, N, cf] of [
      [1, 16, 0.25],
      [3, 20, 0.25],
      [10, 200, 0.1],
      [7, 9, 0.5],
    ])
      expect(binomialCdf(E, N, pessimisticError(E, N, cf))).toBeCloseTo(cf, 9)
    expect(pessimisticError(5, 5)).toBe(1)
    expect(pessimisticError(0, 0)).toBe(1)
  })
})

describe('generaliseRule', () => {
  it('drops a condition while the pessimistic error does not rise', () => {
    const rows = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [1, 1],
      [1, 0],
    ]
    const y = [0, 0, 1, 1, 1, 1]
    const gt = (feature: number) => ({ feature, lower: 0.5, upper: Infinity })
    const rule = { conditions: [gt(0), gt(1)], label: 1 }
    const g = generaliseRule(rows, y, rule)
    expect(g.rule.conditions).toEqual([gt(0)])
    expect(g.deletions).toHaveLength(1)
    expect(g.deletions[0].before).toEqual(ruleError(rows, y, rule))
    expect(g.deletions[0].after).toMatchObject({ N: 4, E: 0 })
  })
})

describe('c45Rules on the Titanic depth-3 tree', () => {
  const le = (feature: number, upper: number): Predicate => ({ feature, lower: -Infinity, upper })
  const gt = (feature: number, lower: number): Predicate => ({ feature, lower, upper: Infinity })
  const between = (feature: number, lower: number, upper: number): Predicate => ({ feature, lower, upper })
  // The leaves of growTree(maxDepth 3, entropy) on [class, sex, age], each labelled by its majority.
  const paths = [
    { conditions: [le(0, 1.5), le(1, 0.5), le(2, 0.5)], label: 1 },
    { conditions: [gt(0, 1.5), le(1, 0.5), le(2, 0.5)], label: 0 },
    { conditions: [le(0, 0.5), le(1, 0.5), gt(2, 0.5)], label: 0 },
    { conditions: [gt(0, 0.5), le(1, 0.5), gt(2, 0.5)], label: 0 },
    { conditions: [le(0, 0.5), gt(1, 0.5)], label: 1 },
    { conditions: [between(0, 0.5, 1.5), gt(1, 0.5)], label: 1 },
    { conditions: [between(0, 1.5, 2.5), gt(1, 0.5)], label: 0 },
    { conditions: [gt(0, 2.5), gt(1, 0.5)], label: 1 },
  ]

  it('keeps the tree’s accuracy, 1740/2201, with six rules', () => {
    const tree = applyClassRules(paths, 0, t.rows).filter((v, i) => v === t.y[i]).length
    expect(tree).toBe(1740)
    const res = c45Rules(t.rows, t.y, paths)
    expect(res.rules.map((r) => `${ruleText(r.conditions, t)} -> ${r.label}`)).toEqual([
      'class ≤ 1.50 ∧ age ≤ 0.50 -> 1',
      'class ≤ 0.50 ∧ sex > 0.50 -> 1',
      'class = 2nd ∧ sex > 0.50 -> 1',
      'class > 2.50 ∧ sex > 0.50 -> 1',
      'sex ≤ 0.50 ∧ age > 0.50 -> 0',
      'class = 3rd -> 0',
    ])
    expect(res.defaultLabel).toBe(0)
    expect(applyClassRules(res.rules, res.defaultLabel, t.rows).filter((v, i) => v === t.y[i]).length).toBe(1740)
  })

  it('traces four deletions, a selection per class and the order', () => {
    const res = c45Rules(t.rows, t.y, paths)
    expect(res.trace.map((s) => s.kind)).toEqual([
      'paths',
      'generalise',
      'generalise',
      'generalise',
      'generalise',
      'select',
      'select',
      'order',
    ])
    const select = res.trace.filter((s): s is Extract<TreeRulesStep, { kind: 'select' }> => s.kind === 'select')
    expect(select.map((s) => [s.label, s.kept, s.dropped])).toEqual([
      [0, [2, 6], [1, 3]],
      [1, [0, 4, 5, 7], []],
    ])
    expect(select[0].bits).toBeCloseTo(1593.74, 2)
    expect(select[1].bits).toBeCloseTo(1604.453, 2)
  })
})
