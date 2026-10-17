/**
 * Sequential covering on the Titanic table (the sequential-covering and decision-lists notes): Laplace, precision and
 * WRAcc reach the attainable training accuracy 1740/2201 (the per-cell majority; FOIL gain stops at 1724), the Laplace
 * scores of the first rule's beam at depths 1 to 3 (0.7877, 0.8564, 0.9231), a wider beam finding first-class women
 * (Laplace 0.9660) first, WRAcc's five-rule list, target mode, the significance test and the unordered vote.
 */
import { describe, expect, it } from 'vitest'
import {
  applyCoveringRules,
  nominalConditions,
  ruleText,
  sequentialCovering,
  type CoveringStep,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
const conditions = nominalConditions([4, 2, 2])
const correct = (pred: ArrayLike<number>) => Array.from(pred).filter((v, i) => v === t.y[i]).length

describe('sequentialCovering on the Titanic table', () => {
  it('reaches the per-cell majority accuracy 1740/2201 with Laplace, precision and WRAcc', () => {
    for (const measure of ['laplace', 'precision', 'wracc'] as const) {
      const r = sequentialCovering(t.rows, t.y, conditions, { measure, beam: 3, maxConditions: 3, minSupport: 2 })
      expect(correct(applyCoveringRules(r, t.rows)), measure).toBe(1740)
    }
    const foil = sequentialCovering(t.rows, t.y, conditions, { measure: 'foil-gain' })
    expect(correct(applyCoveringRules(foil, t.rows))).toBe(1724)
    expect(foil.rules.map((r) => ruleText(r.conditions, t))).toEqual([
      'sex = male',
      'class = 1st',
      'class = 2nd',
      'class = crew',
      'age = child',
    ])
  })

  it('traces the Laplace beam of the first rule: 0.7877, 0.8564, 0.9231', () => {
    const r = sequentialCovering(t.rows, t.y, conditions)
    const grow = r.trace.filter((s): s is Extract<CoveringStep, { kind: 'grow' }> => s.kind === 'grow' && s.rule === 0)
    expect(grow.map((s) => s.depth)).toEqual([1, 2, 3])
    expect(grow.map((s) => +s.best.score.toFixed(4))).toEqual([0.7877, 0.8564, 0.9231])
    expect(grow.map((s) => ruleText(s.best.conditions, t))).toEqual([
      'sex = male',
      'class = 2nd ∧ sex = male',
      'class = 2nd ∧ sex = male ∧ age = child',
    ])
    expect(r.rules[0]).toMatchObject({ label: 1, p: 11, n: 0 })
    expect(r.rules).toHaveLength(10)
    expect(r.defaultLabel).toBe(0)
    const stop = r.trace[r.trace.length - 1]
    expect(stop).toMatchObject({ kind: 'stop', reason: 'rule limit reached', defaultLabel: 0 })
  })

  it('with a beam of 8 finds first-class women (Laplace 0.9660) first', () => {
    const r = sequentialCovering(t.rows, t.y, conditions, { measure: 'laplace', beam: 8 })
    expect(ruleText(r.rules[0].conditions, t)).toBe('class = 1st ∧ sex = female')
    expect(r.rules[0]).toMatchObject({ label: 1, p: 141, n: 4 })
    expect(r.rules[0].score).toBeCloseTo(142 / 147, 12)
  })

  it('gives the five-rule WRAcc list of the note', () => {
    const r = sequentialCovering(t.rows, t.y, conditions, { measure: 'wracc' })
    expect(r.rules.map((x) => `${ruleText(x.conditions, t)} -> ${x.label} ${x.p}/${x.n}`)).toEqual([
      'sex = male ∧ age = adult -> 0 1329/338',
      'class = 3rd -> 0 141/103',
      'class = 1st -> 1 146/4',
      'age = child -> 1 24/0',
      'class = crew -> 1 20/3',
    ])
    expect(r.defaultLabel).toBe(1)
    expect(r.trace[r.trace.length - 1]).toMatchObject({ reason: 'no rule beats the empty rule' })
  })

  it('in target mode removes only the covered positives', () => {
    const r = sequentialCovering(t.rows, t.y, conditions, { mode: 'target', target: 1 })
    expect(r.rules.every((x) => x.label === 1)).toBe(true)
    expect(r.rules.slice(0, 2).map((x) => `${ruleText(x.conditions, t)} ${x.p}/${x.n}`)).toEqual([
      'class = 1st ∧ sex = female 141/4',
      'class = 2nd ∧ age = child 24/0',
    ])
    const removes = r.trace.filter((s) => s.kind === 'remove')
    expect(removes.every((s, k) => s.kind === 'remove' && s.removed === r.rules[k].p)).toBe(true)
    expect(correct(applyCoveringRules(r, t.rows, 'vote'))).toBe(1735)
  })

  it('stops earlier with CN2’s significance test', () => {
    const r = sequentialCovering(t.rows, t.y, conditions, { significance: 3.84 })
    expect(r.rules).toHaveLength(8)
    expect(r.trace[r.trace.length - 1]).toMatchObject({ reason: 'no rule beats the empty rule' })
  })
})
