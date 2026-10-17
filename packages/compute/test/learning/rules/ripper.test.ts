/**
 * RIPPER: the description-length pieces (S(4, 2, 1/2) = 4 bits, a perfect rule set's exceptions cost
 * log2(C + U + 1)), the Titanic worked example's description lengths (2259.9 bits for the empty set, 1982.8 for
 * "female and 1st"), and seeded runs on the Titanic table: the trace's phases, the rules and their accuracy.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  applyRipper,
  exceptionBits,
  nominalConditions,
  ripper,
  ripperDescriptionLength,
  ruleBits,
  ruleText,
  subsetBits,
  thresholdConditions,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
const conditions = nominalConditions([4, 2, 2])
const correct = (pred: ArrayLike<number>, y: readonly number[]) => Array.from(pred).filter((v, i) => v === y[i]).length

describe('description length', () => {
  it('S(t, k, p) and the rule bits', () => {
    expect(subsetBits(4, 2, 0.5)).toBe(4)
    expect(subsetBits(10, 0, 0.3)).toBeCloseTo(-10 * Math.log2(0.7), 12)
    expect(ruleBits(1, 8)).toBeCloseTo(0.5 * subsetBits(8, 1, 1 / 8), 12)
    expect(ruleBits(2, 8)).toBeCloseTo(3.7451124978365313, 12)
    expect(ruleBits(0, 8)).toBe(0)
  })

  it('a perfect rule set costs log2(C + U + 1) in exceptions', () => {
    expect(exceptionBits(10, 10, 0, 0)).toBeCloseTo(Math.log2(21), 12)
  })

  it('of the Titanic worked example: 2259.9 bits for no rule, 1982.8 for female and 1st', () => {
    const positive = t.y.map((v) => v === 1)
    expect(ripperDescriptionLength([], t.rows, positive, undefined, 8)).toBeCloseTo(2259.883, 3)
    const femaleFirst = [conditions[5], conditions[0]]
    expect(ripperDescriptionLength([femaleFirst], t.rows, positive, undefined, 8)).toBeCloseTo(1982.791, 3)
  })
})

describe('ripper on the Titanic table', () => {
  it('IREP* alone learns three rules for the survivors, then rejects a fourth', () => {
    const r = ripper(t.rows, t.y, conditions, stream(3), { optimisations: 0 })
    expect(r.rules.map((x) => ruleText(x.conditions, t))).toEqual([
      'sex = female ∧ class = 1st',
      'sex = female ∧ class = 2nd',
      'sex = female ∧ class = crew',
    ])
    expect(r.rules.every((x) => x.label === 1)).toBe(true)
    expect(r.defaultLabel).toBe(0)
    const phases = r.trace.map((s) => s.phase)
    expect(phases.filter((p) => p === 'accept')).toHaveLength(4)
    expect(phases.filter((p) => p === 'reject')).toHaveLength(1)
    expect(phases.slice(-3)).toEqual(['reject', 'delete', 'done'])
  })

  it('RIPPER2 settles on sex = female, 1708 of 2201 right', () => {
    const r = ripper(t.rows, t.y, conditions, stream(3), { optimisations: 2 })
    expect(r.rules.map((x) => `${ruleText(x.conditions, t)} -> ${x.label}`)).toEqual(['sex = female -> 1'])
    expect(correct(applyRipper(r, t.rows), t.y)).toBe(1708)
    expect(r.trace).toHaveLength(47)
    expect(r.trace.filter((s) => s.phase === 'optimise').length).toBeGreaterThan(0)
  })

  it('prunes by (p − n)/(p + n) on the pruning set, keeping the best prefix', () => {
    const r = ripper(t.rows, t.y, conditions, stream(3), { optimisations: 0 })
    for (const s of r.trace) {
      if (s.phase !== 'prune') continue
      for (const c of s.candidates)
        if (c.p + c.n === 0) expect(c.value).toBe(-Infinity)
        else expect(c.value).toBeCloseTo((c.p - c.n) / (c.p + c.n), 12)
      const best = Math.max(...s.candidates.map((c) => c.value))
      expect(s.candidates[s.kept].value).toBe(best)
      // The shorter prefix wins a tie: no later (shorter) candidate has the same value.
      expect(s.candidates.slice(s.kept + 1).every((c) => c.value < best)).toBe(true)
    }
  })
})

describe('ripper on a band', () => {
  it('learns a rule set that classifies a noiseless band exactly', () => {
    const rows = Array.from({ length: 60 }, (_, i) => [i % 10, Math.floor(i / 10)])
    const y = rows.map(([a]) => (a >= 3 && a <= 6 ? 1 : 0))
    const r = ripper(rows, y, thresholdConditions(rows, 10), stream(1))
    expect(correct(applyRipper(r, rows), y)).toBe(60)
  })
})
