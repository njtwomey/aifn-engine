/**
 * Rule quality measures on the Titanic worked example (P = 711 survivors, N = 1490 deaths): Laplace 142/147,
 * m-estimate, WRAcc two ways, FOIL gain 405.915 (on the engine's `foilGain`), the isometric invariants of coverage
 * space, the limits of the m-estimate, and the empty and universal rules.
 */
import { describe, expect, it } from 'vitest'
import { foilGain } from 'aifn-compute/logic/induction'
import {
  laplace,
  likelihoodRatio,
  mEstimate,
  ripperPruneValue,
  ruleAccuracy,
  ruleQuality,
  RULE_MEASURES,
} from 'aifn-compute/learning/rules'

const T = { P: 711, N: 1490 }

describe('rule quality on the Titanic worked example', () => {
  it('Laplace of first-class women is 142/147', () => {
    expect(laplace({ p: 141, n: 4, ...T })).toBeCloseTo(142 / 147, 12)
    expect(ruleQuality('laplace', { p: 141, n: 4, ...T })).toBeCloseTo(142 / 147, 12)
    expect(laplace({ p: 141, n: 4, ...T }, 3)).toBeCloseTo(142 / 148, 12)
  })

  it('m-estimate of first-class children with m = 10 is 0.5769', () => {
    const v = mEstimate({ p: 6, n: 0, ...T }, 10)
    expect(v).toBeCloseTo((6 + (10 * 711) / 2201) / 16, 12)
    expect(v).toBeCloseTo(0.5769, 4)
    expect(ruleQuality('m-estimate', { p: 6, n: 0, ...T }, { m: 10 })).toBe(v)
  })

  it('WRAcc of women is 0.0873, equal to PN/T² (TPR − FPR)', () => {
    const w = ruleQuality('wracc', { p: 344, n: 126, ...T })
    expect(w).toBeCloseTo((344 * 1490 - 126 * 711) / 2201 ** 2, 12)
    expect(w).toBeCloseTo(((711 * 1490) / 2201 ** 2) * (344 / 711 - 126 / 1490), 12)
    expect(w).toBeCloseTo(0.0873, 4)
  })

  it('FOIL gain of women against the empty rule is 405.915, the engine foilGain with t = p', () => {
    const g = ruleQuality('foil-gain', { p: 344, n: 126, ...T })
    expect(g).toBeCloseTo(405.915, 3)
    expect(g).toBe(foilGain(711, 1490, 344, 126, 344))
    // Against a parent rule, and the edge cases: no positive kept is 0 bits, nothing covered is −∞.
    expect(ruleQuality('foil-gain', { p: 141, n: 4, ...T }, { parent: { p: 344, n: 126 } })).toBe(
      foilGain(344, 126, 141, 4, 141),
    )
    expect(ruleQuality('foil-gain', { p: 0, n: 5, ...T })).toBe(0)
    expect(ruleQuality('foil-gain', { p: 0, n: 0, ...T })).toBe(-Infinity)
  })
})

describe('invariants', () => {
  it('precision is constant on lines through the origin, Laplace on lines through (−1, −1)', () => {
    const precision = (p: number, n: number) => ruleQuality('precision', { p, n, ...T })
    expect(precision(10, 5)).toBeCloseTo(precision(20, 10), 12)
    expect(laplace({ p: 9, n: 4, ...T })).toBeCloseTo(laplace({ p: 19, n: 9, ...T }), 12)
  })

  it('the m-estimate tends to the prior as m grows, and is precision at m = 0', () => {
    expect(mEstimate({ p: 5, n: 0, ...T }, 1e9)).toBeCloseTo(711 / 2201, 6)
    expect(mEstimate({ p: 5, n: 3, ...T }, 0)).toBe(5 / 8)
  })

  it('the empty and the universal rule', () => {
    expect(likelihoodRatio({ p: 0, n: 0, ...T })).toBe(0)
    expect(likelihoodRatio({ p: 711, n: 1490, ...T })).toBeCloseTo(0, 9)
    expect(ruleQuality('wracc', { p: 711, n: 1490, ...T })).toBeCloseTo(0, 15)
    expect(ruleQuality('coverage', { p: 0, n: 0, ...T })).toBe(0)
    expect(ruleQuality('coverage', { p: 711, n: 1490, ...T })).toBe(1)
    for (const m of ['precision', 'laplace', 'm-estimate', 'accuracy', 'wracc', 'ripper-prune'] as const)
      expect(ruleQuality(m, { p: 0, n: 0, ...T })).toBe(-Infinity)
  })

  it('rule accuracy orders rules as p − n, and the pruning value is (p − n)/(p + n)', () => {
    expect(ruleAccuracy({ p: 344, n: 126, ...T })).toBeCloseTo((344 + 1490 - 126) / 2201, 12)
    expect(ripperPruneValue({ p: 23, n: 1, P: 0, N: 0 })).toBeCloseTo(22 / 24, 12)
  })

  it('every listed measure is known, with a TeX formula', () => {
    expect(RULE_MEASURES.map((m) => m.key)).toEqual([
      'coverage',
      'precision',
      'laplace',
      'm-estimate',
      'accuracy',
      'wracc',
      'foil-gain',
      'likelihood-ratio',
      'ripper-prune',
    ])
    for (const m of RULE_MEASURES) expect(Number.isFinite(ruleQuality(m.key, { p: 50, n: 20, ...T }))).toBe(true)
  })
})
