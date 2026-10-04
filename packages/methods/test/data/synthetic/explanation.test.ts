/**
 * Test beds for explanations: shapes and labels; the feasibility task leaves the gap between its blobs empty; the
 * correlated-effects features have about the requested correlation and the noise-free target x₁ + x₂²; planted motifs
 * sit where `t` says; concept images follow their rule, and concept examples always hold their concept.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { correlation } from 'aifn-compute/probability/stats'
import {
  conceptExamples,
  conceptImages,
  correlatedEffects,
  feasibilityTask,
  plantedMask,
  plantedPatterns,
} from 'aifn-methods/data/synthetic'

describe('explanation test beds', () => {
  it('feasibility task leaves the gap empty', () => {
    const d = feasibilityTask(stream('feas'), { n: 600 })
    const x = toFlat(d.x)
    let inGap = 0
    for (let i = 0; i < 600; i++) if (Math.abs(x[2 * i]) < 0.4 && Math.abs(x[2 * i + 1] - 0.7) < 0.3) inGap++
    expect(inGap).toBeLessThan(3)
    expect(new Set(toFlat(d.y!))).toEqual(new Set([0, 1]))
  })
  it('correlated effects', () => {
    const d = correlatedEffects(stream('corr'), { n: 2000, correlation: 0.9 })
    const x = toFlat(d.x)
    const a = Array.from({ length: 2000 }, (_, i) => x[2 * i])
    const b = Array.from({ length: 2000 }, (_, i) => x[2 * i + 1])
    expect(correlation(a, b)).toBeCloseTo(0.9, 1)
    const f = toFlat(d.f!)
    expect(f[3]).toBeCloseTo(a[3] + b[3] ** 2, 12)
  })
  for (const kind of ['image', 'sequence'] as const)
    it(`planted ${kind} motifs sit at t`, () => {
      const d = plantedPatterns(stream(`planted-${kind}`), { n: 50, kind, noise: 0 })
      const x = toFlat(d.x)
      const t = toFlat(d.t!)
      const y = toFlat(d.y!)
      const dim = kind === 'image' ? 64 : 32
      for (let i = 0; i < 50; i++) {
        expect(t[i] >= 0).toBe(y[i] === 1)
        const mask = plantedMask(kind, t[i])
        for (let p = 0; p < dim; p++) expect(x[i * dim + p]).toBeCloseTo(1.2 * mask[p], 12)
      }
    })
  it('concept images follow their rule', () => {
    const d = conceptImages(stream('concepts'), { n: 200, rule: 'dot', noise: 0 })
    const x = toFlat(d.x)
    const y = toFlat(d.y!)
    for (let i = 0; i < 200; i++) expect(x[i * 64 + 7] > 1 ? 1 : 0).toBe(y[i])
    const e = toFlat(conceptExamples(stream('ex'), 'bar', { n: 3, noise: 0 }).x)
    for (let i = 0; i < 3; i++) for (const r of [0, 2, 3, 5, 6]) expect(e[i * 64 + r * 8 + 3]).toBeCloseTo(0.9, 12)
  })
})
