import { describe, expect, it } from 'vitest'
import {
  auroc,
  brierScore,
  metricRegistry,
  ordinalConcordanceIndex,
  rankedProbabilityScore,
} from 'aifn-compute/learning/metrics'
import { fixture } from '../../fixtures'

type Fx = {
  y: number[]
  score: number[]
  c_index: number
  probabilities: number[][]
  rps: number
  binary: { y: number[]; score: number[]; auroc: number; p: number[]; brier: number }
}
const F = fixture<Fx>('learning/metrics')

describe('ordinal C-index', () => {
  it("equals (Somers' D + 1)/2 from scipy, with tied scores", () => {
    expect(ordinalConcordanceIndex(F.y, F.score)).toBeCloseTo(F.c_index, 12)
  })
  it('is the AUROC for two classes (scikit-learn), and agrees with the compute auroc', () => {
    expect(ordinalConcordanceIndex(F.binary.y, F.binary.score)).toBeCloseTo(F.binary.auroc, 12)
    expect(ordinalConcordanceIndex(F.binary.y, F.binary.score)).toBeCloseTo(auroc(F.binary.y, F.binary.score), 12)
  })
  it('matches the O(n²) pair count, and follows a custom label order', () => {
    const y = ['lo', 'mid', 'hi', 'mid', 'lo', 'hi', 'hi']
    const s = [0.1, 0.5, 0.4, 0.5, 0.3, 0.9, 0.1]
    const order = ['lo', 'mid', 'hi']
    const r = y.map((v) => order.indexOf(v))
    let num = 0
    let den = 0
    for (let i = 0; i < y.length; i++)
      for (let j = 0; j < y.length; j++)
        if (r[i] < r[j]) {
          den++
          num += s[i] < s[j] ? 1 : s[i] === s[j] ? 0.5 : 0
        }
    expect(ordinalConcordanceIndex(y, s, { labels: order })).toBeCloseTo(num / den, 14)
    expect(() => ordinalConcordanceIndex([1, 1], [0, 1])).toThrow(/two classes/)
  })
})

describe('ranked probability score', () => {
  it('matches the direct cumulative computation', () => {
    expect(rankedProbabilityScore(F.y, F.probabilities)).toBeCloseTo(F.rps, 12)
  })
  it('is the binary Brier score for two classes (scikit-learn brier_score_loss)', () => {
    const P = F.binary.p.map((p) => [1 - p, p])
    expect(rankedProbabilityScore(F.binary.y, P)).toBeCloseTo(F.binary.brier, 12)
    expect(rankedProbabilityScore(F.binary.y, P)).toBeCloseTo(brierScore(F.binary.y, F.binary.p), 12)
  })
  it('charges more for mass far from the observed class; 0 for a certain correct forecast', () => {
    const near = rankedProbabilityScore([0], [[0.5, 0.5, 0]], { labels: [0, 1, 2] })
    const far = rankedProbabilityScore([0], [[0.5, 0, 0.5]], { labels: [0, 1, 2] })
    expect(far).toBeGreaterThan(near)
    expect(rankedProbabilityScore([2], [[0, 0, 1]], { labels: [0, 1, 2] })).toBe(0)
  })
  it('both are registered with their capabilities', () => {
    expect(metricRegistry.rankedProbabilityScore.info.capability).toBe('predictive')
    expect(metricRegistry.ordinalConcordanceIndex.info.capability).toBe('score')
  })
})
