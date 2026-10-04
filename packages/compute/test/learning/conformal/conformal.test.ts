/**
 * Split conformal prediction: hand cases for the quantile and the scores, and the coverage law by simulation: over
 * many seeded draws of calibration and test data, the mean coverage lies in [1 − α, 1 − α + 1/(n + 1)] up to Monte
 * Carlo error, for every score, and within every group for Mondrian quantiles.
 */
import { describe, expect, it } from 'vitest'
import { categorical, child, normal, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  classificationScores,
  conformalClassification,
  conformalisedQuantileRegression,
  conformalQuantile,
  intervalCoverage,
  mondrianQuantiles,
  setCoverage,
  splitConformalRegression,
  type ClassificationScore,
} from 'aifn-compute/learning/conformal'

const flat = (t: Tensor) => Array.from(toFlat(t))

describe('hand cases', () => {
  it('the conformal quantile is the ⌈(n + 1)(1 − α)⌉-th smallest score, or ∞', () => {
    const s = [5, 1, 9, 3, 7, 2, 8, 4, 6]
    expect(conformalQuantile(s, 0.2)).toBe(8)
    expect(conformalQuantile(s, 0.5)).toBe(5)
    expect(conformalQuantile(s, 0.05)).toBe(Infinity)
  })

  it('the rank is not pushed up by rounding in (n + 1)(1 − α)', () => {
    // (1 − 0.7) · 10 = 3.0000000000000004 in floating point; the rank is 3, not 4.
    const s = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect(conformalQuantile(s, 0.7)).toBe(3)
    expect(conformalQuantile(s, 0.9)).toBe(1)
  })

  it('LAC, APS and RAPS scores', () => {
    const p = [[0.6, 0.3, 0.1]]
    expect(flat(classificationScores(p, [1], { score: 'lac' }))[0]).toBeCloseTo(0.7, 12)
    expect(flat(classificationScores(p, [1], { score: 'aps' }))[0]).toBeCloseTo(0.9, 12)
    expect(flat(classificationScores(p, [2], { score: 'raps', lambda: 0.5, kReg: 1 }))[0]).toBeCloseTo(1 + 0.5 * 2, 12)
  })

  it('sets include every class at or under the quantile; coverage and sizes', () => {
    const cal = {
      probabilities: [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
      ],
      labels: [0, 0, 1, 1],
    }
    // LAC scores 0.1, 0.2, 0.3, 0.6; α = 0.4 → rank ⌈5 · 0.6⌉ = 3 → q̂ = 0.3: classes with p ≥ 0.7.
    const out = conformalClassification(
      cal,
      [
        [0.75, 0.25],
        [0.5, 0.5],
      ],
      0.4,
    )
    expect(out.quantile).toBeCloseTo(0.3, 12)
    expect(toRows(out.sets)).toEqual([
      [1, 0],
      [0, 0],
    ])
    const c = setCoverage(out.sets, [0, 1])
    expect(c.coverage).toBe(0.5)
    expect(c.meanSize).toBe(0.5)
    const iv = intervalCoverage([0, 0], [1, 2], [0.5, 3])
    expect(iv.coverage).toBe(0.5)
    expect(iv.meanSize).toBe(1.5)
  })
})

/** Mean coverage over R seeded experiments must lie in [1 − α, 1 − α + 1/(n + 1)], widened by 4 Monte Carlo SEs. */
function expectCoverage(covered: number[], alpha: number, n: number) {
  const R = covered.length
  const mean = covered.reduce((a, b) => a + b, 0) / R
  const se = Math.sqrt((mean * (1 - mean)) / R)
  expect(mean).toBeGreaterThan(1 - alpha - 4 * se)
  expect(mean).toBeLessThan(1 - alpha + 1 / (n + 1) + 4 * se)
}

describe('coverage laws by simulation', () => {
  const alpha = 0.1
  const n = 40
  const R = 1500

  /** Heteroscedastic regression: y = x + (0.2 + x)ε, x ~ U(0, 1); the model predicts x and a too-narrow band. */
  const regression = (s: Stream, m: number) => {
    const x = Array.from({ length: m }, () => uniform(s))
    const y = x.map((v) => v + (0.2 + v) * normal(s))
    return { y, f: x, lo: x.map((v) => v - 0.8 * (0.2 + v)), hi: x.map((v) => v + 0.8 * (0.2 + v)) }
  }

  it('split conformal regression and CQR cover at 1 − α', () => {
    const split: number[] = []
    const cqr: number[] = []
    for (let r = 0; r < R; r++) {
      const s = child(stream('conformal-regression'), 'run', r)
      const cal = regression(s, n)
      const test = regression(s, 1)
      const a = splitConformalRegression({ targets: cal.y, predictions: cal.f }, test.f, alpha)
      split.push(intervalCoverage(a.lower, a.upper, test.y).coverage)
      const b = conformalisedQuantileRegression(
        { targets: cal.y, lower: cal.lo, upper: cal.hi },
        { lower: test.lo, upper: test.hi },
        alpha,
      )
      cqr.push(intervalCoverage(b.lower, b.upper, test.y).coverage)
    }
    expectCoverage(split, alpha, n)
    expectCoverage(cqr, alpha, n)
  })

  /** A 4-class problem with overconfident probabilities. */
  const classification = (s: Stream, m: number) => {
    const probs: number[][] = []
    const labels: number[] = []
    for (let i = 0; i < m; i++) {
      const z = [0, 1, 2, 3].map(() => normal(s))
      const e = z.map((v) => Math.exp(v - Math.max(...z)))
      const t = e.reduce((a, b) => a + b, 0)
      labels.push(
        categorical(
          s,
          e.map((v) => v / t),
        ),
      )
      const sharp = z.map((v) => Math.exp(3 * (v - Math.max(...z))))
      const st = sharp.reduce((a, b) => a + b, 0)
      probs.push(sharp.map((v) => v / st))
    }
    return { probs, labels }
  }

  it.each(['lac', 'aps', 'raps'] as ClassificationScore[])('%s sets cover at 1 − α', (score) => {
    const covered: number[] = []
    for (let r = 0; r < R; r++) {
      const s = child(stream('conformal-classification'), score, r)
      const cal = classification(s, n)
      const test = classification(s, 1)
      const out = conformalClassification({ probabilities: cal.probs, labels: cal.labels }, test.probs, alpha, {
        score,
        stream: child(s, 'u'),
      })
      covered.push(setCoverage(out.sets, test.labels).coverage)
    }
    expectCoverage(covered, alpha, n)
  })

  it('Mondrian quantiles cover within each group; one pooled quantile does not', () => {
    // Group 0 has noise sd 0.2, group 1 sd 1: the pooled quantile over-covers group 0 and under-covers group 1.
    const pooled = [[], []] as number[][]
    const mondrian = [[], []] as number[][]
    for (let r = 0; r < R; r++) {
      const s = child(stream('mondrian'), 'run', r)
      const scores: number[] = []
      const groups: number[] = []
      for (let i = 0; i < 2 * n; i++) {
        const g = i % 2
        groups.push(g)
        scores.push(Math.abs((g === 0 ? 0.2 : 1) * normal(s)))
      }
      const qg = flat(mondrianQuantiles(scores, groups, alpha))
      const q = conformalQuantile(scores, alpha)
      for (const g of [0, 1]) {
        const test = Math.abs((g === 0 ? 0.2 : 1) * normal(s))
        mondrian[g].push(test <= qg[g] ? 1 : 0)
        pooled[g].push(test <= q ? 1 : 0)
      }
    }
    expectCoverage(mondrian[0], alpha, n)
    expectCoverage(mondrian[1], alpha, n)
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
    expect(mean(pooled[1])).toBeLessThan(1 - alpha - 0.05)
    expect(mean(pooled[0])).toBeGreaterThan(0.97)
  })
})
