/**
 * Bayesian online changepoint detection from `aifn-compute/inference/filtering` on the worked example of the note
 * `bayesian-online-changepoint-detection`: σ = 1, μ ~ N(0, 2²), constant hazard H = 0.2, x = (0.1, −0.4, 0.3, 3.2,
 * 2.8, 3.1).
 */
import { describe, expect, it } from 'vitest'
import {
  betaBernoulli,
  bocpd,
  bocpdForecast,
  bocpdInit,
  bocpdPredictiveDensity,
  bocpdUpdate,
  constantHazard,
  detectChangepoints,
  laggedObservations,
  mapChangepoints,
  normalGamma,
  normalKnownVariance,
  poissonGamma,
  regressionNormalGamma,
  runLengthMass,
  runLengthRow,
  type ConjugatePredictive,
  type RunStats,
} from 'aifn-compute/inference/filtering'
import { exp, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const x = [0.1, -0.4, 0.3, 3.2, 2.8, 3.1]
const H = 0.2
const model = normalKnownVariance({ mean: 0, priorSd: 2, sd: 1 })

/** The recursion written out directly: a run of n values with sum S has mean posterior N(S/p, 1/p), p = 1/4 + n. */
function reference(): number[][] {
  const pdf = (v: number, m: number, s2: number) => Math.exp(-((v - m) ** 2) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2)
  let post = [1]
  let sums = [0]
  const rows: number[][] = []
  for (const v of x) {
    const pred = post.map((_, r) => {
      const p = 1 / 4 + r
      return pdf(v, sums[r] / p, 1 / p + 1)
    })
    const grow = post.map((w, r) => w * pred[r] * (1 - H))
    const reset = post.reduce((acc, w, r) => acc + w * pred[r] * H, 0)
    const joint = [reset, ...grow]
    const z = joint.reduce((a, b) => a + b, 0)
    post = joint.map((j) => j / z)
    sums = [0, ...sums.map((s) => s + v)]
    rows.push(post)
  }
  return rows
}

// The note's table, rounded to three decimals. Its entry at t = 5, r = 2 reads 0.597; the exact value is 0.59647, so
// 0.596 here (a rounding slip in the note).
const table = [
  [0.2, 0.8],
  [0.2, 0.109, 0.691],
  [0.2, 0.101, 0.083, 0.616],
  [0.2, 0.501, 0.103, 0.028, 0.167],
  [0.2, 0.066, 0.596, 0.082, 0.011, 0.045],
  [0.2, 0.046, 0.054, 0.627, 0.057, 0.004, 0.011],
]

describe('the worked example', () => {
  const tr = trace(bocpd(model, x, { hazard: H }), undefined, 10, { keep: 'all' })
  const rows = tr.steps.slice(1).map((s) => Array.from(runLengthRow(s, s.t + 1)))

  it('step 1: the prior predictive density at x₁ is 0.1782 and the posterior is (0.2, 0.8)', () => {
    const d = Math.exp(-(0.1 ** 2) / (2 * 5)) / Math.sqrt(2 * Math.PI * 5)
    expect(d).toBeCloseTo(0.1782, 4)
    const s1 = bocpdUpdate(model, bocpdInit(model), 0.1, { hazard: H })
    expect(Math.exp(s1.logPredictive)).toBeCloseTo(d, 12)
    expect(Array.from(runLengthRow(s1, 2))).toEqual([expect.closeTo(0.2, 12), expect.closeTo(0.8, 12)])
  })

  it('the run-length posteriors equal the recursion written out, and the table to its three decimals', () => {
    expect(tr.meta.stopped).toBe('done')
    expect(tr.meta.steps).toBe(6)
    const ref = reference()
    rows.forEach((row, t) => {
      expect(row.length).toBe(t + 2)
      row.forEach((p, r) => expect(p).toBeCloseTo(ref[t][r], 12))
      row.forEach((p, r) => expect(Math.abs(p - table[t][r])).toBeLessThanOrEqual(5e-4 + 1e-12))
    })
  })

  it('the worked example at t = 5: P(r₅ = 2) = 0.59647', () => {
    expect(rows[4][2]).toBeCloseTo(0.59647, 5)
  })

  it('P(r = 0) = H at every step; the MAP run length drops to 1 at x₄ and then climbs', () => {
    for (const row of rows) expect(row[0]).toBeCloseTo(H, 12)
    expect(tr.steps.slice(1).map((s) => s.map)).toEqual([1, 2, 3, 1, 2, 3])
  })

  it('the evidence is the product of the one-step predictives', () => {
    const logs = tr.steps.slice(1).map((s) => s.logPredictive)
    expect(tr.final.logEvidence).toBeCloseTo(
      logs.reduce((a, b) => a + b, 0),
      12,
    )
    expect(toFlat(tr.final.logPosterior).length).toBe(7)
  })

  it('bocpd satisfies the Algorithm protocol', () => {
    checkProtocol(bocpd(model, x, { hazard: H }), undefined, { steps: 6, record: { map: (s) => s.map } })
  })
})

describe('every conjugate model', () => {
  const series = [0.2, -0.1, 0.4, 0.1, 3.1, 2.7, 3.3, 2.9, 3.0, -1.2, -0.8, -1.1]
  const counts = [1, 0, 2, 1, 0, 6, 5, 7, 4, 6, 1, 0]
  const flips = [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
  const cases: [string, ConjugatePredictive<never, RunStats>, readonly unknown[]][] = [
    ['normal, known variance', normalKnownVariance({ priorSd: 3 }) as never, series],
    ['normal–gamma', normalGamma({ kappa: 0.1 }) as never, series],
    ['Poisson–gamma', poissonGamma({ shape: 2, rate: 1 }) as never, counts],
    ['beta–Bernoulli', betaBernoulli() as never, flips],
    ['AR(1) regression', regressionNormalGamma({ dimension: 2 }) as never, laggedObservations(series, 1)],
  ]
  it.each(cases)('%s: run-length rows sum to 1 with P(r = 0) = H', (_, m, data) => {
    const hazard = 0.1
    let state = bocpdInit(m)
    for (const v of data) {
      state = bocpdUpdate(m, state, v as never, { hazard })
      const row = Array.from(runLengthRow(state, state.t + 1))
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      expect(row[0]).toBeCloseTo(hazard, 12)
      expect(runLengthMass(state, 0)).toBeCloseTo(1, 12)
      expect(Number.isFinite(state.logPredictive)).toBe(true)
    }
    checkProtocol(bocpd(m, data as never[], { hazard }), undefined, { steps: data.length })
  })

  it('the Poisson–gamma predictive is a probability mass; forecasts are its moments', () => {
    const m = poissonGamma({ shape: 2, rate: 1 })
    let state = bocpdInit(m)
    for (const v of counts.slice(0, 6)) state = bocpdUpdate(m, state, v, { hazard: 0.1 })
    const ks = Array.from({ length: 200 }, (_, k) => k)
    const pmf = bocpdPredictiveDensity(m, state, ks)
    expect(pmf.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8)
    const f = bocpdForecast(m, state)
    const mean = pmf.reduce((a, p, k) => a + p * k, 0)
    expect(f.mean).toBeCloseTo(mean, 8)
    expect(f.variance).toBeCloseTo(
      pmf.reduce((a, p, k) => a + p * (k - mean) ** 2, 0),
      6,
    )
  })
})

describe('detection over a series', () => {
  it('finds a clear mean shift, and its evidence matches the recursion', () => {
    const data = [
      ...Array(20)
        .fill(0)
        .map((_, i) => 0.3 * Math.sin(i)),
      ...Array(20)
        .fill(0)
        .map((_, i) => 5 + 0.3 * Math.cos(i)),
    ]
    const m = normalKnownVariance({ priorSd: 5, sd: 0.5 })
    const d = detectChangepoints(m, data, { hazard: constantHazard(50) })
    expect(d.changepoints).toEqual([20])
    expect(d.posterior.shape).toEqual([40, 41])
    const rows = d.posterior.shape[0]
    for (let t = 0; t < rows; t++) {
      const row = toFlat(d.posterior).slice(t * 41, (t + 1) * 41)
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    }
    const tr = trace(bocpd(m, data, { hazard: 1 / 50 }), undefined, 100)
    expect(d.logEvidence).toBeCloseTo(tr.final.logEvidence, 10)
  })

  it('pruning keeps at most maxRuns run lengths and reports the discarded mass', () => {
    const m = normalGamma()
    const data = Array.from({ length: 60 }, (_, i) => Math.sin(i / 3))
    const tr = trace(bocpd(m, data, { hazard: 0.05, maxRuns: 8 }), undefined, 100)
    expect(tr.final.runLengths.shape[0]).toBeLessThanOrEqual(8)
    expect(toFlat(exp(tr.final.logPosterior)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('mapChangepoints backtracks the MAP run lengths', () => {
    expect(mapChangepoints([1, 2, 3, 1, 2, 3])).toEqual([3])
    expect(mapChangepoints([1, 2, 1, 4, 5])).toEqual([])
    expect(() => constantHazard(0.5)).toThrow()
  })
})
