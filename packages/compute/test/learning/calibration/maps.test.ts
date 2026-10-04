/**
 * Calibration maps against scikit-learn and scipy (`fixtures/learning/calibration.json`: Platt's sigmoid, temperature
 * by a bounded scalar search, beta and Dirichlet as unpenalised logistic regressions) and their laws: temperature
 * keeps the predicted class, every map lowers the held-out log loss of an overconfident model, histogram binning
 * returns each bin's frequency, the isotonic map is monotone.
 */
import { describe, expect, it } from 'vitest'
import { categorical, child, normal, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  betaCalibration,
  dirichletCalibration,
  histogramBinning,
  isotonicCalibration,
  plattScaling,
  temperatureScaling,
  topLabelConfidence,
} from 'aifn-compute/learning/calibration'
import { fixture } from '../../fixtures'

type Platt = { scores: number[]; labels: number[]; A: number; B: number }
type Temp = { logits: number[][]; labels: number[]; temperature: number }
type Beta = { scores: number[]; labels: number[]; a: number; b: number; c: number }
type Dir = { probabilities: number[][]; labels: number[]; calibrated: number[][] }
const F = fixture<{
  plattScaling: Platt[]
  temperatureScaling: Temp[]
  betaCalibration: Beta[]
  dirichletCalibration: Dir[]
}>('learning/calibration')
const flat = (t: Tensor) => Array.from(toFlat(t))

describe('against scikit-learn and scipy', () => {
  it.each(F.plattScaling.map((c, i) => [i, c] as const))('Platt scaling (case %i)', (_, c) => {
    const p = plattScaling(tensor(c.scores), tensor(c.labels))
    expect(p.A).toBeCloseTo(c.A, 4)
    expect(p.B).toBeCloseTo(c.B, 4)
  })

  it.each(F.temperatureScaling.map((c, i) => [i, c] as const))('temperature scaling (case %i)', (_, c) => {
    const t = temperatureScaling(c.logits, c.labels)
    expect(t.temperature).toBeCloseTo(c.temperature, 5)
    expect(t.logLossAfter).toBeLessThanOrEqual(t.logLossBefore)
  })

  it.each(F.betaCalibration.map((c, i) => [i, c] as const))('beta calibration (case %i)', (_, c) => {
    const b = betaCalibration(c.scores, c.labels)
    // The reference drops a feature whose coefficient comes out negative and refits, as the paper recommends.
    expect(b.a).toBeCloseTo(c.a, 3)
    expect(b.b).toBeCloseTo(c.b, 3)
    expect(b.c).toBeCloseTo(c.c, 3)
  })

  it.each(F.dirichletCalibration.map((c, i) => [i, c] as const))(
    'Dirichlet calibration, unpenalised (case %i)',
    (_, c) => {
      const d = dirichletCalibration(c.probabilities, c.labels, { lambda: 0, mu: 0, maxSteps: 2000 })
      const got = toRows(d.apply(c.probabilities)) as number[][]
      got.forEach((row, i) => row.forEach((v, k) => expect(v).toBeCloseTo(c.calibrated[i][k], 4)))
    },
  )
})

describe('laws', () => {
  // An overconfident three-class model: logits 3 × the true ones.
  const make = (key: string, n: number) => {
    const s = stream(key)
    const logits: number[][] = []
    const labels: number[] = []
    for (let i = 0; i < n; i++) {
      const z = [normal(s), normal(s), normal(s)].map((v) => 1.5 * v)
      const e = z.map(Math.exp)
      const tot = e.reduce((a, b) => a + b, 0)
      labels.push(
        categorical(
          child(s, 'y', i),
          e.map((v) => v / tot),
        ),
      )
      logits.push(z.map((v) => 3 * v))
    }
    return { logits, labels }
  }
  const train = make('cal-train', 1500)
  const test = make('cal-test', 1500)
  const nll = (P: number[][], y: number[]) =>
    -y.reduce((a, k, i) => a + Math.log(Math.max(P[i][k], 1e-300)), 0) / y.length
  const softmaxRows = (Z: number[][]) =>
    Z.map((z) => {
      const e = z.map((v) => Math.exp(v - Math.max(...z)))
      const t = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / t)
    })

  it('temperature finds T ≈ 3, keeps the predicted class and lowers held-out log loss', () => {
    const t = temperatureScaling(train.logits, train.labels)
    expect(t.temperature).toBeGreaterThan(2.5)
    expect(t.temperature).toBeLessThan(3.6)
    const before = softmaxRows(test.logits)
    const after = toRows(t.apply(test.logits)) as number[][]
    after.forEach((row, i) => expect(row.indexOf(Math.max(...row))).toBe(before[i].indexOf(Math.max(...before[i]))))
    expect(nll(after, test.labels)).toBeLessThan(nll(before, test.labels) - 0.1)
  })

  it('Dirichlet calibration (with ODIR) lowers held-out log loss as much as temperature', () => {
    const P = softmaxRows(train.logits)
    const d = dirichletCalibration(P, train.labels)
    const t = temperatureScaling(train.logits, train.labels)
    const after = toRows(d.apply(softmaxRows(test.logits))) as number[][]
    expect(nll(after, test.labels)).toBeLessThan(nll(toRows(t.apply(test.logits)) as number[][], test.labels) + 0.01)
    expect(d.logLossAfter).toBeLessThan(d.logLossBefore)
  })

  it('histogram binning returns each bin’s fraction of positives', () => {
    const scores = [0.05, 0.15, 0.12, 0.55, 0.58, 0.95]
    const labels = [0, 1, 0, 1, 1, 1]
    const h = histogramBinning(scores, labels, { bins: 4 })
    expect(flat(h.values)).toEqual([1 / 3, 0.375, 1, 1])
    expect(flat(h.apply([0.1, 0.3, 0.6, 1]))).toEqual([1 / 3, 0.375, 1, 1])
  })

  it('histogram binning maps a training score to the bin it was counted in', () => {
    // 0.8999999999999999 · 10 rounds to 9, so the diagram counts it in bin 9, though it lies below the edge 0.9.
    const s = 0.8999999999999999
    const h = histogramBinning([s, 0.85, 0.05], [1, 0, 0], { bins: 10 })
    expect(flat(h.values)[9]).toBe(1)
    expect(flat(h.apply([s]))).toEqual([1])
  })

  it('isotonic calibration is a monotone step map through the PAV fit', () => {
    const scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    const labels = [0, 1, 0, 0, 1, 1]
    const m = isotonicCalibration(scores, labels)
    expect(flat(m.fit)).toEqual([0, 1 / 3, 1 / 3, 1 / 3, 1, 1])
    const v = flat(m.apply([0, 0.15, 0.25, 0.45, 0.55, 2]))
    expect(v).toEqual([0, 0, 1 / 3, 1 / 3, 1, 1])
  })

  it('beta calibration contains the identity on calibrated scores', () => {
    const s = stream('beta-identity')
    const scores: number[] = []
    const labels: number[] = []
    for (let i = 0; i < 20000; i++) {
      const p = 1 / (1 + Math.exp(-2 * normal(s)))
      scores.push(p)
      labels.push(categorical(s, [1 - p, p]))
    }
    const b = betaCalibration(scores, labels)
    expect(Math.abs(b.a - 1)).toBeLessThan(0.1)
    expect(Math.abs(b.b - 1)).toBeLessThan(0.1)
    expect(Math.abs(b.c)).toBeLessThan(0.1)
  })

  it('top-label confidence', () => {
    const r = topLabelConfidence(
      [
        [0.2, 0.7, 0.1],
        [0.5, 0.3, 0.2],
      ],
      [1, 2],
    )
    expect(flat(r.confidence)).toEqual([0.7, 0.5])
    expect(flat(r.correct)).toEqual([1, 0])
    expect(flat(r.predicted)).toEqual([1, 0])
  })
})
