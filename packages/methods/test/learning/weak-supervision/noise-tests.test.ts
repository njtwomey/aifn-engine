/**
 * The anchor-point tests for class-conditional label noise (Poyiadzi et al. 2022; Yang et al. 2024) against statsmodels
 * (`fixtures/learning/weak-supervision.json`: Logit's MLE and inverse Fisher information; a Binomial GLM with kernel
 * variance weights and its HC0 sandwich for the local fits), and by simulation on the papers' data: the size under the
 * null (uniform noise) is near the level, and the power grows with the gap between the noise rates and with the number
 * of anchors; the analytic power equals the level when α = β.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { classConditionalNoise, noiseLayoutAnchors, noiseLayoutPosterior } from 'aifn-methods/data/synthetic'
import {
  classConditionalNoiseTest,
  localLogistic,
  localLogisticCovariance,
  noiseTestPower,
  noiseTestSimulation,
  type NoiseTestCell,
} from 'aifn-methods/learning/weak-supervision'
import { fixture } from '../../fixtures'

type Local = { bandwidth: number; degree: 1 | 2; at: number[]; coefficients: number[]; covariance: number[][] }
const F = fixture<{
  noise: {
    x: number[][]
    y: number[]
    anchors: number[][]
    theta: number[]
    estimates: number[]
    variance: number
    z: number
    local: Local[]
  }
}>('learning/weak-supervision').noise

describe('against statsmodels', () => {
  it('the parametric test: Logit MLE, η̂ at the anchors, v and z', () => {
    const r = classConditionalNoiseTest(fromRows(F.x), F.y, fromRows(F.anchors))
    r.anchorEstimates.forEach((e, i) => expect(e).toBeCloseTo(F.estimates[i], 6))
    expect(r.variance).toBeCloseTo(F.variance, 8)
    expect(r.statistic).toBeCloseTo(F.z, 5)
  })

  it('local logistic fits and their HC0 sandwich', () => {
    for (const c of F.local) {
      const fit = localLogistic(fromRows(F.x), F.y, c.at, { bandwidth: c.bandwidth, degree: c.degree })
      Array.from(fit.coefficients).forEach((b, j) => expect(b).toBeCloseTo(c.coefficients[j], 5))
      const q = fit.basisSize
      for (let j = 0; j < q; j++)
        for (let k = 0; k < q; k++) expect(fit.covariance[j * q + k]).toBeCloseTo(c.covariance[j][k], 5)
      // The cross-covariance of a fit with itself is its own variance.
      expect(localLogisticCovariance(fit, fit)).toBeCloseTo(fit.logitVariance, 12)
    }
  })
})

describe('the noise layouts', () => {
  it('anchors have posterior ½ (strict) or within δ (relaxed)', () => {
    for (const layout of ['gaussians', 'xor', 'asymmetric-xor'] as const) {
      const eta = noiseLayoutPosterior(layout)
      for (const a of noiseLayoutAnchors(stream(layout), layout, 8)) expect(eta(a)).toBeCloseTo(0.5, 8)
      for (const a of noiseLayoutAnchors(stream(layout), layout, 8, 0.05))
        expect(Math.abs(eta(a) - 0.5)).toBeLessThanOrEqual(0.05)
    }
  })

  it('the truth’s posterior is the noisy posterior (1 − α − β)η + β', () => {
    const d = classConditionalNoise(stream(2), { n: 10, alpha: 0.1, beta: 0.3 })
    const truth = d.meta.truth as unknown as { posterior: (x: Tensor) => Tensor; cleanPosterior: (x: Tensor) => Tensor }
    const pts = fromRows([
      [0.3, -0.2],
      [1, 1],
      [-2, 0.5],
    ])
    const noisy = toFlat(truth.posterior(pts))
    const clean = toFlat(truth.cleanPosterior(pts))
    for (let i = 0; i < 3; i++) expect(noisy[2 * i + 1]).toBeCloseTo(0.6 * clean[2 * i + 1] + 0.3, 10)
  })
})

function cell(label: string, alpha: number, beta: number, R: number, k: number, n = 500): NoiseTestCell {
  const datasets = Array.from({ length: R }, (_, r) => {
    const d = classConditionalNoise(child(stream(label), r), { n, alpha, beta, truth: false })
    return { x: d.x, y: d.y }
  })
  const anchors = Array.from({ length: R }, (_, r) =>
    fromRows(noiseLayoutAnchors(child(stream(`${label}/a`), r), 'gaussians', k)),
  )
  return { label, datasets, anchors }
}

function simulate(cells: NoiseTestCell[], options = {}) {
  const it = noiseTestSimulation(cells, options)
  let s = it.next()
  while (!s.done) s = it.next()
  return s.value
}

describe('by simulation', () => {
  it('the parametric test holds its size under uniform noise and has power under class-conditional noise', () => {
    const s = simulate([cell('null', 0.1, 0.1, 200, 4), cell('alt', 0, 0.2, 100, 4)])
    expect(s.rejected05[0]).toBeGreaterThan(0.01)
    expect(s.rejected05[0]).toBeLessThan(0.1)
    expect(s.rejected05[1]).toBeGreaterThan(0.5)
  })

  it('the local test holds its size and detects the noise', () => {
    const s = simulate([cell('local-null', 0.1, 0.1, 100, 2), cell('local-alt', 0, 0.2, 60, 2)], {
      model: 'local',
      bandwidth: 1,
    })
    expect(s.rejected05[0]).toBeLessThan(0.12)
    // The nonparametric test pays for its flexibility in power (Yang et al. 2024).
    expect(s.rejected05[1]).toBeGreaterThan(s.rejected05[0] + 0.1)
  })
})

describe('power (Prop. 3.1)', () => {
  it('is the level when α = β and grows with |β − α| and with the number of anchors', () => {
    expect(noiseTestPower({ variance: 0.001, alpha: 0.1, beta: 0.1, level: 0.05 })).toBeCloseTo(0.05, 10)
    const p = [0.02, 0.05, 0.1, 0.2].map((b) => noiseTestPower({ variance: 0.001, alpha: 0, beta: b }))
    for (let i = 1; i < p.length; i++) expect(p[i]).toBeGreaterThan(p[i - 1])
    const byK = [1, 4, 16].map((k) => noiseTestPower({ variance: 0.002 / k, alpha: 0, beta: 0.05 }))
    for (let i = 1; i < byK.length; i++) expect(byK[i]).toBeGreaterThan(byK[i - 1])
  })
})
