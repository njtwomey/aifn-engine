/**
 * EP over models of the model language: exact on linear-Gaussian trees with Gaussian evidence (against dense Gaussian
 * conditioning), truncated-normal moments for one interval factor, the complement of a one-sided interval, and the
 * models it refuses.
 */
import { describe, expect, it } from 'vitest'
import { run } from 'aifn-compute/foundation/trace'
import { dist, logJoint, model } from 'aifn-compute/inference/model'
import {
  compileGaussianModel,
  intervalTilted,
  modelExpectationPropagation,
} from 'aifn-compute/inference/expectation-propagation'

const marginal = (
  s: { keys: string[]; means: { data: ArrayLike<number> }; variances: { data: ArrayLike<number> } },
  key: string,
) => {
  const k = s.keys.indexOf(key)
  return { mean: s.means.data[k], variance: s.variances.data[k] }
}

describe('modelExpectationPropagation', () => {
  it('is exact on a linear-Gaussian chain with Gaussian observations', () => {
    // a ~ N(1, 2²), b ~ N(a − 0.5, 0.7²), y₁ ~ N(a, 1) = 2, y₂ ~ N(3b + 1, 0.5²) = 4.
    const m = model('chain', (mm) => {
      const a = mm.variable('a', dist.Normal(1, 2))
      const b = mm.variable('b', dist.Normal(mm.deterministic('aShift', 'difference', [a, 0.5]), 0.7))
      mm.observed('y1', dist.Normal(a, 1))
      mm.observed(
        'y2',
        dist.Normal(mm.deterministic('lin', 'sum', [mm.deterministic('b3', 'product', [3, b]), 1]), 0.5),
      )
    })
    const s = run(modelExpectationPropagation(m, { data: { y1: 2, y2: 4 } }, { tolerance: 1e-14 }), undefined, 200)
    expect(s.converged).toBe(true)
    // Dense conditioning: z = (a, b) ~ N(μ, Σ), observations H z + c + noise.
    const mu = [1, 0.5]
    const S = [
      [4, 4],
      [4, 4 + 0.49],
    ]
    const H = [
      [1, 0],
      [0, 3],
    ]
    const R = [1, 0.25]
    const r = [2 - mu[0], 4 - 1 - 3 * mu[1]]
    // Innovation covariance H Σ Hᵀ + R (2 × 2) and gain Σ Hᵀ (H Σ Hᵀ + R)⁻¹.
    const SH = S.map((row) => H.map((h) => row[0] * h[0] + row[1] * h[1]))
    const HSH = H.map((h) => [0, 1].map((j) => h[0] * SH[0][j] + h[1] * SH[1][j]))
    HSH[0][0] += R[0]
    HSH[1][1] += R[1]
    const det = HSH[0][0] * HSH[1][1] - HSH[0][1] * HSH[1][0]
    const inv = [
      [HSH[1][1] / det, -HSH[0][1] / det],
      [-HSH[1][0] / det, HSH[0][0] / det],
    ]
    const K = SH.map((row) => [0, 1].map((j) => row[0] * inv[0][j] + row[1] * inv[1][j]))
    const post = mu.map((m0, i) => m0 + K[i][0] * r[0] + K[i][1] * r[1])
    const cov = [0, 1].map((i) => S[i][i] - (K[i][0] * SH[i][0] + K[i][1] * SH[i][1]))
    expect(marginal(s, 'a').mean).toBeCloseTo(post[0], 10)
    expect(marginal(s, 'b').mean).toBeCloseTo(post[1], 10)
    expect(marginal(s, 'a').variance).toBeCloseTo(cov[0], 10)
    expect(marginal(s, 'b').variance).toBeCloseTo(cov[1], 10)
  })

  it('an interval factor gives the truncated-normal moments; a 0 states the complement of a one-sided interval', () => {
    const truncated = (lower: number, upper: number) =>
      model('truncated', (mm) => {
        const x = mm.variable('x', dist.Normal(1, 2))
        mm.observed('y', dist.Bernoulli(mm.deterministic('inside', 'interval', [x, lower, upper])))
      })
    const s = run(modelExpectationPropagation(truncated(0, 3), { data: { y: 1 } }), undefined, 10)
    const want = intervalTilted(1, 4, 0, 3)
    expect(marginal(s, 'x').mean).toBeCloseTo(want.mean, 12)
    expect(marginal(s, 'x').variance).toBeCloseTo(want.variance, 12)
    const outside = run(modelExpectationPropagation(truncated(-Infinity, 0.5), { data: { y: 0 } }), undefined, 10)
    const above = intervalTilted(1, 4, 0.5, Infinity)
    expect(marginal(outside, 'x').mean).toBeCloseTo(above.mean, 12)
    expect(() => compileGaussianModel(truncated(0, 3), { data: { y: 0 } })).toThrow(/complement/)
    // The joint density reads the indicator.
    expect(logJoint(truncated(0, 3), { x: 4 }, { data: { y: 1 } })).toBe(-Infinity)
    expect(Number.isFinite(logJoint(truncated(0, 3), { x: 2 }, { data: { y: 1 } }))).toBe(true)
  })

  it('does not report convergence when every update was skipped', () => {
    // An empty interval [2, 2] has no mass, so its tilted moments are unusable and every update is skipped.
    const point = model('point', (mm) => {
      const x = mm.variable('x', dist.Normal(1, 2))
      mm.observed('y', dist.Bernoulli(mm.deterministic('inside', 'interval', [x, 2, 2])))
    })
    const s = run(modelExpectationPropagation(point, { data: { y: 1 } }), undefined, 5)
    expect(s.skipped).toBeGreaterThan(0)
    expect(s.converged).toBe(false)
  })

  it('refuses models that are not linear-Gaussian', () => {
    const product = model('product', (mm) => {
      const a = mm.variable('a', dist.Normal(0, 1))
      const b = mm.variable('b', dist.Normal(0, 1))
      mm.observed('y', dist.Normal(mm.deterministic('ab', 'product', [a, b]), 1))
    })
    expect(() => compileGaussianModel(product, { data: { y: 1 } })).toThrow(/multiplies/)
    const gamma = model('gamma', (mm) => {
      mm.variable('g', dist.Gamma(2, 1))
    })
    expect(() => compileGaussianModel(gamma)).toThrow(/Gamma/)
  })
})
