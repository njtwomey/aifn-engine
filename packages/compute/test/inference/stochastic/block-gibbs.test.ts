/**
 * Block Gibbs and Rao–Blackwellisation (`gibbs` with `Block`s, `gaussianConditionals` over a partition,
 * `conditionalMean`, `raoBlackwell`) on a conjugate model with a known posterior: Bayesian linear regression with a
 * normal–gamma prior on the coefficients β and the noise precision τ.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { reshape, tensor, toFlat, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import {
  conditionalMean,
  effectiveSampleSize,
  gaussianConditionals,
  gibbs,
  monteCarloStandardError,
  raoBlackwell,
  sampleChains,
  type Block,
} from 'aifn-compute/inference/stochastic'
import { gammaVariate } from 'aifn-compute/probability/samplers'

const at = (t: Tensor | number, i: number) => (typeof t === 'number' ? t : toFlat(t)[i])

// ── The model ───────────────────────────────────────────────────────────────────────────────────────────────────────
// y = Xβ + ε, ε ~ N(0, 1/τ); β | τ ~ N(m₀, (τΛ₀)⁻¹), τ ~ Gamma(a₀, rate b₀). With nearly collinear columns, β₀ and β₁
// are strongly correlated a posteriori.
// Posterior (Bishop, 2006, §3.3 and exercise 3.12): Λₙ = Λ₀ + XᵀX, mₙ = Λₙ⁻¹(Λ₀m₀ + Xᵀy), aₙ = a₀ + n/2,
// bₙ = b₀ + ½(yᵀy + m₀ᵀΛ₀m₀ − mₙᵀΛₙmₙ); τ ~ Gamma(aₙ, bₙ) and β is Student t with mean mₙ, covariance bₙ/(aₙ − 1)·Λₙ⁻¹.

const n = 30
const noise = toFlat(normals(stream('noise'), n, 0, 0.5))
const u = toFlat(normals(stream('x'), n))
const v = toFlat(normals(stream('collinear'), n, 0, 0.15))
const X = Array.from({ length: n }, (_, i) => [u[i], u[i] + v[i]])
const y = X.map((r, i) => 1 * r[0] - 0.5 * r[1] + noise[i])
const lambda0 = 0.1
const a0 = 2
const b0 = 1

const XtX = [0, 1].map((j) => [0, 1].map((k) => X.reduce((s, r) => s + r[j] * r[k], 0)))
const Xty = [0, 1].map((j) => X.reduce((s, r, i) => s + r[j] * y[i], 0))
const L = [
  [XtX[0][0] + lambda0, XtX[0][1]],
  [XtX[1][0], XtX[1][1] + lambda0],
]
const det = L[0][0] * L[1][1] - L[0][1] * L[1][0]
const Linv = [
  [L[1][1] / det, -L[0][1] / det],
  [-L[1][0] / det, L[0][0] / det],
]
const mn = [Linv[0][0] * Xty[0] + Linv[0][1] * Xty[1], Linv[1][0] * Xty[0] + Linv[1][1] * Xty[1]]
const an = a0 + n / 2
const bn =
  b0 +
  0.5 *
    (y.reduce((s, yi) => s + yi * yi, 0) -
      (mn[0] * (L[0][0] * mn[0] + L[0][1] * mn[1]) + mn[1] * (L[1][0] * mn[0] + L[1][1] * mn[1])))
const posterior = {
  beta: mn,
  betaVariance: [0, 1].map((j) => (bn / (an - 1)) * Linv[j][j]),
  tau: an / bn,
}
const correlation = Linv[0][1] / Math.sqrt(Linv[0][0] * Linv[1][1])

/** ½[(y − Xβ)ᵀ(y − Xβ) + λ₀βᵀβ], the quadratic in τ's conditional. */
const quadratic = (b0_: number, b1: number) =>
  0.5 * (X.reduce((s, r, i) => s + (y[i] - r[0] * b0_ - r[1] * b1) ** 2, 0) + lambda0 * (b0_ * b0_ + b1 * b1))

// x = (β₀, β₁, τ).
const tauShape = a0 + (n + 2) / 2
const tauBlock: Block = {
  coordinates: [2],
  draw: (x, s) => [gammaVariate(s, tauShape, 1 / (b0 + quadratic(at(x, 0), at(x, 1))))],
  mean: (x) => [tauShape / (b0 + quadratic(at(x, 0), at(x, 1)))],
}
/** β | τ ~ N(mₙ, (τΛₙ)⁻¹), jointly: a 2 × 2 Cholesky factor of Λₙ⁻¹/τ. */
const betaBlock: Block = {
  coordinates: [0, 1],
  draw: (x, s: Stream) => {
    const tau = at(x, 2)
    const c00 = Linv[0][0] / tau
    const c01 = Linv[0][1] / tau
    const c11 = Linv[1][1] / tau
    const l00 = Math.sqrt(c00)
    const l10 = c01 / l00
    const l11 = Math.sqrt(c11 - l10 * l10)
    const z = toFlat(normals(s, 2))
    return [mn[0] + l00 * z[0], mn[1] + l10 * z[0] + l11 * z[1]]
  },
  mean: () => mn,
}
/** One coefficient at a time: β_j | β_k, τ ~ N(mₙⱼ − (Λₙⱼₖ/Λₙⱼⱼ)(β_k − mₙₖ), 1/(τΛₙⱼⱼ)). */
const coefficient = (j: number): Block => {
  const k = 1 - j
  const mean = (x: Vector) => [mn[j] - (L[j][k] / L[j][j]) * (at(x, k) - mn[k])]
  return {
    coordinates: [j],
    draw: (x, s) => [mean(x)[0] + toFlat(normals(s, 1))[0] / Math.sqrt(at(x, 2) * L[j][j])],
    mean,
  }
}

const start = { x0: [0, 0, 1] }
const options = { chains: 4, steps: 2000, warmup: 100 }
const block = sampleChains(gibbs([betaBlock, tauBlock]), start, { ...options, stream: stream('block') })
const single = sampleChains(gibbs([coefficient(0), coefficient(1), tauBlock]), start, {
  ...options,
  stream: stream('single'),
})

describe('block Gibbs on a conjugate regression', () => {
  it('the posterior is strongly correlated (the case for blocking)', () => {
    expect(correlation).toBeLessThan(-0.9)
  })

  it('recovers the posterior means and variances of β and τ', () => {
    const mcse = monteCarloStandardError(block.draws) as Tensor
    const draws = toFlat(block.draws)
    const count = draws.length / 3
    const mean = [0, 1, 2].map((k) => draws.reduce((s, x, i) => (i % 3 === k ? s + x : s), 0) / count)
    const variance = [0, 1].map((k) => draws.reduce((s, x, i) => (i % 3 === k ? s + (x - mean[k]) ** 2 : s), 0) / count)
    ;[...posterior.beta, posterior.tau].forEach((m, k) => expect(Math.abs(mean[k] - m)).toBeLessThan(4 * at(mcse, k)))
    variance.forEach((s2, k) => expect(Math.abs(s2 / posterior.betaVariance[k] - 1)).toBeLessThan(0.1))
  })

  it('mixes far better than one coefficient at a time', () => {
    const essBlock = effectiveSampleSize(block.draws) as Tensor
    const essSingle = effectiveSampleSize(single.draws) as Tensor
    for (const k of [0, 1]) expect(at(essBlock, k)).toBeGreaterThan(5 * at(essSingle, k))
    // The one-at-a-time chain is still correct, only slower.
    const mcse = monteCarloStandardError(single.draws) as Tensor
    const draws = toFlat(single.draws)
    const mean0 = draws.reduce((s, x, i) => (i % 3 === 0 ? s + x : s), 0) / (draws.length / 3)
    expect(Math.abs(mean0 - posterior.beta[0])).toBeLessThan(4 * at(mcse, 0))
  })
})

describe('Rao–Blackwellisation', () => {
  it('E[β | τ] = mₙ makes the estimate of E[β] exact, and E[τ | β] reduces the error of E[τ]', () => {
    const rb = raoBlackwell(block.draws, conditionalMean([betaBlock, tauBlock]))
    expect(rb.values.shape).toEqual([4, block.draws.shape[1], 3])
    expect(at(rb.mean, 0)).toBeCloseTo(posterior.beta[0], 12)
    expect(at(rb.mean, 1)).toBeCloseTo(posterior.beta[1], 12)
    const rbMcse = at(monteCarloStandardError(rb.values) as Tensor, 2)
    const plainMcse = at(monteCarloStandardError(block.draws) as Tensor, 2)
    expect(Math.abs(at(rb.mean, 2) - posterior.tau)).toBeLessThan(4 * rbMcse)
    expect(rbMcse).toBeLessThan(0.8 * plainMcse)
  })

  it('a scalar expectation and n×d draws', () => {
    const flat = block.draws.shape[1] * 4
    const rb = raoBlackwell(
      // One long chain: the four chains end to end.
      reshape(block.draws, [flat, 3]),
      (x) => tauShape / (b0 + quadratic(at(x, 0), at(x, 1))),
    )
    expect(rb.values.shape).toEqual([1, flat, 1])
    expect(at(rb.mean, 0)).toBeCloseTo(
      at(raoBlackwell(block.draws, conditionalMean([betaBlock, tauBlock])).mean, 2),
      12,
    )
  })
})

describe('Gaussian block conditionals', () => {
  // x₀ and x₁ correlated 0.95, x₂ independent of both.
  const mean = [1, -1, 2]
  const cov = [
    [1, 0.95, 0],
    [0.95, 1, 0],
    [0, 0, 4],
  ]
  const run = (partition?: number[][]) =>
    sampleChains(
      gibbs(gaussianConditionals(mean, cov, partition)),
      { x0: [0, 0, 0] },
      {
        chains: 4,
        steps: 1500,
        stream: stream('gaussian'),
        warmup: 50,
      },
    )

  it('the joint block of the correlated pair mixes better, and both chains recover the moments', () => {
    const joint = run([[0, 1], [2]])
    const single = run()
    expect(at(effectiveSampleSize(joint.draws) as Tensor, 0)).toBeGreaterThan(
      4 * at(effectiveSampleSize(single.draws) as Tensor, 0),
    )
    for (const r of [joint, single]) {
      const mcse = monteCarloStandardError(r.draws) as Tensor
      const x = toFlat(r.draws)
      mean.forEach((m, k) => {
        const avg = x.reduce((s, v, i) => (i % 3 === k ? s + v : s), 0) / (x.length / 3)
        expect(Math.abs(avg - m)).toBeLessThan(4 * at(mcse, k))
      })
    }
    // The pair's block draws the exact marginal: its draws have the target's correlation.
    const x = toFlat(joint.draws)
    const count = x.length / 3
    let c = 0
    for (let i = 0; i < count; i++) c += (x[3 * i] - 1) * (x[3 * i + 1] + 1)
    expect(c / count).toBeCloseTo(0.95, 1)
  })

  it('the conditional means are those of the Gaussian', () => {
    const [b01, b2] = gaussianConditionals(mean, cov, [[0, 1], [2]])
    const x = tensor([0.3, -0.2, 5]) as Vector
    // Independent blocks: the conditional mean is the marginal mean.
    expect(Array.from(b01.mean!(x))).toEqual([1, -1].map((m) => expect.closeTo(m, 12)))
    expect(Array.from(b2.mean!(x))[0]).toBeCloseTo(2, 12)
    const [s0] = gaussianConditionals(mean, cov)
    // x₀ | x₁, x₂ ~ N(1 + 0.95(x₁ + 1), 1 − 0.95²).
    expect(Array.from(s0.mean!(x))[0]).toBeCloseTo(1 + 0.95 * (-0.2 + 1), 12)
  })

  it('checks blocks', () => {
    expect(() => gibbs([{ coordinates: [0, 2], draw: () => [0, 0] }])).toThrow(/coordinate 1/)
    expect(() => conditionalMean([{ coordinates: [0], draw: () => [0] }])).toThrow(/no mean/)
    const bad = gibbs([{ coordinates: [0, 1], draw: () => [0] }])
    expect(() => sampleChains(bad, { x0: [0, 0] }, { steps: 1, stream: stream(0), chains: 1 })).toThrow(/drew 1/)
  })
})
