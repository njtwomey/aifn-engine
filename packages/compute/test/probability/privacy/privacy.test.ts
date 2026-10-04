/**
 * Differential privacy against references (`fixtures/probability/privacy.json`): RDP of the subsampled Gaussian and its
 * (ε, δ) conversion against Opacus's accountant, and the analytic Gaussian σ against scipy root finding. Laws by
 * simulation: the Laplace mechanism's empirical privacy loss stays within ε; the Gaussian mechanism's δ(ε) equals the
 * Monte Carlo E[(1 − e^{ε−L})₊]; the exponential mechanism's probabilities move by at most e^ε between neighbours;
 * randomised response is unbiased. Composition and zCDP closed forms; clip-and-noise; DP-SGD training.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { add, tensor, toFlat, sub, mul, sum, square, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import {
  advancedComposition,
  analyticGaussianSigma,
  classicGaussianSigma,
  clipAndNoise,
  DEFAULT_ORDERS,
  dpSgdEpsilon,
  exponentialMechanism,
  exponentialMechanismProbabilities,
  gaussianDelta,
  gaussianEpsilon,
  gaussianMechanism,
  gaussianZcdp,
  laplaceMechanism,
  randomisedResponse,
  randomisedResponseEstimate,
  rdpSubsampledGaussian,
  rdpToEpsilon,
  sequentialComposition,
  zcdpToEpsilon,
} from 'aifn-compute/probability/privacy'
import { privateTraining } from 'aifn-compute/nn/training'
import { fixture } from '../../fixtures'

type F = {
  orders: number[]
  accounting: {
    q: number
    sigma: number
    steps: number
    delta: number
    rdp: number[]
    epsilon: number
    order: number
  }[]
  analytic: { sensitivity: number; epsilon: number; delta: number; sigma: number }[]
}
const F = fixture<F>('probability/privacy')

describe('RDP accounting', () => {
  it("uses Opacus's default orders", () => {
    expect(DEFAULT_ORDERS.length).toBe(F.orders.length)
    DEFAULT_ORDERS.forEach((a, i) => expect(a).toBeCloseTo(F.orders[i], 12))
  })
  for (const c of F.accounting)
    it(`matches Opacus for q = ${c.q}, σ = ${c.sigma}, ${c.steps} steps`, () => {
      const rdp = rdpSubsampledGaussian(c.q, c.sigma, c.steps)
      c.rdp.forEach((r, i) => expect(Math.abs(rdp[i] - r)).toBeLessThanOrEqual(1e-9 * (1 + Math.abs(r))))
      const e = rdpToEpsilon(rdp, c.delta)
      expect(e.epsilon).toBeCloseTo(c.epsilon, 9)
      expect(e.order).toBeCloseTo(c.order, 12)
      expect(dpSgdEpsilon(c.q, c.sigma, c.steps, c.delta)).toBeCloseTo(c.epsilon, 9)
    })

  it('gives the full-batch Gaussian RDP α/(2σ²) per step, which is ρα for its zCDP ρ', () => {
    const rdp = rdpSubsampledGaussian(1, 2, 3, [2, 5])
    expect(rdp[0]).toBeCloseTo((3 * 2) / 8, 12)
    expect(rdp[1]).toBeCloseTo(3 * 5 * gaussianZcdp(1, 2), 12)
  })
})

describe('Gaussian mechanism calibration', () => {
  for (const c of F.analytic)
    it(`analytic σ matches scipy for Δ = ${c.sensitivity}, ε = ${c.epsilon}, δ = ${c.delta}`, () => {
      const sigma = analyticGaussianSigma(c.sensitivity, c.epsilon, c.delta)
      expect(sigma).toBeCloseTo(c.sigma, 8)
      expect(gaussianDelta(c.sensitivity, sigma, c.epsilon)).toBeLessThanOrEqual(c.delta * (1 + 1e-9))
    })

  it('gaussianEpsilon inverts the privacy profile and the analytic σ', () => {
    for (const c of F.analytic) expect(gaussianEpsilon(c.sensitivity, c.sigma, c.delta)).toBeCloseTo(c.epsilon, 6)
    expect(gaussianEpsilon(1, 1e3, 0.4)).toBe(0)
  })

  it('analytic σ is below the classic one where the classic applies', () => {
    for (const eps of [0.1, 0.5, 0.9])
      expect(analyticGaussianSigma(1, eps, 1e-5)).toBeLessThan(classicGaussianSigma(1, eps, 1e-5))
  })

  it('δ(ε) equals the Monte Carlo E[(1 − e^{ε − L})₊] of the privacy loss L', () => {
    const sigma = 1.5
    const eps = 0.8
    const x = toFlat(normals(stream('loss'), [200000]))
    let total = 0
    for (const z of x) {
      const out = sigma * z // output on D (f = 0); D′ has f = 1
      const L = (out - 1) ** 2 / (2 * sigma * sigma) - (out * out) / (2 * sigma * sigma)
      total += Math.max(0, 1 - Math.exp(eps - L))
    }
    expect(total / x.length).toBeCloseTo(gaussianDelta(1, sigma, eps), 3)
  })

  it('adds noise of standard deviation σ', () => {
    const out = gaussianMechanism(new Float64Array(50000), 2, stream(1))
    const v = out.reduce((a, b) => a + b * b, 0) / out.length
    expect(Math.sqrt(v)).toBeCloseTo(2, 1)
  })
})

describe('Laplace mechanism', () => {
  it('has variance 2(Δ/ε)² and an empirical privacy loss within ε', () => {
    const eps = 1
    const n = 200000
    const a = laplaceMechanism(new Float64Array(n), 1, eps, stream('a'))
    const b = laplaceMechanism(new Float64Array(n).fill(1), 1, eps, stream('b'))
    const variance = a.reduce((s, u) => s + u * u, 0) / n
    expect(variance).toBeCloseTo(2, 1)
    // Histogram both on bins of width 0.5 over [−3, 4]; the log ratio of well-filled bins stays within ε.
    const bins = (v: Float64Array) => {
      const h = new Float64Array(14)
      for (const u of v) {
        const k = Math.floor((u + 3) / 0.5)
        if (k >= 0 && k < 14) h[k]++
      }
      return h
    }
    const ha = bins(a)
    const hb = bins(b)
    let worst = 0
    for (let k = 0; k < 14; k++)
      if (ha[k] > 2000 && hb[k] > 2000) worst = Math.max(worst, Math.abs(Math.log(ha[k] / hb[k])))
    expect(worst).toBeLessThan(eps + 0.05)
    expect(worst).toBeGreaterThan(eps - 0.1)
  })
})

describe('exponential mechanism and randomised response', () => {
  it('moves each probability by at most e^ε between neighbouring utilities', () => {
    const u = [3, 1, 0.5, 2]
    const v = [2.5, 1.8, 0.5, 1.2] // |u − v| ≤ 1 = Δ
    const p = exponentialMechanismProbabilities(u, 1, 0.7)
    const q = exponentialMechanismProbabilities(v, 1, 0.7)
    for (let i = 0; i < 4; i++) expect(Math.abs(Math.log(p[i] / q[i]))).toBeLessThanOrEqual(0.7 + 1e-12)
    const counts = new Float64Array(4)
    const s = stream('exp')
    for (let k = 0; k < 20000; k++) counts[exponentialMechanism(u, 1, 0.7, s)]++
    for (let i = 0; i < 4; i++) expect(counts[i] / 20000).toBeCloseTo(p[i], 1)
  })

  it('handles 10⁶ candidates and refuses a set with no finite utility (review G2)', () => {
    // Math.max(...logits) overflowed the stack on long candidate lists; all −∞ gave NaN probabilities.
    const u = Float64Array.from({ length: 1_000_000 }, (_, i) => -i)
    const p = exponentialMechanismProbabilities(u, 1, 2)
    expect(p[0]).toBeCloseTo(1 - Math.exp(-1), 12)
    expect(() => exponentialMechanismProbabilities([-Infinity, -Infinity], 1, 1)).toThrow(/finite utility/)
  })

  it('randomised response is unbiased', () => {
    const bits = Float64Array.from({ length: 40000 }, (_, i) => (i % 10 < 3 ? 1 : 0))
    const reports = randomisedResponse(bits, 1, stream('rr'))
    expect(randomisedResponseEstimate(reports, 1)).toBeCloseTo(0.3, 1)
  })
})

describe('composition', () => {
  it('adds sequentially and is tighter by advanced composition for many small steps', () => {
    expect(sequentialComposition([{ epsilon: 0.5, delta: 1e-6 }, { epsilon: 0.2 }])).toEqual({
      epsilon: 0.7,
      delta: 1e-6,
    })
    const k = 1000
    const adv = advancedComposition(0.01, 0, k, 1e-6)
    expect(adv.epsilon).toBeCloseTo(0.01 * Math.sqrt(2 * k * Math.log(1e6)) + k * 0.01 * Math.expm1(0.01), 12)
    expect(adv.epsilon).toBeLessThan(k * 0.01)
  })

  it('converts zCDP to (ε, δ)', () => {
    const rho = gaussianZcdp(1, 3)
    expect(rho).toBeCloseTo(1 / 18, 12)
    expect(zcdpToEpsilon(rho, 1e-5)).toBeCloseTo(rho + 2 * Math.sqrt(rho * Math.log(1e5)), 12)
  })
})

describe('clipAndNoise', () => {
  it('without noise is the mean of the clipped per-example gradients', () => {
    const g = {
      w: tensor([
        [3, 4],
        [0.3, 0.4],
      ]),
      b: tensor([0, 0]),
    }
    const r = clipAndNoise(g, 1, 0, stream(1))
    expect(Array.from(r.norms)).toEqual([5, 0.5])
    expect(r.clippedShare).toBe(0.5)
    const w = toFlat(r.gradient.w)
    expect(w[0]).toBeCloseTo((0.6 + 0.3) / 2, 12)
    expect(w[1]).toBeCloseTo((0.8 + 0.4) / 2, 12)
  })

  it('adds N(0, σ²C²) to the sum before dividing by the batch size', () => {
    const g = { w: tensor(new Array(4).fill(new Array(20000).fill(0))) }
    const r = clipAndNoise(g, 2, 1.5, stream(2))
    const w = toFlat(r.gradient.w)
    const sd = Math.sqrt(w.reduce((a, b) => a + b * b, 0) / w.length)
    expect(sd).toBeCloseTo((1.5 * 2) / 4, 1)
  })
})

describe('privateTraining', () => {
  // Linear regression y = 2x − 1 on 200 points; the loss of one example is its squared error.
  const xs = toFlat(normals(stream('x'), [200]))
  const data = { x: tensor(Array.from(xs, (v) => [v])), y: tensor(Array.from(xs, (v) => 2 * v - 1)) }
  const loss = (p: { w: Tensor; b: Tensor }, e: { x: Tensor; y: Tensor }) =>
    sum(square(sub(add(mul(p.w, e.x), p.b), e.y)))
  const start = { params: { w: tensor([0]), b: tensor([0]) } }

  it('learns with little noise and reports the RDP ε', () => {
    const alg = privateTraining({ loss, data, batchSize: 40, clipNorm: 5, noiseMultiplier: 0.5, delta: 1e-5 })
    const s = run(alg, start, 300, { stream: stream(3) })
    expect(toFlat(s.params.w)[0]).toBeCloseTo(2, 0)
    expect(toFlat(s.params.b)[0]).toBeCloseTo(-1, 0)
    expect(s.epsilon).toBeCloseTo(dpSgdEpsilon(40 / 200, 0.5, 300, 1e-5), 12)
  })

  it('reports ε = ∞ without noise and samples about qn examples per step', () => {
    const s = run(privateTraining({ loss, data, batchSize: 50, noiseMultiplier: 0 }), start, 20, { stream: stream(4) })
    expect(s.epsilon).toBe(Infinity)
    expect(s.batchSize).toBeGreaterThan(25)
    expect(s.batchSize).toBeLessThan(80)
  })
})

describe('per-example gradients through a loss with targets', () => {
  it('vmap(grad) of binary cross-entropy matches one grad per example', async () => {
    const { vmap, grad } = await import('aifn-compute/foundation/autodiff')
    const { binaryCrossEntropyWithLogits } = await import('aifn-compute/learning/losses')
    const w = tensor([0.5, -1])
    const X = tensor([
      [1, 2],
      [0, 1],
      [3, -1],
    ])
    const y = tensor([1, 0, 1])
    const lossOne = (p: Tensor, x: Tensor, t: Tensor) => binaryCrossEntropyWithLogits(sum(mul(p, x)), t)
    const batched = toFlat(vmap(grad(lossOne), { inAxes: [null, 0, 0] })(w, X, y) as Tensor)
    const rows = [
      [1, 2],
      [0, 1],
      [3, -1],
    ]
    rows.forEach((r, i) => {
      const g = toFlat(grad(lossOne)(w, tensor(r), tensor(toFlat(y)[i])) as Tensor)
      expect(batched[2 * i]).toBeCloseTo(g[0], 12)
      expect(batched[2 * i + 1]).toBeCloseTo(g[1], 12)
    })
  })
})
