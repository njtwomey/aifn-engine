import { describe, expect, it } from 'vitest'
import { Beta, Dirichlet, Gamma, Normal } from 'aifn-compute/probability/distributions'
import {
  assumedDensityFiltering,
  dampGaussian,
  divideGaussians,
  epLogEvidence,
  expectationPropagation,
  gaussianMoments,
  intervalTilted,
  messageOf,
  messageToDistribution,
  multiplyGaussians,
  multiplyMessages,
  naturalGaussian,
  probitTilted,
  stepTilted,
  tiltedByQuadrature,
  type EpOptions,
} from 'aifn-compute/inference/expectation-propagation'
import { integrate } from 'aifn-compute/numerics/quadrature'
import { normalCdf } from 'aifn-compute/numerics/special'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

/**
 * ∫ f over [a, b] as a sum over unit panels: a single adaptive rule over a wide range can miss a narrow peak (its
 * first Kronrod estimate sees nothing there and reports convergence).
 */
function panels(f: (t: number) => number, a: number, b: number): number {
  let total = 0
  for (let lo = a; lo < b; lo += 1) total += integrate(f, lo, Math.min(lo + 1, b), { rtol: 1e-13 }).value
  return total
}

/** Moments of N(θ; m, v) f(θ) on [lower, upper] (default ±12 sd) by quadrature: log Z, mean, variance. */
function quadTilted(m: number, v: number, f: (t: number) => number, bounds: [number, number] = [-Infinity, Infinity]) {
  const s = Math.sqrt(v)
  const w = (t: number) => (Math.exp((-0.5 * (t - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)) * f(t)
  const lo = Math.max(m - 12 * s, bounds[0])
  const hi = Math.min(m + 12 * s, bounds[1])
  const z = panels(w, lo, hi)
  const mean = panels((t) => t * w(t), lo, hi) / z
  const second = panels((t) => (t - mean) ** 2 * w(t), lo, hi) / z
  return { logZ: Math.log(z), mean, variance: second }
}

describe('message algebra', () => {
  it('multiply, divide and damp in natural parameters', () => {
    const a = naturalGaussian(1, 2)
    const b = naturalGaussian(-1, 0.5)
    const ab = multiplyGaussians(a, b)
    const m = gaussianMoments(ab)
    expect(m.variance).toBeCloseTo(1 / (1 / 2 + 1 / 0.5), 14)
    expect(m.mean).toBeCloseTo(m.variance * (1 / 2 - 1 / 0.5), 14)
    const back = divideGaussians(ab, b)
    expect(back.precision).toBeCloseTo(a.precision, 14)
    expect(dampGaussian(a, b, 0.25).precision).toBeCloseTo(0.75 * a.precision + 0.25 * b.precision, 14)
  })
  it('batches broadcast elementwise', () => {
    const g = naturalGaussian(tensor([0, 1]), tensor([1, 4]))
    expect(toFlat(gaussianMoments(multiplyGaussians(g, g)).variance as Tensor)).toEqual([0.5, 2])
  })
  it('exponential-family messages round-trip through distribution objects', () => {
    for (const d of [Normal(1, 2), Gamma(3, 2), Beta(2, 5), Dirichlet(tensor([1, 2, 3]))]) {
      const back = messageToDistribution(messageOf(d))
      expect(back.name).toBe(d.name)
      expect(toFlat(tensor(back.mean() as number) as Tensor)).toEqual(
        toFlat(tensor(d.mean() as number) as Tensor).map((x) => expect.closeTo(x, 12)),
      )
    }
    // Beta(2, 5) · Beta(3, 1) ∝ Beta(4, 5): natural parameters add.
    const prod = messageToDistribution(multiplyMessages(messageOf(Beta(2, 5)), messageOf(Beta(3, 1))))
    expect(prod.params.a).toBeCloseTo(4, 12)
    expect(prod.params.b).toBeCloseTo(5, 12)
  })
})

describe('tilted moments against quadrature', () => {
  const cavities = [
    [0.3, 1.2],
    [-2, 0.5],
    [4, 3],
  ]
  it.each(cavities)('step, probit and interval at N(%d, %d)', (m, v) => {
    const cases = [
      [stepTilted(m, v, 0.5), () => 1, [0.5, Infinity]],
      [
        probitTilted(m, v, -1, { offset: 0.2, noiseVariance: 0.7 }),
        (t: number) => normalCdf(-(t - 0.2) / Math.sqrt(0.7)) as number,
      ],
      [intervalTilted(m, v, -1, 1.5), () => 1, [-1, 1.5]],
    ] as const
    for (const [closed, f, bounds] of cases) {
      const q = quadTilted(m, v, f, bounds as [number, number] | undefined)
      expect(closed.logZ).toBeCloseTo(q.logZ, 6)
      expect(closed.mean).toBeCloseTo(q.mean, 6)
      expect(closed.variance).toBeCloseTo(q.variance, 6)
    }
  })
  it('stays finite far in the tails', () => {
    const t = stepTilted(-40, 1)
    expect(t.mean).toBeGreaterThan(0)
    expect(t.variance).toBeGreaterThan(0)
    expect(Number.isFinite(t.logZ)).toBe(true)
  })
  it('broadcasts over tensors', () => {
    const t = probitTilted(tensor([0, 1, 2]), 1)
    expect(t.mean.shape).toEqual([3])
    expect(toFlat(t.mean)[1]).toBeCloseTo(probitTilted(1, 1).mean, 14)
  })
  it('quadrature tilt agrees with the closed form', () => {
    const q = tiltedByQuadrature(0.4, 2, (t) => Math.log(normalCdf(t) as number))
    const c = probitTilted(0.4, 2)
    expect(q.mean).toBeCloseTo(c.mean, 6)
    expect(q.variance).toBeCloseTo(c.variance, 6)
  })
})

describe('EP on scalar problems', () => {
  it('with Gaussian factors EP is exact, evidence included', () => {
    const xs = [0.5, 1.5, -0.2]
    const opts: EpOptions = {
      prior: { mean: 0, variance: 4 },
      factors: 3,
      tilted: (i, c) => {
        const variance = 1 / (1 / c.variance + 1)
        const mean = variance * (c.mean / c.variance + xs[i])
        const s2 = c.variance + 1
        return { logZ: -0.5 * Math.log(2 * Math.PI * s2) - (0.5 * (xs[i] - c.mean) ** 2) / s2, mean, variance }
      },
    }
    const s = run(expectationPropagation(opts), undefined, 20)
    const post = 1 / (1 / 4 + 3)
    expect(s.posterior.variance).toBeCloseTo(post, 12)
    expect(s.posterior.mean).toBeCloseTo(post * (0.5 + 1.5 - 0.2), 12)
    // log N(x; 0, I + 4·11ᵀ) by quadrature over θ.
    const ev = integrate(
      (t) =>
        Math.exp(
          -0.5 * Math.log(8 * Math.PI) -
            (t * t) / 8 +
            xs.reduce((a, xi) => a - 0.5 * Math.log(2 * Math.PI) - 0.5 * (xi - t) ** 2, 0),
        ),
      -30,
      30,
      { rtol: 1e-12 },
    ).value
    expect(epLogEvidence(s)).toBeCloseTo(Math.log(ev), 10)
  })

  // A probit model for one scalar θ ~ N(0, 2): P(yᵢ = 1 | θ) = Φ(yᵢ(θ − cᵢ)). The exact posterior by quadrature.
  const cuts = [-1, 0.5, 0.2, 1.5, -0.3]
  const signs = [1, 1, -1, -1, 1]
  const probit: EpOptions = {
    prior: { mean: 0, variance: 2 },
    factors: cuts.length,
    tilted: (i, c) => probitTilted(c.mean, c.variance, signs[i], { offset: cuts[i] }),
  }
  const logPost = (t: number) =>
    -(t * t) / 4 -
    0.5 * Math.log(4 * Math.PI) +
    cuts.reduce((a, c, i) => a + Math.log(normalCdf(signs[i] * (t - c)) as number), 0)
  const Z = panels((t) => Math.exp(logPost(t)), -15, 15)
  const mean = panels((t) => t * Math.exp(logPost(t)), -15, 15) / Z
  const variance = panels((t) => (t - mean) ** 2 * Math.exp(logPost(t)), -15, 15) / Z

  it('probit factors: EP is close to the exact moments and evidence; the first sweep is ADF', () => {
    const s = run(expectationPropagation(probit), undefined, 500)
    expect(s.converged).toBe(true)
    expect(Math.abs(s.posterior.mean - mean)).toBeLessThan(0.02)
    expect(Math.abs(s.posterior.variance / variance - 1)).toBeLessThan(0.05)
    expect(Math.abs(epLogEvidence(s) - Math.log(Z))).toBeLessThan(0.02)
    const ep = run(expectationPropagation(probit), undefined, cuts.length)
    const adf = run(assumedDensityFiltering(probit), undefined, cuts.length)
    expect(ep.posterior.mean).toBeCloseTo(adf.posterior.mean, 12)
    expect(ep.posterior.variance).toBeCloseTo(adf.posterior.variance, 12)
  })

  it('damped and power EP converge', () => {
    expect(run(expectationPropagation({ ...probit, damping: 0.5 }), undefined, 5000).converged).toBe(true)
    const power = run(expectationPropagation({ ...probit, power: 0.5 }), undefined, 5000)
    expect(power.converged).toBe(true)
    expect(epLogEvidence(power)).toBeNaN()
  })

  it('EP and ADF satisfy the Algorithm protocol', () => {
    const record = { mean: (s: { posterior: { mean: number } }) => s.posterior.mean }
    checkProtocol(expectationPropagation(probit), undefined, { steps: 12, record })
    checkProtocol(expectationPropagation({ ...probit, damping: 0.3, power: 0.7 }), undefined, { steps: 12, record })
    checkProtocol(assumedDensityFiltering(probit), undefined, { steps: 5, record })
  })
})
