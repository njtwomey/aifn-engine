import { stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import {
  bbvi,
  elbo,
  elboGradient,
  fullRankGaussian,
  gradientVariance,
  meanFieldGaussian,
  type GaussianFamily,
} from 'aifn-compute/inference/variational'
import { describe, expect, it } from 'vitest'
import { gaussianTarget } from '../stochastic/targets'
import { checkProtocol } from '../../protocol'

const m = [1, -0.5]
const S = [
  [1, 0.6],
  [0.6, 0.8],
]
const target = gaussianTarget(m, S)
const P = toFlat(inverse(fromRows(S)))

/** The exact ELBO of q = N(μ, C) against the normalised Gaussian target: E_q[log p] + H[q]. */
function exactElbo(family: GaussianFamily, lambda: number[]): number {
  const mu = toFlat(family.mean(lambda))
  const C = toFlat(family.covariance(lambda))
  let tr = 0
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) tr += P[i * 2 + j] * (C[j * 2 + i] + (mu[j] - m[j]) * (mu[i] - m[i]))
  const v = target.logDensity(tensor(m))
  const logNorm = typeof v === 'number' ? v : toFlat(v as Tensor)[0]
  return logNorm - 0.5 * tr + family.entropy(lambda)
}

function numericGrad(f: (l: number[]) => number, l: number[]) {
  return l.map((_, i) => {
    const up = [...l]
    const dn = [...l]
    up[i] += 1e-6
    dn[i] -= 1e-6
    return (f(up) - f(dn)) / 2e-6
  })
}

describe('families', () => {
  it.each([
    ['mean-field', meanFieldGaussian(2), [0.3, -0.2, 0.1, -0.4]],
    ['full-rank', fullRankGaussian(2), [0.3, -0.2, 0.1, 0.5, -0.4]],
  ] as [string, GaussianFamily, number[]][])('%s: score and path kernels match finite differences', (_, fam, l) => {
    const L = Float64Array.from(l)
    const eps = Float64Array.of(0.7, -1.1)
    const x = fam.kernels.transform(L, eps)
    const score = fam.kernels.score(L, eps)
    numericGrad((ll) => fam.logDensity(ll, x), l).forEach((g, i) => expect(score[i]).toBeCloseTo(g, 5))
    // Path: f(x) = a·x, ∇f = a.
    const a = Float64Array.of(0.4, -1.3)
    const path = fam.kernels.pathGrad(L, eps, a)
    const f = (ll: number[]) => {
      const y = fam.kernels.transform(Float64Array.from(ll), eps)
      return a[0] * y[0] + a[1] * y[1]
    }
    numericGrad(f, l).forEach((g, i) => expect(path[i]).toBeCloseTo(g, 5))
    const hg = fam.kernels.entropyGrad(L)
    numericGrad((ll) => fam.entropy(ll), l).forEach((g, i) => expect(hg[i]).toBeCloseTo(g, 5))
    // log q integrates to the right entropy (Monte Carlo).
    const flat = { kind: 'log-density' as const, dim: 2, normalised: false, logDensity: () => 0 }
    expect(elbo(stream(1), flat, fam, l, { samples: 20000 }).value).toBeCloseTo(fam.entropy(l), 1)
  })
})

describe('ELBO estimators', () => {
  const fam = meanFieldGaussian(2)
  const l = [0.2, 0.1, -0.3, 0.2]
  const truth = numericGrad((ll) => exactElbo(fam, ll), l)

  it('elboGradient returns a gradient per parameter, deterministic in its stream', () => {
    const a = elboGradient(stream(9), target, fam, l, { samples: 3 })
    const b = elboGradient(stream(9), target, fam, l, { samples: 3 })
    expect(a.grad.shape).toEqual([4])
    expect(a.draws.shape).toEqual([3, 2])
    expect(toFlat(a.grad)).toEqual(toFlat(b.grad))
    expect(() =>
      elboGradient(stream(1), target, fam, l, { estimator: 'score', samples: 2, baseline: 'control-variate' }),
    ).toThrow()
  })

  it('elbo estimate matches the exact value', () => {
    const e = elbo(stream(1), target, fam, l, { samples: 20000 })
    expect(Math.abs(e.value - exactElbo(fam, l))).toBeLessThan(4 * e.standardError + 1e-3)
  })

  it.each([
    ['reparameterisation', 'none', 1],
    ['score', 'none', 10],
    ['score', 'leave-one-out', 10],
    ['score', 'control-variate', 10],
  ] as const)('%s (%s) is unbiased', (estimator, baseline, samples) => {
    const v = gradientVariance(stream(2), target, fam, l, { estimator, baseline, samples, repeats: 3000 })
    const mean = toFlat(v.mean)
    const se = toFlat(v.variance).map((x) => Math.sqrt(x / v.repeats))
    truth.forEach((g, i) => expect(Math.abs(mean[i] - g)).toBeLessThan(5 * se[i] + 0.02))
  })

  it('reparameterisation has lower variance than the score function; baselines help', () => {
    const tv = (opts: Parameters<typeof gradientVariance>[4]) =>
      gradientVariance(stream(3), target, fam, l, { ...opts, samples: 10, repeats: 300 }).totalVariance
    const reparam = tv({ estimator: 'reparameterisation' })
    const plain = tv({ estimator: 'score', baseline: 'none' })
    const loo = tv({ estimator: 'score', baseline: 'leave-one-out' })
    expect(reparam).toBeLessThan(loo)
    expect(loo).toBeLessThan(plain)
  })

  it('full-rank gradients are unbiased', () => {
    const fr = fullRankGaussian(2)
    const lf = [0.2, 0.1, -0.3, 0.4, 0.2]
    const truthF = numericGrad((ll) => exactElbo(fr, ll), lf)
    const v = gradientVariance(stream(4), target, fr, lf, { samples: 4, repeats: 2000 })
    const mean = toFlat(v.mean)
    truthF.forEach((g, i) => expect(mean[i]).toBeCloseTo(g, 1))
  })
})

describe('bbvi', () => {
  it('mean field finds the mean and the conditional precisions (under-covers)', () => {
    const s = run(bbvi(target, { stepSize: (t) => 0.05 / (1 + t / 100), samples: 4 }), {}, 5000, {
      stream: stream(1),
    })
    const mu = toFlat(s.mean)
    const C = toFlat(s.covariance)
    expect(mu[0]).toBeCloseTo(1, 1)
    expect(mu[1]).toBeCloseTo(-0.5, 1)
    // KL(q‖p) optimum: σᵢ² = 1/Λᵢᵢ, below the marginal variance Σᵢᵢ.
    expect(C[0]).toBeCloseTo(1 / P[0], 1)
    expect(C[3]).toBeCloseTo(1 / P[3], 1)
    expect(C[0]).toBeLessThan(S[0][0])
    expect(s.objective).toBe('KL(q‖p)')
  })

  it('full rank recovers the covariance', () => {
    const s = run(bbvi(target, { family: 'full-rank', stepSize: (t) => 0.05 / (1 + t / 100), samples: 8 }), {}, 5000, {
      stream: stream(2),
    })
    const C = toFlat(s.covariance)
    expect(C[1]).toBeCloseTo(0.6, 1)
    expect(C[3]).toBeCloseTo(0.8, 1)
  })

  it('score estimator also converges', () => {
    const s = run(bbvi(target, { estimator: 'score', samples: 20, stepSize: 0.03 }), {}, 1500, {
      stream: stream(3),
    })
    expect(toFlat(s.mean)[0]).toBeCloseTo(1, 0)
  })

  it('satisfies the Algorithm protocol, mean field and full rank, from a given start', () => {
    const record = { elbo: (st: { elbo: number }) => st.elbo }
    checkProtocol(bbvi(target, { stepSize: 0.05 }), {}, { steps: 12, random: true, record })
    checkProtocol(
      bbvi(target, { family: 'full-rank', stepSize: 0.05 }),
      { mean0: [0.5, 0.5], sd0: 2 },
      {
        steps: 12,
        random: true,
        record,
      },
    )
    checkProtocol(bbvi(target, { estimator: 'score', samples: 5 }), {}, { steps: 8, random: true })
    expect(trace(bbvi(target), {}, 3, { stream: stream(1) }).meta.steps).toBe(3)
  })
})
