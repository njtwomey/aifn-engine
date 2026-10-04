import { describe, expect, it } from 'vitest'
import {
  fitGpClassifier,
  gpClassifier,
  gpClassifierEvidenceGradient,
  gpEp,
  gpEpLogMarginal,
  kernelLogVector,
  laplaceLogMarginal,
  type ClassificationLikelihood,
} from 'aifn-methods/learning/gaussian-processes'
import { moons } from 'aifn-methods/data/synthetic'
import { gram, kernelFromLog, rbf, type Kernel } from 'aifn-compute/learning/kernels'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

const DATA = moons(stream('gpc-test'), { n: 30, noise: 0.3 })
const X = DATA.x as Tensor
const Y = fromData(Float64Array.from(toFlat(DATA.y as Tensor)), [X.shape[0]])

// A 1-D problem with overlapping classes (R&W chapter 3 style): labels switch from 0 to 1 with a noisy boundary.
const X1 = tensor([[-2.1], [-1.6], [-1.2], [-0.9], [-0.5], [-0.2], [0.1], [0.3], [0.7], [1.1], [1.4], [1.9], [2.3]])
const Y1 = tensor([0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1])

function finiteDifference(kernel: Kernel, evidence: (k: Kernel) => number, h = 1e-5): number[] {
  const lv = kernelLogVector(kernel)
  return Array.from(lv.vector, (_, i) => {
    const at = (d: number) => {
      const v = Float64Array.from(lv.vector)
      v[i] += d
      return evidence(kernelFromLog(kernel, lv.unravel(v)))
    }
    return (at(h) - at(-h)) / (2 * h)
  })
}

describe('Laplace evidence gradient', () => {
  for (const likelihood of ['logistic', 'probit'] as ClassificationLikelihood[]) {
    it(`matches finite differences (${likelihood})`, () => {
      const k = rbf({ lengthscale: 0.6, variance: 3 })
      const g = gpClassifierEvidenceGradient(k, X, Y, { likelihood })
      expect(g.value).toBeCloseTo(laplaceLogMarginal(k, X, Y, { likelihood }), 10)
      const fd = finiteDifference(k, (kk) => laplaceLogMarginal(kk, X, Y, { likelihood }))
      fd.forEach((d, i) => expect(Math.abs(g.logGradient[i] - d)).toBeLessThan(1e-5 * (1 + Math.abs(d))))
      // The implicit term matters: it is not small here.
      expect(Math.max(...g.logGradient.map(Math.abs))).toBeGreaterThan(0.1)
      // ∂/∂θ = (∂/∂ log θ) / θ.
      expect(g.kernel.lengthscale as number).toBeCloseTo(g.logGradient[g.names.indexOf('lengthscale')] / 0.6, 10)
    })
  }
})

describe('EP GP classification', () => {
  const k = rbf({ lengthscale: 0.8, variance: 4 })

  it('satisfies the trace protocol', () => {
    expectProtocol(gpEp({ K: gram(k, X1) as Tensor, labels: Y1 }), {}, { n: 20 })
  })

  it('is exact for one point: log Z = log Φ(0) and the site gives the true posterior moments', () => {
    const final = run(gpEp({ K: tensor([[2]]), labels: tensor([1]) }), {}, 10)
    expect(final.converged).toBe(true)
    expect(final.logEvidence).toBeCloseTo(Math.log(0.5), 12)
    // p(f | y=1) ∝ N(f; 0, 2) Φ(f): mean = 2 φ(0)/(Φ(0)√3), from the truncated-normal identities.
    const mean = (2 * (1 / Math.sqrt(2 * Math.PI))) / (0.5 * Math.sqrt(3))
    expect(toFlat(final.mean)[0]).toBeCloseTo(mean, 10)
  })

  it('its evidence equals Π Z̃ᵢ · N(μ̃; 0, K + Σ̃), the normaliser of prior × sites', () => {
    const K = gram(k, X1) as Tensor
    const final = run(gpEp({ K, labels: Y1, tolerance: 1e-12 }), {}, 2000)
    expect(final.converged).toBe(true)
    const n = X1.shape[0]
    const tau = toFlat(final.sitePrecision)
    const nu = toFlat(final.siteShift)
    const S = toFlat(final.covariance)
    const mu = toFlat(final.mean)
    const Kf = toFlat(K)
    const phiLog = (z: number) => {
      // log Φ by erfc-free series is overkill here; the arguments are moderate.
      let t = 0
      const steps = 20000
      const lo = -12
      for (let i = 0; i < steps; i++) {
        const u = lo + ((z - lo) * (i + 0.5)) / steps
        t += Math.exp(-0.5 * u * u)
      }
      return Math.log((t * (z - lo)) / steps / Math.sqrt(2 * Math.PI))
    }
    let total = 0
    const mt = Float64Array.from(nu, (v, i) => v / tau[i])
    for (let i = 0; i < n; i++) {
      const ct = 1 / S[i * n + i] - tau[i]
      const cm = (mu[i] / S[i * n + i] - nu[i]) / ct
      const cv = 1 / ct
      const yi = 2 * toFlat(Y1)[i] - 1
      const logZhat = phiLog((yi * cm) / Math.sqrt(1 + cv))
      const v = cv + 1 / tau[i]
      total += logZhat + 0.5 * Math.log(2 * Math.PI * v) + (0.5 * (mt[i] - cm) ** 2) / v
    }
    // log N(μ̃; 0, K + Σ̃) by Cholesky.
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => Kf[i * n + j] + (i === j ? 1 / tau[i] : 0)),
    )
    const L = A.map(() => new Array<number>(n).fill(0))
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let s = A[i][j]
        for (let p = 0; p < j; p++) s -= L[i][p] * L[j][p]
        L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j]
      }
    const z = new Array<number>(n).fill(0)
    for (let i = 0; i < n; i++) {
      let s = mt[i]
      for (let p = 0; p < i; p++) s -= L[i][p] * z[p]
      z[i] = s / L[i][i]
    }
    let logN = -0.5 * n * Math.log(2 * Math.PI)
    for (let i = 0; i < n; i++) logN += -Math.log(L[i][i]) - 0.5 * z[i] * z[i]
    expect(final.logEvidence).toBeCloseTo(total + logN, 5)
  })

  it('decides like Laplace on easy data, and its evidence is at least Laplace’s (R&W §3.7)', () => {
    const easy = moons(stream('gpc-easy'), { n: 40, noise: 0.1 })
    const x = easy.x as Tensor
    const y = fromData(Float64Array.from(toFlat(easy.y as Tensor)), [x.shape[0]])
    const kk = rbf({ lengthscale: 0.5, variance: 5 })
    const lap = gpClassifier({ kernel: kk, likelihood: 'probit' }).fit(dataset(x, y))
    const ep = gpClassifier({ kernel: kk, method: 'ep' }).fit(dataset(x, y))
    expect(ep.converged).toBe(true)
    expect(ep.method).toBe('ep')
    const grid = fromData(
      Float64Array.from({ length: 400 }, (_, i) =>
        i % 2 === 0 ? -1.5 + (4 * ((i / 2) % 20)) / 19 : -1 + (2.5 * Math.floor(i / 40)) / 9,
      ),
      [200, 2],
    )
    const a = toFlat(lap.decide(grid))
    const b = toFlat(ep.decide(grid))
    const agree = a.filter((v, i) => v === b[i]).length / a.length
    expect(agree).toBeGreaterThan(0.95)
    expect(Array.from(toFlat(ep.decide(x)))).toEqual(Array.from(toFlat(y)))
    // Laplace underestimates the evidence (Kuss and Rasmussen 2005; R&W Fig. 3.9): EP ≥ Laplace, same probit model.
    for (const kernel of [kk, rbf({ lengthscale: 1, variance: 20 }), k]) {
      expect(gpEpLogMarginal(kernel, x, y)).toBeGreaterThanOrEqual(
        laplaceLogMarginal(kernel, x, y, { likelihood: 'probit' }),
      )
    }
    expect(gpEpLogMarginal(k, X1, Y1)).toBeGreaterThanOrEqual(laplaceLogMarginal(k, X1, Y1, { likelihood: 'probit' }))
    // EP's predictive probabilities are less extreme than Laplace's mode-based ones only through variance; both are
    // probabilities.
    toFlat(ep.expect(grid)).forEach((p) => expect(p >= 0 && p <= 1).toBe(true))
  })

  it('EP evidence gradient matches finite differences', () => {
    const g = gpClassifierEvidenceGradient(k, X1, Y1, { method: 'ep' })
    const fd = finiteDifference(k, (kk) => gpEpLogMarginal(kk, X1, Y1, { tolerance: 1e-12 }))
    fd.forEach((d, i) => expect(Math.abs(g.logGradient[i] - d)).toBeLessThan(1e-5 * (1 + Math.abs(d))))
  })

  it('runs on compute multivariate EP: evidence and gradient pinned to the former private loop (1e-8)', () => {
    // Values recorded from the private R&W Algorithm 3.5 loop before it was replaced by multivariateExpectationPropagation.
    expect(Math.abs(gpEpLogMarginal(k, X1, Y1, { tolerance: 1e-12 }) - -8.860735922948066)).toBeLessThan(1e-8)
    expect(Math.abs(gpEpLogMarginal(k, X, Y, { tolerance: 1e-12 }) - -15.056102726929986)).toBeLessThan(1e-8)
    const g = gpClassifierEvidenceGradient(k, X1, Y1, { method: 'ep' })
    expect(Math.abs(g.logGradient[0] - 0.8492033045910778)).toBeLessThan(1e-8)
    expect(Math.abs(g.logGradient[1] - -0.5740131416201419)).toBeLessThan(1e-8)
    const init = run(gpEp({ K: gram(k, X1) as Tensor, labels: Y1 }), {}, 0)
    expect(Number.isNaN(init.logEvidence)).toBe(true)
  })

  it('rejects the logistic link', () => {
    expect(() => gpClassifier({ kernel: k, method: 'ep', likelihood: 'logistic' })).toThrow(/probit/)
  })
})

describe('fitGpClassifier', () => {
  for (const method of ['laplace', 'ep'] as const) {
    it(`climbs the ${method} evidence by L-BFGS to a stationary point`, () => {
      const k = rbf({ lengthscale: 0.2, variance: 1 })
      const start = method === 'ep' ? gpEpLogMarginal(k, X, Y) : laplaceLogMarginal(k, X, Y)
      const fit = fitGpClassifier(k, X, Y, { method, maxIterations: 60 })
      expect(fit.logMarginal).toBeGreaterThan(start + 1)
      const again = method === 'ep' ? gpEpLogMarginal(fit.kernel, X, Y) : laplaceLogMarginal(fit.kernel, X, Y)
      expect(fit.logMarginal).toBeCloseTo(again, 5)
      const g = gpClassifierEvidenceGradient(fit.kernel, X, Y, { method })
      if (fit.converged) g.logGradient.forEach((v) => expect(Math.abs(v)).toBeLessThan(1e-3))
      // The recorded evidence rises along the path.
      const series = toFlat(fit.training.series.logMarginal)
      expect(series[series.length - 1]).toBeGreaterThan(series[0])
      expect(fit.names).toEqual(expect.arrayContaining(['lengthscale', 'variance']))
    })
  }

  it('a log hyperprior holds the signal variance near its start', () => {
    const k = rbf({ lengthscale: 0.5, variance: 2 })
    const free = fitGpClassifier(k, X, Y, { maxIterations: 40 })
    const held = fitGpClassifier(k, X, Y, { maxIterations: 40, logPriorScale: 0.25 })
    const dist = (f: typeof free) => Math.abs(Math.log(f.kernel.params.variance as number) - Math.log(2))
    expect(dist(held)).toBeLessThan(dist(free) + 1e-9)
  })
})
