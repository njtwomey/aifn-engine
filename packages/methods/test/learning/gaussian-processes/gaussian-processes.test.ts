import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import {
  gpClassifier,
  fitGp,
  fitSparseGp,
  gaussianProcessRegressor,
  gpPosterior,
  kernelLogVector,
  laplaceMode,
  logMarginalLikelihood,
  logMarginalLikelihoodGradient,
  samplePrior,
  sparseGp,
} from 'aifn-methods/learning/gaussian-processes'
import {
  gram,
  kernelDiagonal,
  kernelProfile,
  kernelFromLog,
  linearKernel,
  matern,
  periodic,
  polynomial,
  productKernel,
  rationalQuadratic,
  rbf,
  sumKernel,
  white,
  type Kernel,
} from 'aifn-compute/learning/kernels'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, sum, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../protocol'
import { fixture } from '../../fixtures'

type Grams = Record<string, { params: Record<string, number | number[]>; xy: number[][]; xx: number[][] }>
type Fixture = {
  x: number[][]
  y: number[][]
  grams: Grams
  regression: {
    x: number[]
    y: number[]
    xs: number[]
    noise: number
    variance: number
    lengthscale: number
    mean: number[]
    cov: number[][]
    lml: number
    log_gradient: number[]
    lml_white: number
    fitted: { variance: number; lengthscale: number; noise: number; lml: number }
  }
  classification: {
    x: number[]
    y: number[]
    xs: number[]
    variance: number
    lengthscale: number
    mode: number[]
    lml: number
    proba: number[]
    probit: Record<'laplace' | 'ep', { mode: number[]; lml: number; mean: number[]; var: number[]; proba: number[] }>
  }
}

const F = fixture<Fixture>('learning/gaussian-processes')

function close(actual: number[] | number[][], expected: number[] | number[][], tol: number) {
  const a = actual.flat()
  const e = expected.flat()
  expect(a.length).toBe(e.length)
  a.forEach((v, i) => expect(Math.abs(v - e[i])).toBeLessThan(tol * (1 + Math.abs(e[i]))))
}

const asParam = (v: number | number[]) => (typeof v === 'number' ? v : tensor(v))

function kernelFor(name: string, p: Record<string, number | number[]>): Kernel {
  const lengthscale = p.lengthscale === undefined ? undefined : asParam(p.lengthscale)
  const variance = p.variance as number
  switch (name) {
    case 'rbf':
    case 'rbf_ard':
      return rbf({ lengthscale, variance })
    case 'matern12':
      return matern(0.5, { lengthscale, variance })
    case 'matern32':
      return matern(1.5, { lengthscale, variance })
    case 'matern52':
      return matern(2.5, { lengthscale, variance })
    case 'rq':
      return rationalQuadratic({ lengthscale, variance, alpha: p.alpha as number })
    case 'periodic':
      return periodic({ lengthscale, variance, period: p.period as number })
    case 'linear':
      return linearKernel({ variance, bias: p.bias as number })
    case 'polynomial':
      return polynomial(3, { variance, bias: p.bias as number })
  }
  throw new Error(name)
}

describe('kernels', () => {
  const x = tensor(F.x)
  const y = tensor(F.y)
  for (const [name, g] of Object.entries(F.grams)) {
    it(`${name} matches scikit-learn`, () => {
      const k = kernelFor(name, g.params)
      close(toRows(gram(k, x, y)), g.xy, 1e-10)
      close(toRows(gram(k, x)), g.xx, 1e-10)
      close(
        toFlat(kernelDiagonal(k, x)),
        g.xx.map((r, i) => r[i]),
        1e-10,
      )
    })
  }

  it('white noise is on the diagonal of a set against itself only', () => {
    const k = sumKernel(rbf(), white({ variance: 0.5 }))
    const K = toRows(gram(k, x))
    const cross = toRows(gram(k, x, x))
    expect(K[0][0] - cross[0][0]).toBeCloseTo(0.5, 12)
    expect(K[0][1]).toBeCloseTo(cross[0][1], 12)
  })

  it('sums and products combine elementwise and rebuild from their parameter trees', () => {
    const a = rbf({ lengthscale: 0.5 })
    const b = periodic({ period: 2 })
    const p = productKernel(a, b)
    const expected = toFlat(gram(a, x)).map((v, i) => v * toFlat(gram(b, x))[i])
    close(toFlat(gram(p, x)), expected, 1e-12)
    const pv = kernelLogVector(p)
    expect(pv.names).toEqual([
      'terms[0].lengthscale',
      'terms[0].variance',
      'terms[1].lengthscale',
      'terms[1].period',
      'terms[1].variance',
    ])
    const again = kernelFromLog(p, pv.unravel(pv.vector))
    close(toFlat(gram(again, x)), toFlat(gram(p, x)), 1e-14)
  })

  it('gradients in the hyperparameters match finite differences, including Matérn on the diagonal', () => {
    for (const make of [
      (l: number) => matern(0.5, { lengthscale: l }),
      (l: number) => matern(2.5, { lengthscale: l }),
      (l: number) => rbf({ lengthscale: l }),
    ]) {
      const report = gradCheck((l: number) => sum(gram(make(l as never), x)), 0.7)
      expect(report.ok).toBe(true)
      expect(Number.isFinite(grad((l: number) => sum(gram(make(l as never), x)))(0.7) as number)).toBe(true)
    }
  })

  it('kernelProfile gives k(τ, 0)', () => {
    const lags = linspace(-2, 2, 5)
    close(
      toFlat(kernelProfile(rbf({ lengthscale: 1 }), lags)),
      [-2, -1, 0, 1, 2].map((t) => Math.exp((-t * t) / 2)),
      1e-14,
    )
  })
})

describe('gp regression', () => {
  const R = F.regression
  const k = rbf({ lengthscale: R.lengthscale, variance: R.variance })
  const x = tensor(R.x)
  const y = tensor(R.y)
  const xs = tensor(R.xs)

  it('posterior mean, covariance and log marginal likelihood match scikit-learn', () => {
    const post = gpPosterior(k, x, y, { noiseVariance: R.noise })
    const p = post.predict(xs, { full: true })
    close(toFlat(p.mean), R.mean, 1e-8)
    close(toRows(p.covariance!), R.cov, 1e-8)
    close(
      toFlat(p.variance),
      R.cov.map((r, i) => r[i]),
      1e-8,
    )
    expect(post.logMarginal.value).toBeCloseTo(R.lml, 8)
    expect(post.jitter).toBe(0)
    expect(logMarginalLikelihood(k, x, y, { noiseVariance: R.noise }).value).toBeCloseTo(R.lml, 8)
  })

  it('the log-marginal gradient matches scikit-learn', () => {
    const g = logMarginalLikelihoodGradient(k, x, y, { noiseVariance: R.noise })
    expect(g.value).toBeCloseTo(R.lml_white, 8)
    // Our order: lengthscale, variance, noise; scikit-learn's: constant (variance), lengthscale, noise.
    const lg = Array.from(g.logGradient)
    close([lg[1], lg[0], lg[2]], R.log_gradient, 1e-6)
    expect(g.kernel.lengthscale as number).toBeCloseTo(lg[0] / R.lengthscale, 10)
    const g0 = logMarginalLikelihoodGradient(k, x, y, { noiseVariance: 0 })
    expect(Number.isFinite(g0.noiseVariance)).toBe(true)
  })

  it('fitGp reaches the maximum scikit-learn finds', () => {
    const fit = fitGp(k, x, y, { noiseVariance: 0.1, restarts: 3, stream: stream('fit') })
    expect(fit.logMarginal).toBeGreaterThan(R.fitted.lml - 1e-4)
    expect(fit.kernel.params.lengthscale as number).toBeCloseTo(R.fitted.lengthscale, 2)
    expect(fit.noiseVariance).toBeCloseTo(R.fitted.noise, 3)
  })

  it('draws have the prior covariance and are deterministic', () => {
    const grid = linspace(0, 1, 5)
    const a = samplePrior(stream('p'), k, grid, 4000)
    const b = samplePrior(stream('p'), k, grid, 3)
    expect(toRows(b.draws)[0]).toEqual(toRows(a.draws)[0])
    const rows = toRows(a.draws)
    const K = toRows(gram(k, grid))
    const c01 = rows.reduce((s, r) => s + r[0] * r[3], 0) / rows.length
    expect(Math.abs(c01 - K[0][3])).toBeLessThan(0.1)
  })

  it('interpolates without noise, reporting the jitter it needed', () => {
    const post = gpPosterior(rbf({ lengthscale: 3 }), linspace(0, 1, 30), linspace(0, 1, 30))
    expect(post.jitter).toBeGreaterThan(0)
    expect(post.failed).toBe(false)
  })

  it('the estimator predicts a Gaussian and samples', () => {
    const model = gaussianProcessRegressor({ kernel: k, noiseVariance: R.noise }).fit(dataset(x, y))
    close(toFlat(model.decide(xs)), R.mean, 1e-8)
    const draws = model.sample(stream('s'), xs, 2)
    expect(draws.shape).toEqual([2, R.xs.length])
    expect(toFlat(model.expect(xs))[0]).toBeCloseTo(R.mean[0], 6)
  })
})

describe('sparse gp', () => {
  const R = F.regression
  const k = rbf({ lengthscale: R.lengthscale, variance: R.variance })
  const x = tensor(R.x)
  const y = tensor(R.y)
  const xs = tensor(R.xs)

  it('reproduces the exact GP when Z = X (FITC and VFE)', () => {
    for (const method of ['fitc', 'vfe'] as const) {
      const s = sparseGp(k, x, y, x, { method, noiseVariance: R.noise, relativeJitter: 1e-12 })
      expect(s.logMarginal).toBeCloseTo(R.lml, 4)
      close(toFlat(s.predict(xs).mean), R.mean, 1e-4)
    }
  })

  it('the VFE bound lies below the exact log marginal likelihood', () => {
    const z = linspace(0, 5, 4)
    const vfe = sparseGp(k, x, y, z, { method: 'vfe', noiseVariance: R.noise })
    const dtc = sparseGp(k, x, y, z, { method: 'dtc', noiseVariance: R.noise })
    expect(vfe.logMarginal).toBeLessThan(R.lml)
    expect(vfe.logMarginal).toBeLessThan(dtc.logMarginal)
    expect(dtc.terms.trace).toBe(0)
  })

  it('fitting inducing inputs raises the bound', () => {
    const z = linspace(0, 1, 4)
    const before = sparseGp(k, x, y, z, { noiseVariance: R.noise }).logMarginal
    const fit = fitSparseGp(k, x, y, z, { noiseVariance: R.noise, maxSteps: 50 })
    expect(fit.model.logMarginal).toBeGreaterThan(before)
  })
})

describe('gp classification', () => {
  const C = F.classification
  const k = rbf({ lengthscale: C.lengthscale, variance: C.variance })
  const x = tensor(C.x)
  const y = tensor(C.y)

  it('the Laplace mode and log marginal likelihood match scikit-learn', () => {
    const model = gpClassifier({ kernel: k }).fit(dataset(x, y))
    close(toFlat(model.mode), C.mode, 1e-6)
    expect(model.logMarginal).toBeCloseTo(C.lml, 6)
    close(toFlat(model.expect(tensor(C.xs))), C.proba, 2e-3)
    expect(model.converged).toBe(true)
  })

  for (const method of ['laplace', 'ep'] as const)
    it(`probit ${method}: mode, evidence, latent predictive and probabilities match R&W's algorithms in numpy`, () => {
      const want = C.probit[method]
      const model = gpClassifier({ kernel: k, method, likelihood: 'probit', tolerance: 1e-12 }).fit(dataset(x, y))
      expect(model.converged).toBe(true)
      close(toFlat(model.mode), want.mode, 1e-7)
      expect(model.logMarginal).toBeCloseTo(want.lml, 8)
      const latent = model.latent(tensor(C.xs))
      close(toFlat(latent.mean), want.mean, 1e-7)
      close(toFlat(latent.variance), want.var, 1e-7)
      close(toFlat(model.expect(tensor(C.xs))), want.proba, 1e-8)
    })

  it('satisfies the trace protocol', () => {
    const alg = laplaceMode({ K: gram(k, x) as Tensor, labels: y, likelihood: 'probit' })
    expectProtocol(alg, {}, { n: 6 })
  })
})
