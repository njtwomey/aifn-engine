import { describe, expect, it } from 'vitest'
import { deviance, irls } from 'aifn-methods/learning/generalised'
import { family, link, poissonFamily, type FamilyName, type LinkName } from 'aifn-compute/probability/likelihoods'
import {
  glm,
  multinomialLogisticRegression,
  negativeBinomialAlternation,
  negativeBinomialRegression,
  softmaxNewton,
} from 'aifn-methods/learning/generalised/glm'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../../protocol'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { fixture } from '../../../fixtures'

type Fit = {
  family: FamilyName
  link: LinkName
  theta: number
  y: number[]
  weights: number[] | null
  offset: number[] | null
  coef: number[]
  se: number[]
  p: number[]
  dispersion: number
  deviance: number
  pearson: number[]
  deviance_residuals: number[]
}
type Fixture = {
  x: number[][]
  fits: Record<string, Fit>
  logistic: { y: number[]; coef: number[] }
  nb_ml: { coef: number[]; theta: number }
  multinomial: { y: number[]; coef: number[][]; se: number[][] }
}
const F = fixture<Fixture>('learning/generalised/glm')
const x = tensor(F.x)

function close(actual: number[], expected: number[], tol: number) {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol * (1 + Math.abs(expected[i]))))
}

describe('links', () => {
  it('inverse undoes link and derivative is dμ/dη', () => {
    for (const name of [
      'identity',
      'log',
      'logit',
      'probit',
      'cloglog',
      'inverse',
      'inverse-squared',
      'sqrt',
    ] as const) {
      const l = link(name)
      const eta = name === 'inverse' || name === 'inverse-squared' || name === 'sqrt' ? 0.7 : -0.4
      expect(l.link(l.inverse(eta) as number) as number).toBeCloseTo(eta, 10)
      const report = gradCheck((e: number) => l.inverse(e), eta)
      expect(report.ok).toBe(true)
      const h = 1e-6
      const fd = ((l.inverse(eta + h) as number) - (l.inverse(eta - h) as number)) / (2 * h)
      expect(l.derivative(eta) as number).toBeCloseTo(fd, 6)
    }
  })
})

describe('glm', () => {
  for (const [key, f] of Object.entries(F.fits)) {
    it(`${key} (${f.family}, ${f.link}) matches the maximum-likelihood reference`, () => {
      const fam = family(f.family, { theta: f.theta })
      const model = glm({ family: fam, link: f.link, tolerance: 1e-12, maxSteps: 100 }).fit({
        x,
        y: tensor(f.y),
        weights: f.weights ? tensor(f.weights) : undefined,
        offset: f.offset ? tensor(f.offset) : undefined,
      })
      expect(model.converged).toBe(true)
      close(toFlat(model.coefficients), f.coef, 1e-6)
      close(toFlat(model.standardErrors), f.se, 1e-5)
      close(toFlat(model.pValues), f.p, 1e-4)
      expect(model.dispersion).toBeCloseTo(f.dispersion, 6)
      expect(model.deviance).toBeCloseTo(f.deviance, 6)
      close(toFlat(model.residuals('pearson')), f.pearson, 1e-5)
      close(toFlat(model.residuals('deviance')), f.deviance_residuals, 1e-5)
      expect(model.nullDeviance).toBeGreaterThanOrEqual(model.deviance - 1e-9)
    })
  }

  it('the Bernoulli GLM agrees with scikit-learn logistic regression', () => {
    const m = glm({ family: family('binomial'), tolerance: 1e-12 }).fit({ x, y: tensor(F.logistic.y) })
    close(toFlat(m.coefficients), F.logistic.coef, 1e-5)
  })

  it('predicts, samples and keeps the IRLS trace', () => {
    const f = F.fits.poisson
    const model = glm({ family: poissonFamily() }).fit({ x, y: tensor(f.y), offset: tensor(f.offset!) })
    const draws = model.sample(stream('glm'), x, 3)
    expect(draws.shape).toEqual([3, F.x.length])
    expect(model.training.series.deviance.shape[0]).toBe(model.training.steps.length)
    expect(toFlat(model.expect(x))[0]).toBeCloseTo(Math.exp(toFlat(model.forward(x))[0]), 6)
  })

  it('satisfies the trace protocol', () => {
    const f = F.fits.gamma
    const n = F.x.length
    const design = fromData(Float64Array.from(F.x.flatMap((r) => [...r, 1])), [n, 3])
    const alg = irls({ design, y: tensor(f.y), family: family('gamma'), link: link('log') })
    const long = trace(alg, {}, 5)
    expect(deviance(family('gamma'), tensor(f.y), long.final.mu)).toBeCloseTo(long.final.deviance, 10)
    expectProtocol(alg, {}, { n: 5, record: { deviance: (s) => s.deviance } })
  })
})

describe('negative binomial with θ estimated', () => {
  it('reaches the joint maximum-likelihood estimate', () => {
    const m = negativeBinomialRegression().fit({ x, y: tensor(F.fits['negative-binomial'].y) })
    expect(m.family.params.theta).toBeCloseTo(F.nb_ml.theta, 3)
    close(toFlat(m.coefficients), F.nb_ml.coef, 1e-4)
  })
})

describe('multinomial logistic regression', () => {
  it('contrasts against class 0 match the maximum-likelihood reference with standard errors', () => {
    const m = multinomialLogisticRegression({ tolerance: 1e-14 }).fit(dataset(x, tensor(F.multinomial.y)))
    close(toRows(m.contrasts.coefficients).flat(), F.multinomial.coef.flat(), 1e-5)
    close(toRows(m.contrasts.standardErrors).flat(), F.multinomial.se.flat(), 1e-4)
  })
})

describe('trace protocol', () => {
  it('the negative binomial alternation and the softmax Newton method follow it', () => {
    const y = tensor(F.fits['negative-binomial'].y)
    expectProtocol(negativeBinomialAlternation({ x, y }), undefined, { n: 4, record: { theta: (s) => s.theta } })
    const n = F.x.length
    const design = fromData(Float64Array.from(F.x.flatMap((r) => [...r, 1])), [n, 3])
    const labels = tensor(F.multinomial.y)
    const K = Math.max(...F.multinomial.y) + 1
    const alg = softmaxNewton({ design, labels, columns: K, l2: 0.1, intercept: true, tolerance: 1e-12 })
    expectProtocol(alg, {}, { n: 5 })
  })
})
