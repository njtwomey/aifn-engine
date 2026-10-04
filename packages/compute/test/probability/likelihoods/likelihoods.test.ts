import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  binomialFamily,
  checkLink,
  family,
  gammaFamily,
  gaussianFamily,
  inverseGaussianFamily,
  likelihood,
  link,
  negativeBinomialFamily,
  ordinalLikelihood,
  poissonFamily,
  type Family,
  type LinkName,
  type OrdinalLinkName,
  type OrdinalModel,
} from 'aifn-compute/probability/likelihoods'
import { orderedBijector } from 'aifn-compute/probability/bijectors'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { sum, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Case = {
  y: number[]
  mu: number[]
  weights: number[]
  logProb: number[]
  variance: number[]
  unitDeviance: number[]
}
const F = fixture<{
  phi: number
  theta: number
  gaussian: Case
  gamma: Case
  inverseGaussian: Case
  poisson: Case
  negativeBinomial: Case
  binomial: Case
  ordinal: { eta: number[]; thresholds: number[]; probabilities: Record<string, number[][]> }
}>('probability/likelihoods')

const flat = (v: Value) => (typeof v === 'number' ? [v] : toFlat(v as Tensor))
const num = (v: Value) => flat(v)[0]
const close = (a: Value, b: number[], tol: number) => {
  const x = flat(a)
  expect(x.length).toBe(b.length)
  x.forEach((v, i) => expect(Math.abs(v - b[i]) / Math.max(1, Math.abs(b[i]))).toBeLessThan(tol))
}

const families: [keyof typeof F, () => Family, number][] = [
  ['gaussian', gaussianFamily, F.phi],
  ['gamma', gammaFamily, F.phi],
  ['inverseGaussian', inverseGaussianFamily, F.phi],
  ['poisson', poissonFamily, 1],
  ['negativeBinomial', () => negativeBinomialFamily(F.theta), 1],
  ['binomial', binomialFamily, 1],
]

describe('exponential-dispersion families against scipy.stats', () => {
  it.each(families)('%s: logProb, variance and unit deviance', (key, make, phi) => {
    const c = F[key] as Case
    const fam = make()
    const [y, mu, w] = [tensor(c.y), tensor(c.mu), tensor(c.weights)]
    close(fam.logProb(y, mu, phi, w), c.logProb, 1e-12)
    close(fam.variance(mu), c.variance, 1e-12)
    close(fam.unitDeviance(y, mu), c.unitDeviance, 1e-10)
    expect(fam.logLikelihood(y, mu, phi, w)).toBeCloseTo(
      c.logProb.reduce((a, b) => a + b, 0),
      10,
    )
    expect(fam.validMean(mu)).toBe(true)
  })

  it('family() builds each family by name; canonical links are the textbook ones', () => {
    const canonical: Record<string, LinkName> = {
      gaussian: 'identity',
      binomial: 'logit',
      poisson: 'log',
      gamma: 'inverse',
      'inverse-gaussian': 'inverse-squared',
    }
    for (const [name, l] of Object.entries(canonical)) {
      const fam = family(name as never)
      expect(fam.name).toBe(name)
      expect(fam.canonicalLink).toBe(l)
    }
    expect(family('negative-binomial', { theta: 3 }).params.theta).toBe(3)
    expect(() => negativeBinomialFamily(0)).toThrow()
  })

  it('predictive distributions have the family mean and variance', () => {
    const mu = tensor([0.8, 2.5])
    const phi = 0.4
    for (const [, make] of families.filter(([k]) => k !== 'binomial')) {
      const fam = make()
      const d = fam.predictive(mu, phi)
      close(d.mean!() as Value, [0.8, 2.5], 1e-12)
      const v = flat(fam.variance(mu)).map((x) => x * (fam.dispersion ?? phi))
      close(d.variance!() as Value, v, 1e-12)
      expect(flat(d.sample(stream(1), { shape: [3] }) as Value).length).toBe(6)
    }
  })
})

describe('links and the likelihood', () => {
  it.each(['identity', 'log', 'logit', 'probit', 'cloglog', 'inverse', 'inverse-squared', 'sqrt'] as LinkName[])(
    '%s: inverse ∘ link = id and derivative = dμ/dη',
    (name) => {
      const g = link(name)
      const mu = tensor([0.2, 0.45, 0.7])
      close(g.inverse(g.link(mu)), [0.2, 0.45, 0.7], 1e-12)
      const eta = toFlat(g.link(mu) as Tensor)
      const d = flat(g.derivative(tensor(eta)))
      eta.forEach((e, i) => {
        const h = 1e-6
        const fd = (num(g.inverse(e + h)) - num(g.inverse(e - h))) / (2 * h)
        expect(d[i]).toBeCloseTo(fd, 6)
      })
    },
  )

  it.each(families)(
    '%s: the score is ∂ logLik/∂η (autodiff), under the canonical and a non-canonical link',
    (key, make, phi) => {
      const c = F[key] as Case
      const fam = make()
      const links: LinkName[] = [fam.canonicalLink, fam.name === 'binomial' ? 'probit' : 'log']
      for (const l of links) {
        if (l === 'log' && fam.name === 'gaussian') continue
        const lik = likelihood(fam, l)
        const y = tensor(c.y)
        const w = tensor(c.weights)
        const eta = lik.link.link(tensor(c.mu))
        const g = grad((e: Value) => sum(lik.logLik(y, e, { dispersion: phi, weights: w })))(eta)
        close(lik.score(y, eta, { dispersion: phi, weights: w }), flat(g as Value), 1e-8)
        close(lik.mean(eta), c.mu, 1e-12)
        close(lik.unitDeviance(y, eta), c.unitDeviance, 1e-10)
      }
    },
  )
})

describe('family–link pairs', () => {
  it('each family takes its canonical and default links, and checkLink rejects the others by name', () => {
    for (const name of ['gaussian', 'binomial', 'poisson', 'gamma', 'inverse-gaussian', 'negative-binomial'] as const) {
      const fam = family(name)
      expect(fam.links).toContain(fam.canonicalLink)
      expect(fam.links).toContain(fam.defaultLink)
      for (const l of fam.links) expect(checkLink(fam, l).name).toBe(l)
    }
    expect(() => checkLink(family('poisson'), 'logit', 'gam')).toThrow(
      'gam: the poisson family does not take the logit link; use one of log, identity, sqrt',
    )
    expect(() => likelihood(family('binomial'), 'identity')).toThrow(/does not take the identity link/)
    expect(() => likelihood(family('gamma'), 'probit')).toThrow(DomainError)
  })
})

describe('ordinal likelihoods', () => {
  const O = F.ordinal
  const cases: [OrdinalModel, OrdinalLinkName][] = [
    ['cumulative', 'logit'],
    ['cumulative', 'probit'],
    ['cumulative', 'cloglog'],
    ['continuation-ratio', 'logit'],
    ['continuation-ratio', 'probit'],
    ['continuation-ratio', 'cloglog'],
    ['adjacent-category', 'logit'],
  ]
  it.each(cases)('%s %s: class probabilities match the defining formula; rows sum to 1', (model, l) => {
    const lik = ordinalLikelihood(model, l)
    const p = lik.probabilities(tensor(O.eta), tensor(O.thresholds)) as Tensor
    expect(p.shape).toEqual([4, 4])
    close(p, O.probabilities[`${model}-${l}`].flat(), 1e-12)
    toRows(p).forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 13))
    const y = [0, 3, 1, 2]
    const ll = flat(lik.logLik(y, tensor(O.eta), tensor(O.thresholds)))
    y.forEach((k, i) => expect(ll[i]).toBeCloseTo(Math.log(O.probabilities[`${model}-${l}`][i][k]), 12))
  })

  it('proportional odds: the log cumulative odds are θ_k − η', () => {
    const p = toRows(
      ordinalLikelihood('cumulative', 'logit').probabilities(tensor([0.3]), tensor([-1, 0.5])) as Tensor,
    )[0]
    expect(Math.log(p[0] / (1 - p[0]))).toBeCloseTo(-1 - 0.3, 12)
    expect(Math.log((p[0] + p[1]) / p[2])).toBeCloseTo(0.5 - 0.3, 12)
  })

  it('log-likelihoods differentiate through ordered thresholds', () => {
    const lik = ordinalLikelihood('cumulative', 'probit')
    const ordered = orderedBijector()
    const f = (raw: Value) => sum(lik.logLik([0, 2, 1], tensor([0.1, -0.4, 0.9]), ordered.forward(raw)))
    const raw = tensor([-0.5, 0.2])
    const g = flat(grad(f)(raw) as Value)
    const h = 1e-6
    for (let i = 0; i < 2; i++) {
      const up = [-0.5, 0.2]
      const down = [-0.5, 0.2]
      up[i] += h
      down[i] -= h
      expect(g[i]).toBeCloseTo((num(f(tensor(up))) - num(f(tensor(down)))) / (2 * h), 6)
    }
  })

  it('rejects the adjacent-category model with a non-logit link', () => {
    expect(() => ordinalLikelihood('adjacent-category', 'probit')).toThrow()
  })
})
