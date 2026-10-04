/**
 * Smoke tests for aifn-compute/distributions: one check per family of exports. Reference values were computed with
 * scipy.stats (univariate families), scipy.stats.multivariate_normal / dirichlet / multinomial / wishart and
 * torch.distributions.kl_divergence, and are hard-coded here. Fuller fixture-driven tests are still to come.
 */

import { describe, expect, it } from 'vitest'
import * as D from 'aifn-compute/probability/distributions'
import * as B from 'aifn-compute/probability/bijectors'
import { stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import { grad } from 'aifn-compute/foundation/autodiff'
import { mean as tmean } from 'aifn-compute/foundation/tensor'
import { fromRows, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { checkGradient } from '../../foundation/tensor/check-gradient'

const n = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const close = (a: number, b: number, tol = 1e-9) => {
  if (Number.isNaN(b)) expect(a).toBeNaN()
  else expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)
}

// [family, x, logpdf/logpmf(x), cdf(x), ppf(0.7), mean, variance, entropy] from scipy.stats.
const SCIPY: [string, D.Univariate, number, number, number, number, number, number, number][] = [
  ['Normal', D.Normal(1, 2), 0.5, -1.643335713764618, 0.4012936743170763, 2.0488010254160813, 1, 4, 2.112085713764618],
  [
    'LogNormal',
    D.LogNormal(0.2, 0.7),
    2,
    -1.5035680572689705,
    0.7594380212470001,
    1.7631118755605766,
    1.5604901958326667,
    1.5397719762048734,
    1.2622635892659402,
  ],
  [
    'StudentT',
    D.StudentT(4, 1, 2),
    5,
    -3.406844384971535,
    0.9419417382415922,
    2.137298126099409,
    1,
    8,
    2.3749071974386116,
  ],
  [
    'Cauchy',
    D.Cauchy(0, 1.5),
    -1,
    -1.917919774082882,
    0.3128329581890012,
    1.0898137920080413,
    NaN,
    NaN,
    2.9364893550774553,
  ],
  [
    'Laplace',
    D.Laplace(1, 2),
    -1,
    -2.3862943611198904,
    0.18393972058572117,
    2.021651247531981,
    1,
    8,
    2.386294361119891,
  ],
  [
    'Logistic',
    D.Logistic(1, 2),
    2,
    -2.1413011489201588,
    0.6224593312018546,
    2.694595720774407,
    1,
    13.159472534785811,
    2.6931471805599454,
  ],
  ['Uniform', D.Uniform(-2, 3), 0.5, -1.6094379124341003, 0.5, 1.5, 0.5, 2.083333333333333, 1.6094379124341003],
  [
    'Exponential',
    D.Exponential(0.7),
    2,
    -1.7566749439387324,
    0.7534030360583935,
    1.7199611490370512,
    1.4285714285714286,
    2.0408163265306123,
    1.3566749439387324,
  ],
  [
    'Gamma',
    D.Gamma(2.5, 1.3),
    2,
    -1.1890514384642736,
    0.6080371084003665,
    2.332473070828809,
    1.923076923076923,
    1.4792899408284022,
    1.4675836450375634,
  ],
  [
    'InverseGamma',
    D.InverseGamma(3, 2),
    0.5,
    0.1588830833596716,
    0.2381033055535443,
    1.045054497051086,
    1,
    1,
    0.695157020726022,
  ],
  [
    'Beta',
    D.Beta(2, 3),
    0.3,
    0.5675839575845993,
    0.34829999999999994,
    0.5084047548725844,
    0.4,
    0.04,
    -0.2349066497880008,
  ],
  ['ChiSquare', D.ChiSquare(3), 2, -1.5723649429247, 0.42759329552912023, 3.6648707831703193, 3, 6, 2.0541199559354117],
  [
    'Weibull',
    D.Weibull(1.5, 2),
    2,
    -1.2876820724517808,
    0.6321205588285577,
    2.2634684588927163,
    1.805490585901867,
    1.5027611392557279,
    1.4800872940856251,
  ],
  [
    'Gumbel',
    D.Gumbel(1, 2),
    0.5,
    -1.7271725972476868,
    0.27692033409990896,
    3.0618608663174456,
    2.1544313298030655,
    6.579736267392906,
    2.270362845461478,
  ],
  [
    'VonMises',
    D.VonMises(0.5, 2),
    1,
    -0.906705484111556,
    0.7381922144185191,
    0.9092592364629697,
    0.5,
    NaN,
    1.2663212919642852,
  ],
  [
    'TruncatedNormal',
    D.TruncatedNormal(1, 2, -1, 4),
    2,
    -1.481596569642245,
    0.687903654148042,
    2.053587195278017,
    1.2903748943050524,
    1.6627400246295574,
    1.5749787701262732,
  ],
  ['Bernoulli', D.Bernoulli(0.3), 1, -1.2039728043259361, 1, 0, 0.3, 0.21, 0.6108643020548935],
  ['Binomial', D.Binomial(10, 0.3), 3, -1.321151277766889, 0.6496107184000002, 4, 3, 2.1, 1.779078784090063],
  ['Poisson', D.Poisson(2.5), 3, -1.5428872736055896, 0.7575761331330662, 3, 2.5, 2.5, 1.8307266079269722],
  [
    'Geometric',
    D.Geometric(0.3),
    3,
    -1.917322692203401,
    0.6569999999999999,
    4,
    3.3333333333333335,
    7.777777777777779,
    2.036214340182978,
  ],
  [
    'NegativeBinomial',
    D.NegativeBinomial(3, 0.4),
    5,
    -2.258477876728995,
    0.68460544,
    6,
    4.5,
    11.25,
    2.4880372470519614,
  ],
  [
    'Hypergeometric',
    D.Hypergeometric(20, 7, 12),
    5,
    -1.2515253445750196,
    0.8944272445820434,
    5,
    4.2,
    1.1494736842105264,
    1.4873805858442077,
  ],
  [
    'DiscreteUniform',
    D.DiscreteUniform(1, 6),
    3,
    -1.791759469228055,
    0.5,
    5,
    3.5,
    2.9166666666666665,
    1.791759469228055,
  ],
]

describe('distributions: univariate families against scipy.stats', () => {
  for (const [name, d, x, lp, cdf, q, mean, variance, entropy] of SCIPY) {
    it(name, () => {
      close(n(d.logProb(x)), lp)
      close(n(d.cdf(x)), cdf)
      close(n(d.quantile(0.7)), q)
      close(n(d.mean()), mean)
      if (name !== 'VonMises') close(n(d.variance()), variance)
      close(n(d.entropy()), entropy)
      close(n(d.survival(x)) + n(d.cdf(x)), 1, 1e-12)
    })
  }

  it('Categorical from probabilities and from logits', () => {
    const p = D.Categorical(tensor([0.2, 0.5, 0.3]))
    const l = D.Categorical({ logits: tensor([Math.log(0.2), Math.log(0.5), Math.log(0.3)]) })
    expect(toFlat(p.logProb(tensor([0, 1, 2])) as Tensor)).toEqual(toFlat(p.logProb(tensor([0, 1, 2])) as Tensor))
    close(n(l.logProb(1)), Math.log(0.5))
    close(n(p.cdf(1)), 0.7)
    close(n(p.entropy()), -(0.2 * Math.log(0.2) + 0.5 * Math.log(0.5) + 0.3 * Math.log(0.3)))
    expect(n(p.logProb(3))).toBe(-Infinity)
    // A class of probability 0 does not poison the others' log-probabilities (0 · log 0).
    const certain = D.Categorical(tensor([1, 0]))
    expect(n(certain.logProb(0))).toBe(0)
    expect(n(certain.logProb(1))).toBe(-Infinity)
  })

  it('numbers in give numbers out; tensors and batches broadcast', () => {
    expect(typeof D.Normal(0, 1).logProb(0.5)).toBe('number')
    const batch = D.Normal(tensor([0, 1]), 2)
    expect(batch.batchShape).toEqual([2])
    const lp = batch.logProb(tensor([[0], [1], [2]])) as Tensor
    expect(lp.shape).toEqual([3, 2])
    const draws = batch.sample(stream(1), { shape: [5] }) as Tensor
    expect(draws.shape).toEqual([5, 2])
    expect(() => D.Normal(0, -1)).toThrow(DomainError)
  })
})

describe('distributions: sampling', () => {
  it('draws match the moments and the cdf (KS) and are deterministic', () => {
    for (const d of [D.Gamma(2.5, 1.3), D.VonMises(0.5, 2), D.TruncatedNormal(1, 2, -1, 4), D.Weibull(1.5, 2)]) {
      const x = toFlat(d.sample(stream('smoke'), { shape: [4000] }) as Tensor).sort((a, b) => a - b)
      const cdf = toFlat(d.cdf(tensor(x)) as Tensor)
      let ks = 0
      x.forEach((_, i) => (ks = Math.max(ks, Math.abs((i + 1) / x.length - cdf[i]), Math.abs(i / x.length - cdf[i]))))
      expect(ks, d.name).toBeLessThan(1.63 / Math.sqrt(x.length))
      expect(toFlat(d.sample(stream('smoke'), { shape: [3] }) as Tensor)).toEqual(
        toFlat(d.sample(stream('smoke'), { shape: [3] }) as Tensor),
      )
    }
    const counts = toFlat(D.NegativeBinomial(3, 0.4).sample(stream('nb'), { shape: [20000] }) as Tensor)
    close(counts.reduce((a, b) => a + b, 0) / counts.length, 4.5, 0.03)
  })
})

describe('distributions: reparameterised draws (rsample)', () => {
  const families: [string, (p: Value) => D.Distribution][] = [
    ['Normal loc', (p) => D.Normal(p, 2)],
    ['Normal scale', (p) => D.Normal(1, p)],
    ['LogNormal', (p) => D.LogNormal(p, 0.5)],
    ['Laplace', (p) => D.Laplace(0, p)],
    ['Logistic', (p) => D.Logistic(p, 1)],
    ['Cauchy', (p) => D.Cauchy(p, 1)],
    ['Uniform', (p) => D.Uniform(0, p)],
    ['Exponential', (p) => D.Exponential(p)],
    ['Gumbel', (p) => D.Gumbel(p, 1)],
    ['Weibull', (p) => D.Weibull(1.5, p)],
  ]
  it.each(families)('%s: rsample draws the same values as sample, and differentiates in the parameter', (_, make) => {
    const d = make(1.3)
    expect(typeof d.rsample).toBe('function')
    const a = toFlat(d.rsample!(stream('r'), { shape: [64] }) as Tensor)
    const b = toFlat(d.sample(stream('r'), { shape: [64] }) as Tensor)
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12))
    // d/dp E_n[x(p, ε)] by autodiff equals central differences with the same noise.
    const f = (p: Value) => tmean(make(p).rsample!(stream('g'), { shape: [64] }))
    const g = n(grad(f)(1.3))
    const h = 1e-6
    close(g, (n(f(1.3 + h)) - n(f(1.3 - h))) / (2 * h), 1e-6)
  })

  it('pathwise gradients estimate d/dμ E[x²] = 2μ for a normal', () => {
    const f = (mu: Value) => tmean(D.Normal(mu, 1).rsample!(stream('pathwise'), { shape: [20000] }), undefined)
    const sq = (mu: Value) => {
      const x = D.Normal(mu, 1).rsample!(stream('pathwise'), { shape: [20000] })
      return tmean(D.Normal(0, 1).logProb(x))
    }
    expect(n(grad(f)(0.7))).toBeCloseTo(1, 12)
    // E[log φ(x)] = −½ log 2π − ½(μ² + 1): derivative −μ.
    expect(n(grad(sq)(0.7))).toBeCloseTo(-0.7, 1)
  })

  it('families without a pathwise draw have no rsample', () => {
    for (const d of [D.Gamma(2, 1), D.Beta(2, 3), D.StudentT(3), D.Poisson(2), D.VonMises(0, 1)])
      expect(d.rsample, d.name).toBeUndefined()
  })
})

describe('distributions: multivariate', () => {
  const cov = fromRows([
    [2, 0.5, 0.3],
    [0.5, 1, 0.2],
    [0.3, 0.2, 1.5],
  ])
  const mvn = D.MultivariateNormal(tensor([1, -1, 0.5]), { covariance: cov })

  it('MultivariateNormal against scipy and the closed-form conditional', () => {
    const lp = toFlat(
      mvn.logProb(
        fromRows([
          [0, 0, 0],
          [1, 2, -1],
        ]),
      ) as Tensor,
    )
    close(lp[0], -4.458011643536986)
    close(lp[1], -9.654830729024061)
    close(n(mvn.entropy()), 4.71795200138987)
    const c = mvn.condition([1], tensor([0.3]))
    expect(toFlat(c.mean() as Tensor).map((v) => Number(v.toFixed(12)))).toEqual([1.65, 0.76])
    expect(toFlat(c.covariance() as Tensor).map((v) => Number(v.toFixed(12)))).toEqual([1.75, 0.2, 0.2, 1.46])
    expect(toFlat(mvn.marginal([2, 0]).covariance() as Tensor).map((v) => Number(v.toFixed(12)))).toEqual([
      1.5, 0.3, 0.3, 2,
    ])
    const precision = D.MultivariateNormal(tensor([1, -1, 0.5]), { precision: cov })
    close(n(precision.logProb(tensor([0, 0, 0]))), -3.533179197838167)
  })

  it('Dirichlet, Multinomial and Wishart against scipy', () => {
    const dir = D.Dirichlet(tensor([2, 3, 1.5]))
    close(n(dir.logProb(tensor([0.2, 0.5, 0.3]))), 1.4924784412154817)
    close(n(dir.entropy()), -1.101605452817366)
    expect(n(dir.logProb(tensor([0.2, 0.5, 0.4])))).toBe(-Infinity)
    const mn = D.Multinomial(6, tensor([0.2, 0.5, 0.3]))
    close(n(mn.logProb(tensor([1, 3, 2]))), -2.0024805005437063)
    close(n(mn.entropy()), 2.790656747469693)
    const w = D.Wishart(5, cov)
    close(
      n(
        w.logProb(
          fromRows([
            [3, 1, 0],
            [1, 4, 0.5],
            [0, 0.5, 2],
          ]),
        ),
      ),
      -11.413762916766835,
    )
    close(n(w.entropy()), 14.803208677213703)
  })
})

describe('distributions: composition', () => {
  it('Mixture, Transformed and Independent', () => {
    const mix = D.Mixture([0.3, 0.7], [D.Normal(-1, 0.5), D.Normal(2, 1)])
    close(n(mix.logProb(0)), -2.656574268737182)
    close(n(mix.cdf(n(mix.quantile(0.5)))), 0.5, 1e-12)
    close(n(mix.variance()), 2.665)
    const t = D.Transformed(D.Normal(0.2, 0.7), B.expBijector)
    close(n(t.logProb(2)), n(D.LogNormal(0.2, 0.7).logProb(2)), 1e-14)
    expect(n(t.logProb(-1))).toBe(-Infinity)
    close(n(t.quantile(0.9)), n(D.LogNormal(0.2, 0.7).quantile(0.9)), 1e-14)
    const ind = D.Independent(D.Normal(tensor([0, 1]), tensor([1, 2])))
    expect([ind.batchShape, ind.eventShape]).toEqual([[], [2]])
    close(n(ind.logProb(tensor([0.5, 0.5]))), n(D.Normal(0, 1).logProb(0.5)) + n(D.Normal(1, 2).logProb(0.5)))
  })
})

describe('distributions: maps, supports and pushforwards', () => {
  const fmt = (b: B.Bijector | B.ManyToOneMap, x: B.Interval) => B.formatInterval(B.imageOf(b, x))
  it('images of supports', () => {
    expect(fmt(B.sigmoidBijector, B.REALS)).toBe('(0, 1)')
    expect(fmt(B.chainBijectors(B.logBijector, B.affineBijector(0, 1)), B.interval(0, 1, '()'))).toBe('(−∞, 0)')
    expect(fmt(B.affineBijector(1, -2), B.interval(0, 1))).toBe('[−1, 1]')
    expect(fmt(B.squareMap, B.supportInterval(D.Exponential(1).support))).toBe('[0, ∞)')
    expect(fmt(B.squareMap, B.interval(-1, 2))).toBe('[0, 4]')
    expect(fmt(B.powerBijector(-1), B.interval(1, 2))).toBe('[0.5, 1]')
    expect(B.formatInterval(B.supportInterval(D.Transformed(D.Normal(0, 1), B.tanhBijector).support))).toBe('(−1, 1)')
    // Beta through logit (the inverse of the sigmoid) covers ℝ; an Exponential through log too (0 has mass zero).
    expect(D.Transformed(D.Exponential(2), B.logBijector).support.type).toBe('real')
    expect(() => D.Transformed(D.Normal(0, 1), B.logBijector)).toThrow(/ℝ is not inside the domain \(0, ∞\) of log/)
    expect(() => D.Transformed(D.Poisson(2), B.logBijector)).toThrow(DomainError)
  })
  it('bijectors: inverse and log-Jacobian', () => {
    const x = 0.7
    for (const b of [
      B.tanhBijector,
      B.softplusBijector,
      B.normalCdfBijector,
      B.powerBijector(2.5),
      B.chainBijectors(B.affineBijector(0, 0.5), B.sigmoidBijector),
    ]) {
      const y = n(b.forward(x))
      close(n(b.inverse(y)), x, 1e-10)
      const h = 1e-6
      const slope = (n(b.forward(x + h)) - n(b.forward(x - h))) / (2 * h)
      close(n(b.logAbsDetJacobian(x)), Math.log(Math.abs(slope)), 1e-7)
    }
  })
  it('Pushforward sums over preimages', () => {
    const chi = D.Pushforward(D.Normal(0, 1), B.squareMap)
    for (const y of [0.3, 1, 4]) {
      close(n(chi.logProb(y)), n(D.ChiSquare(1).logProb(y)), 1e-12)
      close(n(chi.cdf(y)), n(D.ChiSquare(1).cdf(y)), 1e-12)
    }
    expect(n(chi.logProb(-1))).toBe(-Infinity)
    close(n(chi.cdf(n(chi.quantile(0.3)))), 0.3, 1e-9)
    // A base straddling 0 unevenly: Uniform(−1, 2) squared has density 1/(3√y) on (0, 1) and 1/(6√y) on (1, 4).
    const u = D.Pushforward(D.Uniform(-1, 2), B.squareMap)
    close(n(u.prob(0.25)), 1 / (3 * 0.5), 1e-12)
    close(n(u.prob(2.25)), 1 / (6 * 1.5), 1e-12)
    close(n(u.cdf(4)), 1, 1e-12)
    expect(B.formatInterval(B.supportInterval(u.support))).toBe('[0, 4]')
    const draws = toFlat(u.sample(stream('square'), { shape: [200] }) as Tensor)
    expect(Math.min(...draws)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...draws)).toBeLessThanOrEqual(4)
  })
})

describe('distributions: KL divergences against torch', () => {
  it('closed forms', () => {
    const mvn = D.MultivariateNormal(tensor([1, -1, 0.5]), {
      covariance: fromRows([
        [2, 0.5, 0.3],
        [0.5, 1, 0.2],
        [0.3, 0.2, 1.5],
      ]),
    })
    const cases: [D.Distribution, D.Distribution, number][] = [
      [D.Normal(1, 2), D.Normal(0, 1.5), 0.32342904],
      [
        mvn,
        D.MultivariateNormal(tensor([0, 0, 0]), {
          covariance: fromRows([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
          ]),
        }),
        1.26043719,
      ],
      [D.Beta(2, 3), D.Beta(1.5, 0.7), 0.64471054],
      [D.Gamma(2, 3), D.Gamma(1.5, 0.7), 0.74020767],
      [D.Dirichlet(tensor([2, 3, 1.5])), D.Dirichlet(tensor([1, 1, 4])), 3.37513685],
      [D.Categorical(tensor([0.2, 0.5, 0.3])), D.Categorical({ logits: tensor([0, 1, -1]) }), 0.17795289],
      [D.Bernoulli(0.3), D.Bernoulli({ logits: 1.2 }), 0.49241817],
      [D.Poisson(2), D.Poisson(3.5), 0.38076854],
      [D.Exponential(2), D.Exponential(3.5), 0.19038427],
    ]
    for (const [p, q, ref] of cases) close(n(D.kl(p, q)), ref, 1e-6)
    close(n(D.kl(D.Normal(1, 2), D.Normal(1, 2))), 0, 1e-15)
    expect(() => D.kl(D.Cauchy(0, 1), D.Normal(0, 1))).toThrow()
    close(n(D.klMonteCarlo(stream('kl'), D.Normal(1, 2), D.Normal(0, 1.5), 20000)), 0.32342904, 0.03)
  })
})

describe('distributions: numerical divergences', () => {
  it('quadrature against closed forms, and the method reported', () => {
    // KL(Laplace(0, 1) ‖ N(0, 1)) = −H(Laplace) + ½ log 2π + E[x²]/2 = ½ log 2π − log 2.
    const laplace = D.Laplace(0, 1)
    const normal = D.Normal(0, 1)
    close(D.klNumerical(laplace, normal).value, 0.5 * Math.log(2 * Math.PI) - Math.LN2, 1e-8)
    close(D.klNumerical(D.Normal(1, 2), D.Normal(0, 1.5)).value, 0.32342904, 1e-7)
    close(D.klNumerical(D.Beta(2, 3), D.Beta(1.5, 0.7)).value, 0.64471054, 1e-6)
    expect(D.klNumerical(normal, D.Gamma(2, 1)).value).toBe(Infinity)
    expect(D.klAuto(D.Gamma(2, 3), D.Gamma(1.5, 0.7)).method).toBe('closed form')
    const auto = D.klAuto(laplace, normal)
    expect(auto.method).toBe('quadrature')
    expect(auto.converged).toBe(true)
    const mix = D.Mixture([0.3, 0.7], [D.Normal(-2, 0.5), D.Normal(1, 1)])
    close(D.klNumerical(mix, mix).value, 0, 1e-10)
    close(D.entropyNumerical(D.Normal(1, 2)).value, n(D.Normal(1, 2).entropy()), 1e-8)
    expect(D.entropyAuto(mix).method).toBe('quadrature')
    const ce = D.crossEntropyAuto(mix, normal)
    close(ce.value, D.entropyAuto(mix).value + D.klNumerical(mix, normal).value, 1e-8)
    close(D.crossEntropyAuto(D.Normal(1, 2), D.Normal(0, 1.5)).value, n(D.Normal(1, 2).entropy()) + 0.32342904, 1e-7)
  })
  it('Jensen–Shannon, the integrand and the Monte Carlo standard error', () => {
    close(D.jensenShannonNumerical(D.Uniform(0, 1), D.Uniform(2, 3)).value, Math.LN2, 1e-8)
    close(D.jensenShannonNumerical(D.Normal(0, 1), D.Normal(0, 1)).value, 0, 1e-12)
    const a = D.jensenShannonNumerical(D.Normal(0, 1), D.Laplace(1, 2)).value
    close(a, D.jensenShannonNumerical(D.Laplace(1, 2), D.Normal(0, 1)).value, 1e-9)
    const xs = Array.from({ length: 4001 }, (_, i) => -20 + i * 0.01)
    const f = toFlat(D.klIntegrand(D.Normal(1, 2), D.Normal(0, 1.5), tensor(xs)) as Tensor)
    close(f.reduce((t, v) => t + v, 0) * 0.01, 0.32342904, 1e-4)
    expect(n(D.klIntegrand(D.Normal(0, 1), D.Uniform(0, 1), -1))).toBe(Infinity)
    expect(n(D.klIntegrand(D.Uniform(0, 1), D.Normal(0, 1), -1))).toBe(0)
    const mc = D.klMonteCarloWithError(stream('kl'), D.Normal(1, 2), D.Normal(0, 1.5), 20000)
    close(mc.value, n(D.klMonteCarlo(stream('kl'), D.Normal(1, 2), D.Normal(0, 1.5), 20000)), 1e-12)
    expect(Math.abs(mc.value - 0.32342904)).toBeLessThan(4 * mc.standardError)
    expect(mc.standardError).toBeGreaterThan(0.001)
    expect(mc.standardError).toBeLessThan(0.02)
  })
})

describe('distributions: exponential families and gradients', () => {
  it('log p = Σ η·T − A + log h', () => {
    const x = 0.7
    for (const d of [D.Normal(1, 2), D.Gamma(2.5, 1.3), D.Beta(2, 3), D.Exponential(0.7), D.InverseGamma(3, 2)]) {
      const f = d.expFamily!
      const eta = f.naturalParams()
      const T = f.sufficientStats(x)
      const dot = eta.reduce<number>((s, e, i) => s + n(e) * n(T[i]), 0)
      close(dot - n(f.logPartition()) + n(f.logBaseMeasure(x)), n(d.logProb(x)), 1e-12)
    }
    for (const d of [
      D.Poisson(2.5),
      D.Binomial(10, 0.3),
      D.Geometric(0.3),
      D.NegativeBinomial(3, 0.4),
      D.Bernoulli(0.3),
    ]) {
      const f = d.expFamily!
      const k = 1
      close(n(f.naturalParams()[0]) * k - n(f.logPartition()) + n(f.logBaseMeasure(k)), n(d.logProb(k)), 1e-12)
    }
  })

  it('logProb is differentiable in the value and the parameters', () => {
    checkGradient((x, m, s) => D.Normal(m, s).logProb(x), [tensor([0.3, 1.2]), 0.5, 1.7])
    checkGradient((x, a, b) => D.Gamma(a, b).logProb(x), [tensor([0.3, 1.2]), 2.5, 1.3], { tol: 1e-5 })
    checkGradient((x, a, b) => D.Beta(a, b).logProb(x), [tensor([0.3, 0.8]), 2, 3], { tol: 1e-5 })
    checkGradient((x, v, s) => D.StudentT(v, 0, s).logProb(x), [tensor([0.3, 1.2]), 4, 1.5], { tol: 1e-5 })
    checkGradient((x, k) => D.VonMises(0.2, k).logProb(x), [tensor([0.3, 1.2]), 2.5], { tol: 1e-5 })
    checkGradient((p) => D.Poisson(p).logProb(tensor([0, 3])), [2.5])
    checkGradient(
      (m) =>
        D.MultivariateNormal(m, {
          covariance: fromRows([
            [2, 0.5],
            [0.5, 1],
          ]),
        }).logProb(tensor([0.3, 1])),
      [tensor([1, -1])],
    )
    checkGradient((a, b) => D.kl(D.Beta(a, b), D.Beta(1.5, 0.7)), [2, 3], { tol: 1e-5 })
  })
})
