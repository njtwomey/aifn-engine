import { describe, expect, it } from 'vitest'
import * as D from 'aifn-compute/probability/distributions'
import { grad } from 'aifn-compute/foundation/autodiff'
import { tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

/**
 * The points at which the scipy fixtures (fixtures.test.ts) caught tail-accuracy bugs, kept as named regression tests.
 * Each was off before the fix: log tails computed as log(cdf) or log(survival) near 1 (0 or rounded), the Cauchy
 * quantile through tan(π(p − ½)), the Laplace quantile through log1p(−2|p − ½|) (−∞ at p = 1e-300), the inverse-gamma
 * quantile through a gamma quantile at 1 − p (0 at p = 1e-300), gamma upper quantiles by bisection on a cdf near 1
 * (~2e-6 relative), and the uniform survival function as 1 − cdf. References: scipy.stats, mpmath where scipy
 * underflows or rounds (gen/probability/distributions.py).
 */

type Ctor = (...a: unknown[]) => D.Univariate
const build = (name: string, args: number[]) => (D as unknown as Record<string, Ctor>)[name](...args) as D.Univariate
const num = (v: Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
const close = (got: number, want: number, rtol: number) =>
  expect(Math.abs(got - want), `${got} vs ${want}`).toBeLessThanOrEqual(rtol * Math.abs(want))

type Method = 'logcdf' | 'logSurvival' | 'survival' | 'quantile'
const PINNED: [string, string, number[], Method, number, number][] = [
  ['StudentT', 'StudentT', [3.5, -1, 1.5], 'logcdf', 1e4, -7.354099423993921e-14],
  ['StudentT', 'StudentT', [3.5, -1, 1.5], 'logSurvival', -1e6, -7.356700258226648e-21],
  ['Cauchy', 'Cauchy', [0.5, 2], 'logcdf', 1e12, -6.366197723681024e-13],
  ['Cauchy', 'Cauchy', [0.5, 2], 'logSurvival', -1e10, -6.366197723560146e-11],
  ['Cauchy', 'Cauchy', [0.5, 2], 'quantile', 1e-300, -6.366197723675814e299],
  ['Laplace', 'Laplace', [1, 0.7], 'logSurvival', -400, -8.133433920332643e-250],
  ['Laplace', 'Laplace', [1, 0.7], 'quantile', 1e-300, -482.0576665023576],
  ['Uniform', 'Uniform', [-2, 3], 'survival', 2.9999999, 1.9999999967268424e-8],
  ['Gamma', 'Gamma', [2.7, 1.8], 'logcdf', 20, -6.961280716998938e-14],
  ['Gamma', 'Gamma', [2.7, 1.8], 'logSurvival', 1e-30, -1.1722812872809138e-81],
  ['Gamma', 'Gamma', [2.7, 1.8], 'quantile', 0.999999999999, 18.4453283541054],
  ['GammaWithScale', 'GammaWithScale', [0.4, 3], 'logcdf', 500, -8.648342320343863e-75],
  ['GammaWithScale', 'GammaWithScale', [0.4, 3], 'logSurvival', 1e-40, -7.262710394194784e-17],
  ['GammaWithScale', 'GammaWithScale', [0.4, 3], 'quantile', 0.999999999999, 74.64867252112815],
  ['ChiSquare', 'ChiSquare', [7], 'logcdf', 50, -1.4444852883542293e-8],
  ['ChiSquare', 'ChiSquare', [7], 'logSurvival', 1e-20, -7.598900579074907e-73],
  ['ChiSquare underflow', 'ChiSquare', [7], 'logSurvival', 1500, -734.6474561470056],
  ['ChiSquare', 'ChiSquare', [7], 'quantile', 0.999999999999, 70.838475756726],
  ['InverseGamma', 'InverseGamma', [3, 2], 'logcdf', 1e4, -1.3331333493333333e-12],
  ['InverseGamma underflow', 'InverseGamma', [3, 2], 'logcdf', 0.001, -1985.4903422616424],
  ['InverseGamma', 'InverseGamma', [3, 2], 'logSurvival', 0.02, -1.8976107553682285e-40],
  ['InverseGamma', 'InverseGamma', [3, 2], 'quantile', 1e-300, 0.0028441552351649367],
  ['Beta', 'Beta', [2.5, 0.6], 'logcdf', 0.99999999999999, -7.3621740733105776e-9],
  ['Beta', 'Beta', [2.5, 0.6], 'logSurvival', 1e-30, -4.4404366484386703e-76],
  ['Gumbel', 'Gumbel', [0.5, 1.5], 'logSurvival', -8, -2.875882525963561e-126],
  ['TruncatedNormal', 'TruncatedNormal', [0.5, 1.2, -1, 4], 'logcdf', 3.9999999, -5.294459058771956e-10],
  ['TruncatedNormal tail', 'TruncatedNormal', [0, 1, 8, Infinity], 'logcdf', 10, -1.224867603620687e-8],
  ['NegativeBinomial', 'NegativeBinomial', [3.5, 0.4], 'logcdf', 80, -2.1288729831605563e-15],
]

describe('tail accuracy at the points the fixtures pinned', () => {
  for (const [label, family, args, method, at, want] of PINNED)
    it(`${label}.${method}(${at})`, () =>
      close(num(build(family, args)[method](at)), want, method === 'quantile' ? 1e-10 : 1e-9))
})

/** scipy.stats isf at q = 1e-12, 1e-100 and 0.3. */
const ISF: [string, number[], number[]][] = [
  ['Normal', [1.5, 2], [15.568967650602263, 44.046907121930644, 2.548801025416082]],
  ['StudentT', [3.5, -1, 1.5], [4743.546025697023, 6.592526269032273e28, -0.13699194734165343]],
  ['Laplace', [1, 0.7], [19.85651175475802, 161.69575348319123, 1.3575779366361935]],
  ['Logistic', [-0.5, 1.3], [35.42032745070581, 298.83606208922595, 0.6014872185033648]],
  ['Exponential', [2.5], [11.05240844637142, 92.10340371976184, 0.4815891217303745]],
  ['Gamma', [2.7, 1.8], [18.445315421696897, 132.85654959366434, 1.8147504956270502]],
  ['ChiSquare', [7], [70.83842825582607, 485.59684328569676, 8.383430828608386]],
  ['InverseGamma', [3, 2], [11005.924153896041, 2.3712622029933752e33, 1.045054497051086]],
  ['Weibull', [1.7, 2.2], [15.498984026372893, 53.94625752178227, 2.453828778543324]],
  ['Gumbel', [0.5, 1.5], [41.946531673892075, 345.8877639491069, 2.046395649738085]],
]

describe('isf: the inverse survival function keeps relative accuracy for tiny q', () => {
  for (const [family, args, want] of ISF)
    it(family, () => {
      const d = build(family, args)
      const got = Array.from(toFlat(d.isf(tensor([1e-12, 1e-100, 0.3])) as Tensor))
      want.forEach((w, i) => close(got[i], w, 1e-10))
    })
  it('Cauchy: isf(q) = −quantile(q) about the location, to 1e299', () => {
    close(num(D.Cauchy(0.5, 2).isf(1e-300)), 6.366197723675814e299, 1e-12)
  })
  it('the default inverts the survival function numerically (VonMises, Mixture)', () => {
    const v = D.VonMises(0.7, 2.5)
    close(num(v.survival(v.isf(0.2))), 0.2, 1e-9)
    const m = D.Mixture([0.3, 0.7], [D.Normal(-1, 1), D.Normal(2, 0.5)])
    close(num(m.survival(m.isf(1e-30))), 1e-30, 1e-8)
  })
  it('Beta: isf(q) = 1 − I⁻¹(b, a; q)', () => {
    close(num(D.Beta(2.5, 0.6).isf(0.3)), 0.949429101427429, 1e-12)
  })
})

describe('the new quantiles are differentiable in p and the rate', () => {
  it('Gamma: dQ/dp = 1/density at the quantile, dQ/dβ = −Q/β', () => {
    const p = 0.999999999999
    const x = num(D.Gamma(2.7, 1.8).quantile(p))
    const dp = grad((u: Value) => D.Gamma(2.7, 1.8).quantile(u))(p) as number
    close(dp, 1 / Math.exp(num(D.Gamma(2.7, 1.8).logProb(x))), 1e-8)
    const db = grad((b: Value) => D.Gamma(2.7, b).quantile(0.3))(1.8) as number
    close(db, -num(D.Gamma(2.7, 1.8).quantile(0.3)) / 1.8, 1e-12)
  })
  it('Cauchy and Laplace: dQ/dp = 1/density at p = 1e-3', () => {
    for (const d of [D.Cauchy(0.5, 2), D.Laplace(1, 0.7)]) {
      const x = num(d.quantile(1e-3))
      const dp = grad((u: Value) => d.quantile(u))(1e-3) as number
      close(dp, 1 / Math.exp(num(d.logProb(x))), 1e-9)
    }
  })
  it('logcdf near 1 has the density over the cdf as derivative (Gamma, StudentT, Beta)', () => {
    for (const [d, x] of [
      [D.Gamma(2.7, 1.8), 20],
      [D.StudentT(3.5, -1, 1.5), 50],
      [D.Beta(2.5, 0.6), 0.999],
    ] as const) {
      const g = grad((v: Value) => d.logcdf(v))(x) as number
      close(g, Math.exp(num(d.logProb(x)) - num(d.logcdf(x))), 1e-9)
    }
  })
})
