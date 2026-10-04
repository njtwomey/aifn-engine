import { describe, expect, it } from 'vitest'
import * as D from 'aifn-compute/probability/distributions'
import { expBijector, squareMap } from 'aifn-compute/probability/bijectors'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

/**
 * Every distribution family against scipy.stats over wide grids, far tails, extreme quantiles and batches; the
 * compositions; and every closed-form KL rule against torch.distributions (gen/probability/distributions.py).
 */
type Family = {
  family: string
  args: (number | number[])[]
  x: number[]
  logProb: number[]
  cdf?: number[]
  logcdf?: number[]
  survival?: number[]
  logSurvival?: number[]
  p?: number[]
  quantile?: number[]
  mean: number | null
  variance: number | null
  entropy: number | null
}
type Batch = { family?: string; args: (number | number[])[]; x: number[][]; logProb: number[][]; cdf: number[][] }
type Fixture = {
  families: Record<string, Family>
  batches: Record<string, Batch>
  multivariate: {
    MultivariateNormal: { loc: number[]; covariance: number[][]; x: number[][]; logProb: number[]; entropy: number }
    Dirichlet: {
      concentration: number[]
      x: number[][]
      logProb: number[]
      entropy: number
      mean: number[]
      variance: number[]
    }
    Multinomial: { n: number; p: number[]; x: number[][]; logProb: number[]; entropy: number }
    Wishart: { df: number; scale: number[][]; x: number[][][]; logProb: number[]; entropy: number; mean: number[][] }
  }
  compose: {
    Mixture: {
      weights: number[]
      components: [number, number][]
      x: number[]
      logProb: number[]
      cdf: number[]
      mean: number
    }
    Independent: { loc: number[]; scale: number[]; x: number[][]; logProb: number[]; entropy: number }
    Transformed: { x: number[]; logProb: number[]; cdf: number[] }
    Pushforward: { x: number[]; logProb: number[]; cdf: number[] }
  }
  kl: Record<string, { p: unknown[]; q: unknown[]; value: number }>
}
const F = fixture<Fixture>('probability/distributions')

type Ctor = (...a: unknown[]) => D.Distribution
const ctor = (name: string) => (D as unknown as Record<string, Ctor>)[name]
const arg = (a: unknown): unknown => (Array.isArray(a) ? tensor(a as number[] | number[][]) : a)
const flat = (v: unknown): number[] => (typeof v === 'number' ? [v] : Array.from(toFlat(v as Tensor)))

/** Relative agreement, exact for infinities; reference NaN or null is skipped (scipy has no value there). */
function agree(got: number[], want: (number | null)[], rtol: number, what: string) {
  expect(got.length, what).toBe(want.length)
  want.forEach((w, i) => {
    if (w === null || Number.isNaN(w)) return
    const g = got[i]
    if (!Number.isFinite(w)) return expect(g, `${what}[${i}]`).toBe(w)
    expect(Math.abs(g - w), `${what}[${i}]: ${g} vs ${w}`).toBeLessThanOrEqual(rtol * Math.abs(w) + 1e-300)
  })
}

/** Tolerances: densities and probabilities to 1e-9 relative; quantiles to 1e-8 (they invert a cdf). */
const RTOL = {
  logProb: 1e-9,
  cdf: 1e-9,
  logcdf: 1e-9,
  survival: 1e-9,
  logSurvival: 1e-9,
  quantile: 1e-8,
  moment: 1e-10,
}

describe('univariate families match scipy.stats', () => {
  for (const [name, f] of Object.entries(F.families))
    describe(name, () => {
      const d = ctor(f.family)(...f.args.map(arg)) as D.Univariate
      const x = tensor(f.x)
      it('log density', () => agree(flat(d.logProb(x)), f.logProb, RTOL.logProb, `${name}.logProb`))
      for (const m of ['cdf', 'logcdf', 'survival', 'logSurvival'] as const)
        if (f[m]) it(m, () => agree(flat(d[m](x)), f[m]!, RTOL[m], `${name}.${m}`))
      if (f.quantile)
        it('quantile at extreme probabilities', () =>
          agree(flat(d.quantile(tensor(f.p!))), f.quantile!, RTOL.quantile, `${name}.quantile`))
      it('mean, variance and entropy', () => {
        if (f.mean !== null) agree(flat(d.mean()), [f.mean], RTOL.moment, `${name}.mean`)
        if (f.variance !== null) agree(flat(d.variance()), [f.variance], RTOL.moment, `${name}.variance`)
        if (f.entropy !== null) agree(flat(d.entropy()), [f.entropy], RTOL.moment, `${name}.entropy`)
      })
    })
})

describe('batched parameters broadcast as numpy does', () => {
  for (const [name, b] of Object.entries(F.batches))
    it(name, () => {
      const d = ctor(b.family ?? name)(...b.args.map(arg)) as D.Univariate
      expect(d.batchShape).toEqual([3])
      const x = tensor(b.x)
      const lp = d.logProb(x) as Tensor
      expect(lp.shape).toEqual([4, 3])
      agree(flat(lp), b.logProb.flat(), 1e-10, `${name}.logProb`)
      agree(flat(d.cdf(x)), b.cdf.flat(), 1e-10, `${name}.cdf`)
    })
})

describe('multivariate families match scipy.stats', () => {
  const M = F.multivariate
  it('MultivariateNormal', () => {
    const d = D.MultivariateNormal(tensor(M.MultivariateNormal.loc), {
      covariance: tensor(M.MultivariateNormal.covariance),
    })
    agree(flat(d.logProb(tensor(M.MultivariateNormal.x))), M.MultivariateNormal.logProb, 1e-11, 'mvn.logProb')
    agree(flat(d.entropy()), [M.MultivariateNormal.entropy], 1e-12, 'mvn.entropy')
  })
  it('Dirichlet', () => {
    const d = D.Dirichlet(tensor(M.Dirichlet.concentration))
    agree(flat(d.logProb(tensor(M.Dirichlet.x))), M.Dirichlet.logProb, 1e-10, 'dirichlet.logProb')
    agree(flat(d.entropy()), [M.Dirichlet.entropy], 1e-11, 'dirichlet.entropy')
    agree(flat(d.mean()), M.Dirichlet.mean, 1e-12, 'dirichlet.mean')
    agree(flat(d.variance()), M.Dirichlet.variance, 1e-12, 'dirichlet.variance')
  })
  it('Multinomial', () => {
    const d = D.Multinomial(M.Multinomial.n, tensor(M.Multinomial.p))
    agree(flat(d.logProb(tensor(M.Multinomial.x))), M.Multinomial.logProb, 1e-11, 'multinomial.logProb')
    agree(flat(d.entropy()), [M.Multinomial.entropy], 1e-10, 'multinomial.entropy')
  })
  it('Wishart', () => {
    const d = D.Wishart(M.Wishart.df, tensor(M.Wishart.scale))
    agree(flat(d.logProb(tensor(M.Wishart.x))), M.Wishart.logProb, 1e-10, 'wishart.logProb')
    agree(flat(d.entropy()), [M.Wishart.entropy], 1e-10, 'wishart.entropy')
    agree(flat(d.mean()), M.Wishart.mean.flat(), 1e-12, 'wishart.mean')
  })
})

describe('compositions', () => {
  const C = F.compose
  it('Mixture of normals', () => {
    const m = D.Mixture(
      C.Mixture.weights,
      C.Mixture.components.map(([mu, s]) => D.Normal(mu, s)),
    )
    agree(flat(m.logProb(tensor(C.Mixture.x))), C.Mixture.logProb, 1e-11, 'mixture.logProb')
    agree(flat(m.cdf(tensor(C.Mixture.x))), C.Mixture.cdf, 1e-11, 'mixture.cdf')
    agree(flat(m.mean()), [C.Mixture.mean], 1e-12, 'mixture.mean')
  })
  it('Independent normals', () => {
    const d = D.Independent(D.Normal(tensor(C.Independent.loc), tensor(C.Independent.scale)))
    agree(flat(d.logProb(tensor(C.Independent.x))), C.Independent.logProb, 1e-12, 'independent.logProb')
    agree(flat(d.entropy()), [C.Independent.entropy], 1e-12, 'independent.entropy')
  })
  it('Transformed: exp of a standard normal is log-normal', () => {
    const d = D.Transformed(D.Normal(0, 1), expBijector)
    agree(flat(d.logProb(tensor(C.Transformed.x))), C.Transformed.logProb, 1e-11, 'transformed.logProb')
    agree(flat(d.cdf(tensor(C.Transformed.x))), C.Transformed.cdf, 1e-11, 'transformed.cdf')
  })
  it('Pushforward: the square of a standard normal is χ²₁', () => {
    const d = D.Pushforward(D.Normal(0, 1), squareMap)
    agree(flat(d.logProb(tensor(C.Pushforward.x))), C.Pushforward.logProb, 1e-11, 'pushforward.logProb')
    agree(flat(d.cdf(tensor(C.Pushforward.x))), C.Pushforward.cdf, 1e-10, 'pushforward.cdf')
  })
})

describe('closed-form KL rules match torch.distributions', () => {
  const build = (name: string, a: unknown[]) =>
    name === 'MultivariateNormal'
      ? D.MultivariateNormal(tensor(a[0] as number[]), { covariance: tensor(a[1] as number[][]) })
      : ctor(name)(...a.map(arg))
  for (const [pair, c] of Object.entries(F.kl))
    it(pair, () => {
      const [p, q] = pair.split('|')
      expect(D.hasKl(build(p, c.p), build(q, c.q))).toBe(true)
      agree(flat(D.kl(build(p, c.p), build(q, c.q))), [c.value], 1e-10, `kl ${pair}`)
    })
})
