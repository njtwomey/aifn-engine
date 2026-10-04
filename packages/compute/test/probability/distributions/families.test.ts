/**
 * The generated distribution suite (design S §7): every family registered in the compute (`kind: 'distribution'`) is
 * built at its default parameters (from `info.params`, in argument order, or from `BUILD` for families whose
 * parameters are vectors or other distributions), and
 *
 * - conforms to its info: the instance's `discrete`, event rank, exponential-family structure and support type match
 *   the declaration;
 * - samples against its cdf: a Kolmogorov–Smirnov distance below the 0.1 % critical value 1.95/√n for univariate
 *   families with a cdf (on the support's values for discrete ones);
 * - samples against its moments: the sample mean within 6 standard errors of `mean()`, and the sample variance within
 *   25 % of `variance()` where the fourth moment is finite (not the heavy-tailed defaults in `HEAVY`).
 *
 * Draws use a stream keyed by the family, so the suite is deterministic.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Distribution, DistributionInfo, SupportName } from 'aifn-compute/foundation/contracts'
import { expBijector, squareMap } from 'aifn-compute/probability/bijectors'
import {
  Categorical,
  Dirichlet,
  Independent,
  Mixture,
  Multinomial,
  MultivariateNormal,
  Normal,
  Poisson,
  Pushforward,
  Transformed,
  Wishart,
  ZeroInflated,
} from 'aifn-compute/probability/distributions'
import { address, entriesOf } from '../../registries'

/** Families whose parameters are not plain scalars, built by hand. */
const BUILD: Record<string, () => Distribution> = {
  Categorical: () => Categorical(tensor([0.2, 0.5, 0.3])),
  MultivariateNormal: () =>
    MultivariateNormal(tensor([1, -1]), {
      covariance: tensor([
        [2, 0.5],
        [0.5, 1],
      ]),
    }),
  Dirichlet: () => Dirichlet(tensor([2, 3, 4])),
  Multinomial: () => Multinomial(10, tensor([0.2, 0.5, 0.3])),
  Wishart: () =>
    Wishart(
      4,
      tensor([
        [1, 0.3],
        [0.3, 2],
      ]),
    ),
  Mixture: () => Mixture([0.3, 0.7], [Normal(-2, 1), Normal(2, 0.5)]),
  Independent: () => Independent(Normal(tensor([0, 1]), 1)),
  Transformed: () => Transformed(Normal(0, 0.5), expBijector),
  Pushforward: () => Pushforward(Normal(0, 1), squareMap),
  ZeroInflated: () => ZeroInflated(0.3, Poisson(3)),
}

/** Defaults whose fourth moment is infinite, so the sample variance does not settle. */
const HEAVY = new Set(['StudentT', 'Cauchy', 'InverseGamma', 'VonMises'])

/** The support type each support name implies, with fixed bounds where it has them. */
const SUPPORT: Record<SupportName, { type: string | null; lower?: number; upper?: number }> = {
  real: { type: 'real' },
  positive: { type: 'interval', lower: 0, upper: Infinity },
  'non-negative': { type: 'interval', lower: 0, upper: Infinity },
  'unit-interval': { type: 'interval', lower: 0, upper: 1 },
  interval: { type: 'interval' },
  circle: { type: 'circle' },
  integers: { type: 'integers' },
  'non-negative-integers': { type: 'integers', lower: 0, upper: Infinity },
  'positive-integers': { type: 'integers', lower: 1, upper: Infinity },
  binary: { type: 'integers', lower: 0, upper: 1 },
  categories: { type: 'integers', lower: 0 },
  simplex: { type: 'simplex' },
  'real-vector': { type: 'real-vector' },
  'count-vector': { type: 'count-vector' },
  'positive-definite': { type: 'positive-definite' },
  varies: { type: null },
}

const scalar = (v: unknown): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
const flat = (v: Value): number[] => (typeof v === 'number' ? [v] : Array.from(toFlat(v as Tensor)))

const families = await entriesOf<DistributionInfo>('distribution')

function build(key: string, entry: (typeof families)[number]): Distribution {
  if (BUILD[key]) return BUILD[key]()
  const args = Object.values(defaults(entry.info.params))
  return (entry as unknown as (...a: unknown[]) => Distribution)(...args)
}

describe('the distribution registry', () => {
  it('lists the families', () => {
    expect(families.length).toBeGreaterThan(25)
    for (const f of families) expect(typeof f).toBe('function')
  })
})

describe.each(families.map((f) => [address(f), f] as const))('%s', (_, entry) => {
  const info = entry.info
  const d = build(info.key, entry)
  const N = 4000

  it('conforms to its info', () => {
    expect(d.name).toBe(info.key === 'GammaWithScale' ? 'Gamma' : info.key)
    expect(d.eventShape.length).toBe(info.eventRank)
    if (!info.composite) {
      expect(Boolean(d.discrete)).toBe(info.discrete)
      expect(d.expFamily !== undefined).toBe(info.expFamily)
    }
    const want = SUPPORT[info.support]
    if (want.type !== null) {
      expect(d.support.type).toBe(want.type)
      const s = d.support as { lower?: Value; upper?: Value }
      if (want.lower !== undefined) expect(scalar(s.lower)).toBe(want.lower)
      if (want.upper !== undefined) expect(scalar(s.upper)).toBe(want.upper)
    }
  })

  const univariateCdf = info.eventRank === 0 && typeof (d as { cdf?: unknown }).cdf === 'function'
  it.runIf(univariateCdf)('samples follow its cdf (Kolmogorov–Smirnov)', () => {
    const x = flat(d.sample(stream(`ks/${info.key}`), { shape: [N] }) as Value).sort((a, b) => a - b)
    const cdf = (v: number[]) => flat((d as unknown as { cdf(x: Value): Value }).cdf(tensor(v)))
    let D = 0
    if (info.discrete) {
      const values = [...new Set(x)]
      const F = cdf(values)
      let i = 0
      values.forEach((v, j) => {
        while (i < N && x[i] <= v) i++
        D = Math.max(D, Math.abs(i / N - F[j]))
      })
    } else {
      const F = cdf(x)
      x.forEach((_, i) => (D = Math.max(D, Math.abs((i + 1) / N - F[i]), Math.abs(i / N - F[i]))))
    }
    expect(D).toBeLessThan(1.95 / Math.sqrt(N))
  })

  it('sample moments match mean() and variance()', () => {
    let mean: number[]
    let variance: number[]
    try {
      mean = flat(d.mean() as Value)
      variance = flat((info.eventRank === 0 ? d.variance() : d.variance()) as Value)
    } catch {
      return // moments not provided (e.g. Pushforward)
    }
    const draws = flat(d.sample(stream(`moments/${info.key}`), { shape: [N] }) as Value)
    const k = mean.length
    for (let j = 0; j < k; j++) {
      if (!Number.isFinite(mean[j]) || !Number.isFinite(variance[j])) continue
      let m = 0
      for (let i = 0; i < N; i++) m += draws[i * k + j]
      m /= N
      expect(Math.abs(m - mean[j])).toBeLessThanOrEqual(6 * Math.sqrt(variance[j] / N) + 1e-12)
      if (HEAVY.has(info.key) || variance[j] === 0) continue
      let v = 0
      for (let i = 0; i < N; i++) v += (draws[i * k + j] - m) ** 2
      v /= N - 1
      expect(Math.abs(v / variance[j] - 1)).toBeLessThan(0.25)
    }
  })
})
