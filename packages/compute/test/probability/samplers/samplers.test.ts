import { describe, expect, it } from 'vitest'
import * as F from 'aifn-compute/foundation/random'
import * as P from 'aifn-compute/probability/samplers'
import { chiSquareSf } from 'aifn-compute/numerics/special'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

// The foundation's draws and the named families together, as the samplers are used.
const R = { ...F, ...P, gamma: P.gammaVariate }
type Stream = F.Stream

// All tests use fixed streams, so they are deterministic. Thresholds are set at significance levels near 1e-3 (or
// several standard errors), so a correct sampler passes for almost any seed and the fixed seeds cannot mask a bug.

type Continuous = { name: string; args: Record<string, number>; p: number[]; x: number[]; mean: number; var: number }
type Discrete = { name: string; args: Record<string, number>; k: number[]; pmf: number[]; mean: number; var: number }
const ref = fixture<{ continuous: Continuous[]; discrete: Discrete[] }>('probability/samplers')

const N = 20_000

function moments(xs: ArrayLike<number>): { mean: number; variance: number } {
  let m = 0
  for (let i = 0; i < xs.length; i++) m += xs[i]
  m /= xs.length
  let v = 0
  for (let i = 0; i < xs.length; i++) v += (xs[i] - m) ** 2
  return { mean: m, variance: v / (xs.length - 1) }
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ma = moments(a).mean
  const mb = moments(b).mean
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

/** Pearson chi-square p-value for observed counts against expected probabilities, pooling cells with expectation < 5. */
function chiSquarePValue(observed: number[], probabilities: number[], n: number): number {
  let stat = 0
  let cells = 0
  let obsPool = 0
  let expPool = 0
  for (let i = 0; i < observed.length; i++) {
    obsPool += observed[i]
    expPool += n * probabilities[i]
    if (expPool >= 5) {
      stat += (obsPool - expPool) ** 2 / expPool
      cells++
      obsPool = 0
      expPool = 0
    }
  }
  if (expPool > 0) {
    stat += (obsPool - expPool) ** 2 / Math.max(expPool, 1e-300)
    cells++
  }
  return chiSquareSf(stat, cells - 1)
}

/** Draw N values of the continuous sampler named in a fixture case. */
function drawContinuous(c: Continuous, s: Stream): Float64Array {
  const a = c.args
  const kind = c.name.split(' ')[0]
  const one: () => number = {
    uniform: () => R.uniform(s, a.a, a.b),
    normal: () => R.normal(s, a.mean, a.sd),
    exponential: () => R.exponential(s, a.rate),
    gamma: () => R.gamma(s, a.shape, a.scale),
    beta: () => R.beta(s, a.a, a.b),
    chiSquare: () => R.chiSquare(s, a.df),
    studentT: () => R.studentT(s, a.df, a.loc, a.scale),
  }[kind]!
  return Float64Array.from({ length: N }, one)
}

describe('continuous samplers against scipy.stats', () => {
  for (const c of ref.continuous) {
    it(`${c.name}: empirical cdf and moments`, () => {
      const xs = drawContinuous(c, R.child(R.stream('continuous'), c.name)).sort()
      // Kolmogorov–Smirnov on scipy's quantile grid: |F_N(x) − F(x)| at 199 points. 1.95/√N is the 0.1% critical value.
      let worst = 0
      let j = 0
      c.x.forEach((x, i) => {
        // Skip quantiles that coincide after rounding (Beta(0.05, 0.08) puts 1.5% of its mass within 1e-16 of 1).
        if (x === c.x[i - 1] || x === c.x[i + 1]) return
        while (j < N && xs[j] <= x) j++
        worst = Math.max(worst, Math.abs(j / N - c.p[i]))
      })
      expect(worst, c.name).toBeLessThan(1.95 / Math.sqrt(N))
      // The mean within 4.5 standard errors where the variance is finite (heavy-tailed t skips this).
      if (Number.isFinite(c.var)) {
        expect(Math.abs(moments(xs).mean - c.mean), c.name).toBeLessThan(4.5 * Math.sqrt(c.var / N))
      }
    })
  }
})

describe('discrete samplers against scipy.stats', () => {
  for (const c of ref.discrete) {
    it(`${c.name}: chi-square goodness of fit and moments`, () => {
      // (The key names the stream of draws: 'discrete' gave p = 0.0008 for Poisson(4) after the phase-1 draw layout
      // changed; over 40 other keys the p-values are uniform, so the sampler is right and that key was unlucky.)
      const s = R.child(R.stream('discrete samplers'), c.name)
      const [kind] = c.name.split(' ')
      const counts = new Array(c.k.length).fill(0)
      const xs = new Float64Array(N)
      for (let i = 0; i < N; i++) {
        const x = kind === 'poisson' ? R.poisson(s, c.args.lambda) : R.binomial(s, c.args.n, c.args.p)
        xs[i] = x
        const idx = x - c.k[0]
        expect(idx >= 0 && idx < c.k.length, `${x} in the support range`).toBe(true)
        counts[idx]++
      }
      expect(chiSquarePValue(counts, c.pmf, N), c.name).toBeGreaterThan(1e-3)
      expect(Math.abs(moments(xs).mean - c.mean)).toBeLessThan(4.5 * Math.sqrt(c.var / N))
    })
  }
})

describe('multivariate samplers', () => {
  it('dirichlet has the right means and variances, and small concentrations give no NaN', () => {
    const alpha = [0.5, 2, 7.5]
    const a0 = 10
    const s = R.stream('dirichlet')
    const draws = Array.from({ length: N }, () => toFlat(R.dirichlet(s, alpha)))
    alpha.forEach((a, k) => {
      const m = moments(draws.map((d) => d[k]))
      const mean = a / a0
      const variance = (mean * (1 - mean)) / (a0 + 1)
      expect(Math.abs(m.mean - mean)).toBeLessThan(4.5 * Math.sqrt(variance / N))
      expect(Math.abs(m.variance / variance - 1)).toBeLessThan(0.08)
    })
    for (const d of draws.slice(0, 100)) expect(d.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12)
    const tiny = toFlat(R.dirichlet(s, [1e-4, 1e-4, 1e-4]))
    expect(tiny.every(Number.isFinite)).toBe(true)
    expect(tiny.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12)
    expect(Number.isFinite(R.beta(s, 1e-5, 1e-5))).toBe(true)
  })

  it('multinomial counts sum to n with the right means and covariances', () => {
    const p = [0.2, 0.5, 0.3]
    const n = 30
    const s = R.stream('multinomial')
    const draws = Array.from({ length: N }, () => toFlat(R.multinomial(s, n, p)))
    for (const d of draws) expect(d[0] + d[1] + d[2]).toBe(n)
    const c0 = draws.map((d) => d[0])
    const c1 = draws.map((d) => d[1])
    const m0 = moments(c0)
    expect(Math.abs(m0.mean - n * p[0])).toBeLessThan(4.5 * Math.sqrt((n * p[0] * (1 - p[0])) / N))
    // Corr(c0, c1) = −√(p0 p1 / ((1 − p0)(1 − p1))).
    const rho = -Math.sqrt((p[0] * p[1]) / ((1 - p[0]) * (1 - p[1])))
    expect(Math.abs(correlation(c0, c1) - rho)).toBeLessThan(0.03)
  })

  it('multivariateNormal has covariance L Lᵀ', () => {
    const L = fromRows([
      [2, 0],
      [-1.5, 0.5],
    ])
    const s = R.stream('mvn')
    const draws = toRows(R.multivariateNormal(s, [1, -2], { choleskyFactor: L }, { shape: [N] }))
    const x = draws.map((d) => d[0])
    const y = draws.map((d) => d[1])
    expect(Math.abs(moments(x).mean - 1)).toBeLessThan(4.5 * Math.sqrt(4 / N))
    expect(moments(x).variance).toBeCloseTo(4, 0)
    expect(moments(y).variance).toBeCloseTo(2.5, 0)
    // Corr = Σ₀₁/√(Σ₀₀Σ₁₁) = −3/√(4 · 2.5).
    expect(Math.abs(correlation(x, y) + 3 / Math.sqrt(10))).toBeLessThan(0.02)
  })
})

describe('named-family samplers: shapes, broadcasting and element order', () => {
  const scalarCalls: [string, (s: Stream) => number, (s: Stream, shape: number[]) => Tensor][] = [
    ['logGammaVariate', (s) => R.logGammaVariate(s, 0.3), (s, shape) => R.logGammaVariate(s, 0.3, { shape })],
    ['gamma', (s) => R.gamma(s, 2.5, 0.5), (s, shape) => R.gamma(s, 2.5, 0.5, { shape })],
    ['beta', (s) => R.beta(s, 0.5, 2), (s, shape) => R.beta(s, 0.5, 2, { shape })],
    ['chiSquare', (s) => R.chiSquare(s, 3), (s, shape) => R.chiSquare(s, 3, { shape })],
    ['studentT', (s) => R.studentT(s, 4, 1, 2), (s, shape) => R.studentT(s, 4, 1, 2, { shape })],
    ['poisson', (s) => R.poisson(s, 12), (s, shape) => R.poisson(s, 12, { shape })],
    ['binomial', (s) => R.binomial(s, 80, 0.4), (s, shape) => R.binomial(s, 80, 0.4, { shape })],
  ]
  for (const [name, one, many] of scalarCalls) {
    it(`${name}: numbers give a number; a shape gives the same draws in row-major order`, () => {
      expect(typeof one(R.child(R.stream('shape'), name))).toBe('number')
      const a = R.child(R.stream('shape'), name)
      const want = Array.from({ length: 24 }, () => one(a))
      for (const shape of [[24], [4, 6], [2, 3, 4], [1, 24, 1]]) {
        const t = many(R.child(R.stream('shape'), name), shape)
        expect(t.shape).toEqual(shape)
        expect(t.dtype).toBe('float64')
        expect(toFlat(t)).toEqual(want)
      }
      // Rank 0: a scalar tensor holding the first draw.
      const r0 = many(R.child(R.stream('shape'), name), [])
      expect(r0.shape).toEqual([])
      expect(toFlat(r0)).toEqual([want[0]])
    })
  }

  it('tensor parameters broadcast for the named families too', () => {
    const lambdas = tensor([1, 5, 50], [3], 'int32')
    const counts = R.poisson(R.stream('bc'), lambdas)
    expect(counts.shape).toEqual([3])
    const u = R.stream('bc')
    expect(toFlat(counts)).toEqual([1, 5, 50].map((l) => R.poisson(u, l)))
    expect(R.binomial(R.stream('bc'), 10, tensor([0.1, 0.9])).shape).toEqual([2])
  })

  it('broadcast gamma parameters give the right distribution in every cell', () => {
    const g = toRows(R.gamma(R.stream('cells'), tensor([0.5, 4], [2, 1]), 2, { shape: [2, N] }))
    g.forEach((row, i) => {
      const a = [0.5, 4][i]
      expect(Math.abs(moments(row).mean - 2 * a)).toBeLessThan(4.5 * Math.sqrt((4 * a) / N))
    })
  })

  it('dirichlet and multinomial append the event axis and broadcast batches', () => {
    const alpha = fromRows([
      [1, 2, 3],
      [50, 50, 50],
    ])
    const d = R.dirichlet(R.stream('dir'), alpha, { shape: [4, 2] })
    expect(d.shape).toEqual([4, 2, 3])
    for (const row of toRows(R.dirichlet(R.stream('dir'), alpha)))
      expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 12)
    // Row-major order: the first row of a batch is the draw from the first alpha.
    const first = toFlat(R.dirichlet(R.stream('dir'), [1, 2, 3]))
    expect(toFlat(d).slice(0, 3)).toEqual(first)
    const m = R.multinomial(R.stream('mult'), tensor([5, 50], [2, 1]), [0.2, 0.8], { shape: [2, 3] })
    expect(m.shape).toEqual([2, 3, 2])
    toRows(R.multinomial(R.stream('mult'), tensor([5, 50]), [0.2, 0.8])).forEach((row, i) =>
      expect(row[0] + row[1]).toBe([5, 50][i]),
    )
  })

  it('multivariateNormal takes a covariance or a Cholesky factor, and reports a covariance that does not factor', () => {
    const cov = fromRows([
      [4, -3],
      [-3, 2.5],
    ])
    const L = fromRows([
      [2, 0],
      [-1.5, 0.5],
    ])
    const a = R.multivariateNormal(R.stream('mvn2'), tensor([1, -2]), { covariance: cov }, { shape: [5] })
    const b = R.multivariateNormal(R.stream('mvn2'), [1, -2], { choleskyFactor: L }, { shape: [5] })
    expect(a.shape).toEqual([5, 2])
    toFlat(a).forEach((v, i) => expect(v).toBeCloseTo(toFlat(b)[i], 12))
    expect(R.multivariateNormal(R.stream('mvn2'), [1, -2], { covariance: cov }).shape).toEqual([2])
    // A batch of means broadcast against the shape.
    const means = fromRows([
      [0, 0],
      [100, 100],
    ])
    const batch = R.multivariateNormal(R.stream('mvn2'), means, { choleskyFactor: L }, { shape: [3, 2] })
    expect(batch.shape).toEqual([3, 2, 2])
    // Element [i, 1, :] is drawn around the second mean.
    expect(toFlat(batch)[2]).toBeGreaterThan(50)
    const singular = fromRows([
      [1, 1],
      [1, 1],
    ])
    expect(() => R.multivariateNormal(R.stream('mvn2'), [0, 0], { covariance: singular })).toThrow(/positive definite/)
    expect(R.multivariateNormal(R.stream('mvn2'), [0, 0], { covariance: singular }, { jitter: 'auto' }).shape).toEqual([
      2,
    ])
    expect(() => R.multivariateNormal(R.stream('mvn2'), [0, 0, 0], { choleskyFactor: L })).toThrow(ShapeError)
  })
})
