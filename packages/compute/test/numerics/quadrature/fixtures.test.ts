import { describe, expect, it } from 'vitest'
import {
  adaptiveSimpson,
  gaussHermite,
  gaussKronrod,
  gaussLaguerre,
  gaussLegendre,
  halton,
  integrate,
  integrate2d,
  integrateGauss,
  monteCarlo,
  normalExpectation,
  romberg,
  simpson,
  sobol,
  trapezoid,
  trapezoidSamples,
} from 'aifn-compute/numerics/quadrature'
import { stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

type Integral = { a: number; b: number; value: number; singular?: boolean }
type Rule = { nodes: number[]; weights: number[] }
type Fixture = {
  integrals: Record<string, Integral>
  infinite: Record<string, Integral>
  adaptiveSimpson: { integrands: string[] }
  gaussKronrod: { integrands: string[] }
  romberg: { integrands: string[] }
  rules: {
    legendre: Record<string, Rule>
    hermite: Record<string, Rule>
    hermiteProbabilists: Record<string, Rule>
    laguerre: Record<string, Rule>
    genLaguerre: Record<string, Rule & { alpha: number; n: number }>
    integrateGauss: { a: number; b: number; n: number; value: number }
    normalExpectation: { mean: number; sd: number; value: number }
  }
  composite: {
    a: number
    b: number
    n: number
    trapezoid: number
    simpson: number
    samplesX: number[]
    samplesY: number[]
    trapezoidSamples: number
  }
  halton: { n: number; d: number; points: number[][] }
  sobol: { n: number; d: number; points: number[][] }
  monteCarlo: Record<string, { lo: number[]; hi: number[]; value: number }>
  integrate2d: { x: [number, number]; y: [number, number]; value: number }
}
const F = fixture<Fixture>('numerics/quadrature')

/** The integrands of the generator, by name (gen/numerics/quadrature.py). */
const f: Record<string, (x: number) => number> = {
  exp: Math.exp,
  cos: Math.cos,
  poly7: (x) => 3 * x ** 7 - 2 * x ** 4 + x - 5,
  runge: (x) => 1 / (1 + 25 * x * x),
  oscillatory: (x) => Math.sin(20 * x) * Math.exp(-x),
  gaussian: (x) => Math.exp(-0.5 * x * x),
  kink: (x) => Math.abs(x - 0.3),
  sqrt: Math.sqrt,
  power15: (x) => x ** 1.5,
  log: (x) => (x > 0 ? Math.log(x) : 0),
  inverseSqrt: (x) => (x > 0 ? 1 / Math.sqrt(x) : 0),
  logistic: (x) => 1 / (1 + Math.exp(-x)),
  reversed: Math.sin,
  gaussianLine: (x) => Math.exp(-x * x),
  cauchyHalf: (x) => 1 / (1 + x * x),
  gammaThree: (x) => x * x * Math.exp(-x),
  dampedCos: (x) => Math.exp(-x) * Math.cos(x),
  leftTail: Math.exp,
  normalCdf: (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI),
}
const box: Record<string, (x: { data: ArrayLike<number> }) => number> = {
  expSum2: (x) => Math.exp(x.data[0] + x.data[1]),
  gaussian3: (x) => Math.exp(-0.5 * (x.data[0] ** 2 + x.data[1] ** 2 + x.data[2] ** 2)),
  product4: (x) => Math.sin(x.data[0]) * Math.sin(x.data[1]) * Math.sin(x.data[2]) * Math.sin(x.data[3]),
}

const near = (got: number, want: number, rtol: number, atol = 0) =>
  expect(Math.abs(got - want), `${got} vs ${want}`).toBeLessThanOrEqual(atol + rtol * Math.abs(want))
const nearAll = (got: ArrayLike<number>, want: number[], rtol: number, atol = 0) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => near(got[i], w, rtol, atol))
}

describe('adaptive algorithms match scipy.integrate.quad', () => {
  it.each(F.adaptiveSimpson.integrands)('adaptive Simpson: %s', (name) => {
    const { a, b, value } = F.integrals[name]
    const s = run(adaptiveSimpson(f[name], { tolerance: 1e-12 }), { a, b }, 1e6)
    // √x: Simpson's error on [0, h] falls as h^1.5 while the tolerance halves per level, so the intervals at 0 reach
    // the depth limit unresolved (expected of the method); the estimate is still accurate.
    expect(s.converged || name === 'sqrt').toBe(true)
    near(s.value, value, 1e-9, 1e-10)
  })
  it.each(F.gaussKronrod.integrands)('Gauss–Kronrod: %s', (name) => {
    const { a, b, value } = F.integrals[name]
    const s = run(gaussKronrod(f[name], { atol: 1e-13, rtol: 1e-12 }), { a, b }, 2000)
    near(s.value, value, 1e-11, 1e-12)
    // The error estimate is honest: it bounds the true error.
    expect(Math.abs(s.value - value)).toBeLessThanOrEqual(Math.max(s.error, 1e-14 * Math.abs(value)))
  })
  it.each(F.romberg.integrands)('Romberg: %s', (name) => {
    const { a, b, value } = F.integrals[name]
    const s = run(romberg(f[name]), { a, b }, 20)
    expect(s.converged).toBe(true)
    near(s.value, value, 1e-11, 1e-12)
  })
  it.each(Object.keys(F.infinite))('integrate with infinite limits: %s', (name) => {
    const { a, b, value } = F.infinite[name]
    const r = integrate(f[name], a, b, { atol: 1e-13, rtol: 1e-12 })
    expect(r.converged).toBe(true)
    near(r.value, value, 1e-10, 1e-12)
  })
})

describe('Gauss rules match numpy and scipy.special', () => {
  const check = (rule: { nodes: { data: ArrayLike<number> }; weights: { data: ArrayLike<number> } }, ref: Rule) => {
    nearAll(rule.nodes.data, ref.nodes, 1e-12, 1e-14)
    nearAll(rule.weights.data, ref.weights, 1e-10, 1e-300)
  }
  for (const [n, ref] of Object.entries(F.rules.legendre))
    it(`Legendre n = ${n}`, () => check(gaussLegendre(Number(n)), ref))
  for (const [n, ref] of Object.entries(F.rules.hermite))
    it(`Hermite n = ${n}`, () => check(gaussHermite(Number(n)), ref))
  for (const [n, ref] of Object.entries(F.rules.hermiteProbabilists))
    it(`probabilists' Hermite n = ${n}`, () => check(gaussHermite(Number(n), { probabilists: true }), ref))
  for (const [n, ref] of Object.entries(F.rules.laguerre))
    it(`Laguerre n = ${n}`, () => check(gaussLaguerre(Number(n)), ref))
  for (const [key, ref] of Object.entries(F.rules.genLaguerre))
    it(`generalised Laguerre ${key}`, () => check(gaussLaguerre(ref.n, { alpha: ref.alpha }), ref))
  it('integrateGauss and normalExpectation', () => {
    const g = F.rules.integrateGauss
    near(
      integrateGauss((x) => Math.exp(Math.sin(x)), g.a, g.b, { n: g.n }),
      g.value,
      1e-12,
    )
    const e = F.rules.normalExpectation
    near(normalExpectation(Math.cos, e.mean, e.sd), e.value, 1e-12)
  })
})

describe('composite rules and point sets match scipy', () => {
  it('trapezoid, Simpson and trapezoid on samples', () => {
    const c = F.composite
    const g = (x: number) => Math.exp(Math.sin(x))
    near(trapezoid(g, c.a, c.b, { n: c.n }), c.trapezoid, 1e-14)
    near(simpson(g, c.a, c.b, { n: c.n }), c.simpson, 1e-14)
    near(trapezoidSamples(c.samplesY, c.samplesX), c.trapezoidSamples, 1e-14)
  })
  it('Halton and Sobol (unscrambled) match scipy.stats.qmc', () => {
    toRows(halton(F.halton.n, F.halton.d)).forEach((row, i) => nearAll(row, F.halton.points[i], 0, 1e-15))
    toRows(sobol(F.sobol.n, F.sobol.d)).forEach((row, i) => nearAll(row, F.sobol.points[i], 0, 1e-15))
  })
  it('integrate2d matches scipy dblquad', () => {
    const r = F.integrate2d
    near(
      integrate2d((x, y) => Math.exp(-x * y) * Math.cos(x + y), r.x, r.y, { n: 20 }),
      r.value,
      1e-12,
    )
  })
})

describe('Monte Carlo against scipy nquad', () => {
  for (const [name, ref] of Object.entries(F.monteCarlo))
    it(`${name}: within four standard errors`, () => {
      const s = run(monteCarlo(box[name], { lo: ref.lo, hi: ref.hi, batch: 500 }), undefined, 40, {
        stream: stream(7),
      })
      expect(s.n).toBe(20000)
      expect(Math.abs(s.value - ref.value)).toBeLessThan(4 * s.standardError)
      expect(s.standardError).toBeLessThan(0.05 * Math.abs(ref.value))
    })
})
