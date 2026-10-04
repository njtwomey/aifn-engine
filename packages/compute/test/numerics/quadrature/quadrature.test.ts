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
  integrateMonteCarlo,
  monteCarlo,
  normalExpectation,
  quasiMonteCarlo,
  simpson,
  sobol,
  trapezoid,
  trapezoidSamples,
} from 'aifn-compute/numerics/quadrature'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { extend, run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const close = (a: number[], b: number[], tol: number) =>
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol))

describe('Gaussian rules match numpy / scipy', () => {
  it('Legendre, Hermite and Laguerre nodes and weights', () => {
    // numpy.polynomial.legendre.leggauss(5), hermite.hermgauss(6), laguerre.laggauss(4); scipy roots_genlaguerre(3, 1.5).
    const l = gaussLegendre(5)
    close(toFlat(l.nodes), [-0.906179845938664, -0.5384693101056831, 0, 0.5384693101056831, 0.906179845938664], 1e-15)
    close(
      toFlat(l.weights),
      [0.2369268850561894, 0.47862867049936625, 0.5688888888888886, 0.47862867049936625, 0.2369268850561894],
      1e-15,
    )
    const h = gaussHermite(6)
    close(
      toFlat(h.nodes),
      [
        -2.3506049736744923, -1.335849074013697, -0.4360774119276165, 0.4360774119276165, 1.335849074013697,
        2.3506049736744923,
      ],
      1e-14,
    )
    close(
      toFlat(h.weights),
      [
        0.004530009905508835, 0.15706732032285647, 0.7246295952243924, 0.7246295952243924, 0.15706732032285647,
        0.004530009905508835,
      ],
      1e-14,
    )
    const g = gaussLaguerre(4)
    close(toFlat(g.nodes), [0.3225476896193924, 1.7457611011583465, 4.536620296921128, 9.395070912301133], 1e-13)
    close(
      toFlat(g.weights),
      [0.6031541043416337, 0.35741869243779956, 0.038887908515005384, 0.0005392947055613296],
      1e-14,
    )
    const ga = gaussLaguerre(3, { alpha: 1.5 })
    close(toFlat(ga.nodes), [1.2204023175588838, 3.808880721467068, 8.470716960974048], 1e-13)
    close(toFlat(ga.weights), [0.7306378943500158, 0.5662491006866058, 0.03245339314251526], 1e-13)
    expect(toFlat(gaussHermite(40, { probabilists: true }).weights).reduce((a, b) => a + b, 0)).toBeCloseTo(
      Math.sqrt(2 * Math.PI),
      12,
    )
  })
  it('an n-point rule is exact to degree 2n − 1', () => {
    expect(integrateGauss((x) => x ** 9 + x ** 8, -1, 2, { n: 5 })).toBeCloseTo(
      2 ** 10 / 10 - 0.1 + 2 ** 9 / 9 + 1 / 9,
      11,
    )
    expect(normalExpectation((x) => x ** 4, 1, 2, { n: 3 })).toBeCloseTo(1 + 6 * 4 + 3 * 16, 10)
  })
})

describe('Newton–Cotes and Romberg', () => {
  it('trapezoid is O(h²), Simpson O(h⁴), Romberg converges fast', () => {
    const e1 = Math.abs(trapezoid(Math.exp, 0, 1, { n: 10 }) - (Math.E - 1))
    const e2 = Math.abs(trapezoid(Math.exp, 0, 1, { n: 20 }) - (Math.E - 1))
    expect(e1 / e2).toBeCloseTo(4, 1)
    const s1 = Math.abs(simpson(Math.exp, 0, 1, { n: 10 }) - (Math.E - 1))
    const s2 = Math.abs(simpson(Math.exp, 0, 1, { n: 20 }) - (Math.E - 1))
    expect(s1 / s2).toBeCloseTo(16, 0)
    const r = integrate(Math.exp, 0, 1, { method: 'romberg' })
    expect(r.converged).toBe(true)
    expect(r.value).toBeCloseTo(Math.E - 1, 13)
    expect(trapezoidSamples([0, 1, 4], [0, 1, 2])).toBe(3)
  })
})

describe('adaptive quadrature', () => {
  it('matches scipy.integrate.quad, including infinite limits', () => {
    expect(integrate((x) => Math.exp(-x * x), -Infinity, Infinity).value).toBeCloseTo(1.7724538509055159, 10)
    expect(integrate((x) => 1 / (1 + x * x), 0, Infinity).value).toBeCloseTo(Math.PI / 2, 10)
    const r = integrate(Math.sqrt, 0, 1)
    expect(r.converged).toBe(true)
    expect(Math.abs(r.value - 2 / 3)).toBeLessThanOrEqual(Math.max(r.error, 1e-12))
    expect(integrate(Math.sin, Math.PI, 0).value).toBeCloseTo(-2, 12)
  })
  it('adaptive Simpson refines near a kink', () => {
    const s = run(
      adaptiveSimpson((x) => Math.abs(x - 0.3), { tolerance: 1e-10 }),
      { a: 0, b: 1 },
      10000,
    )
    expect(s.converged).toBe(true)
    expect(s.value).toBeCloseTo((0.3 ** 2 + 0.7 ** 2) / 2, 9)
  })
  it('protocol: extend equals a longer trace', () => {
    const alg = gaussKronrod((x) => 1 / Math.sqrt(x), { atol: 0, rtol: 0 })
    const record = { value: (s: { value: number }) => s.value }
    const a = trace(alg, { a: 0, b: 1 }, 12, { record })
    const b = extend(trace(alg, { a: 0, b: 1 }, 5, { record }), alg, 7)
    expect(toFlat(b.series.value)).toEqual(toFlat(a.series.value))
  })
})

describe('several dimensions', () => {
  it('product rules integrate polynomials exactly', () => {
    expect(integrate2d((x, y) => x * x * y, [0, 1], [0, 2], { n: 3 })).toBeCloseTo(2 / 3, 13)
    expect(integrate2d((x, y) => x * y ** 3, [0, 1], [0, 1], { n: 4, rule: 'simpson' })).toBeCloseTo(1 / 8, 13)
  })
  it('Halton and Sobol match scipy (unscrambled)', () => {
    close(
      toFlat(halton(5, 3)),
      [0, 0, 0, 0.5, 1 / 3, 0.2, 0.25, 2 / 3, 0.4, 0.75, 1 / 9, 0.6, 0.125, 4 / 9, 0.8],
      1e-15,
    )
    const s = toRows(sobol(16, 21))
    close(
      s[3],
      [
        0.25, 0.75, 0.75, 0.75, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.25, 0.75, 0.25, 0.75,
        0.25, 0.75, 0.75,
      ],
      1e-15,
    )
    close(
      s[11],
      [
        0.4375, 0.5625, 0.1875, 0.6875, 0.8125, 0.0625, 0.6875, 0.6875, 0.6875, 0.0625, 0.9375, 0.3125, 0.1875, 0.1875,
        0.5625, 0.1875, 0.5625, 0.0625, 0.6875, 0.5625, 0.9375,
      ],
      1e-15,
    )
    close(
      s[15],
      [
        0.0625, 0.9375, 0.5625, 0.3125, 0.6875, 0.1875, 0.8125, 0.3125, 0.3125, 0.6875, 0.0625, 0.1875, 0.3125, 0.5625,
        0.9375, 0.8125, 0.9375, 0.9375, 0.3125, 0.6875, 0.8125,
      ],
      1e-15,
    )
  })
  it('Monte Carlo and QMC estimates lie within a few standard errors, QMC far tighter', () => {
    const f = (x: { data: ArrayLike<number> }) => Math.exp(x.data[0] + x.data[1]) // ∫∫ over [0,1]² = (e − 1)²
    const exact = (Math.E - 1) ** 2
    const mc = integrateMonteCarlo(stream(1), f, [0, 0], [1, 1], { n: 20000 })
    expect(Math.abs(mc.value - exact)).toBeLessThan(4 * mc.standardError)
    const q = quasiMonteCarlo(stream(2), f, [0, 0], [1, 1], { n: 1024 })
    expect(Math.abs(q.value - exact)).toBeLessThan(1e-3)
    expect(q.standardError).toBeLessThan(mc.standardError)
    const t = run(monteCarlo(f, { lo: [0, 0], hi: [1, 1] }), undefined, 10, { stream: stream(3) })
    expect(t.n).toBe(1000)
  })
})

describe('the Algorithm protocol', () => {
  it('adaptive Simpson, Gauss–Kronrod and Monte Carlo', () => {
    checkProtocol(
      adaptiveSimpson((x) => Math.abs(x - 0.3), { tolerance: 1e-12 }),
      { a: 0, b: 1 },
      {
        steps: 20,
        record: { value: (s) => s.value },
      },
    )
    checkProtocol(
      gaussKronrod((x) => 1 / Math.sqrt(x), { atol: 0, rtol: 0 }),
      { a: 0, b: 1 },
      { steps: 10 },
    )
    const f = (x: { data: ArrayLike<number> }) => Math.exp(x.data[0] + x.data[1])
    checkProtocol(monteCarlo(f, { lo: [0, 0], hi: [1, 1], batch: 50 }), undefined, {
      steps: 8,
      random: true,
      record: { value: (s) => s.value },
    })
  })
})

describe('integrate: breakpoints and starting panels', () => {
  const width = 0.05
  const peak = (x: number) => Math.exp(-0.5 * ((x - 1.5) / width) ** 2)
  const exact = width * Math.sqrt(2 * Math.PI)
  it('a narrow peak on a wide range is missed from one panel and found with points', () => {
    // The reported failure: both rules agree that f is ~0 on [−40, 40] and the run stops at once.
    expect(integrate(peak, -40, 40).value).toBeLessThan(1e-100)
    const r = integrate(peak, -40, 40, { points: [1.5] })
    expect(r.converged).toBe(true)
    expect(r.value).toBeCloseTo(exact, 10)
    expect(integrate((x) => x * peak(x), -40, 40, { points: [1.5] }).value / r.value).toBeCloseTo(1.5, 9)
    expect(integrate(peak, -40, 40, { panels: 64 }).value).toBeCloseTo(exact, 9)
  })
  it('breakpoints map through the transforms of infinite limits', () => {
    expect(integrate(peak, -Infinity, Infinity, { points: [1.5] }).value).toBeCloseTo(exact, 9)
    expect(integrate(peak, 0, Infinity, { points: [1.5] }).value).toBeCloseTo(exact, 9)
    expect(integrate(peak, -Infinity, 3, { points: [1.5] }).value).toBeCloseTo(exact, 9)
  })
  it('a kink at a breakpoint converges in the first intervals; reversed limits negate', () => {
    const r = integrate((x) => Math.abs(x - 0.3), 0, 1, { points: [0.3] })
    expect(r.value).toBeCloseTo(0.5 * (0.3 ** 2 + 0.7 ** 2), 14)
    expect(r.intervals).toBe(2)
    expect(integrate((x) => Math.abs(x - 0.3), 1, 0, { points: [0.3] }).value).toBeCloseTo(-r.value, 14)
    expect(() => integrate(peak, 0, 1, { panels: 0 })).toThrow(/panels/)
  })
})
