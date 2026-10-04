import { describe, expect, it } from 'vitest'
import {
  akima,
  bspline,
  bsplineBasis,
  chebyshevNodes,
  cubicSpline,
  cyclicBsplineBasis,
  cyclicDifferencePenalty,
  derivativePenalty,
  differenceMatrix,
  evaluatePiecewise,
  integratePiecewise,
  interpolatingPolynomial,
  leastSquaresSpline,
  lebesgueFunction,
  linearInterpolant,
  pchip,
  pspline,
  psplineGcvPath,
  smoothingSpline,
  tensorProductBasis,
  tensorProductPenalties,
  thinPlateRegressionBasis,
  thinPlateSpline,
  uniformKnots,
} from 'aifn-compute/numerics/interpolate'
import { linspace, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Fixture = {
  x: number[]
  y: number[]
  t: number[]
  ts: number[]
  cubic: Record<string, { v: number[]; d1: number[]; d2: number[]; y?: number[] }>
  pchip: number[]
  akima: number[]
  makima: number[]
  smoothing: { lam: number; w: number[]; v: number[]; vw: number[] }
  polynomial: { v: number[]; d1: number[] }
  bspline: { knots: number[]; x: number[]; design: Record<string, number[][]> }
  lsq: { x: number[]; y: number[]; coef: number[]; v: number[] }
  pspline: { coef: number[]; edf: number; gcv: number; lam: number; segments: number }
  tps: { x: number[][]; y: number[]; g: number[][]; v: number[]; vs: number[] }
}
const F = fixture<Fixture>('numerics/interpolate')

function close(actual: number[], expected: number[], tol: number) {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol * (1 + Math.abs(expected[i]))))
}

const x = tensor(F.x)
const y = tensor(F.y)
const t = tensor(F.t)

describe('interpolants', () => {
  for (const bc of ['not-a-knot', 'natural', 'clamped'] as const) {
    it(`cubic spline (${bc}) matches scipy with derivatives`, () => {
      const pp = cubicSpline(x, y, { bc })
      const ref = F.cubic[bc]
      close(toFlat(evaluatePiecewise(pp, t)), ref.v, 1e-9)
      close(toFlat(evaluatePiecewise(pp, t, { derivative: 1 })), ref.d1, 1e-8)
      close(toFlat(evaluatePiecewise(pp, t, { derivative: 2 })), ref.d2, 1e-8)
    })
  }
  it('cubic spline with given end slopes and periodic ends', () => {
    close(toFlat(evaluatePiecewise(cubicSpline(x, y, { bc: { first: [0.5, -1] } }), t)), F.cubic.first.v, 1e-9)
    const inside = F.t.map((v, i) => [v, i] as const).filter(([v]) => v >= F.x[0] && v <= F.x.at(-1)!)
    const pp = cubicSpline(x, tensor(F.cubic.periodic.y!), { bc: 'periodic' })
    const got = toFlat(evaluatePiecewise(pp, tensor(inside.map(([v]) => v))))
    close(
      got,
      inside.map(([, i]) => F.cubic.periodic.v[i]),
      1e-9,
    )
  })
  it('PCHIP and Akima match scipy', () => {
    close(toFlat(evaluatePiecewise(pchip(x, y), t)), F.pchip, 1e-9)
    close(toFlat(evaluatePiecewise(akima(x, y), t)), F.akima, 1e-9)
    close(toFlat(evaluatePiecewise(akima(x, y, { method: 'makima' }), t)), F.makima, 1e-9)
  })
  it('the smoothing spline matches scipy, with and without weights', () => {
    const ts = tensor(F.ts)
    close(toFlat(evaluatePiecewise(smoothingSpline(x, y, { lambda: F.smoothing.lam }).spline, ts)), F.smoothing.v, 1e-8)
    const sw = smoothingSpline(x, y, { lambda: F.smoothing.lam, weights: tensor(F.smoothing.w) })
    close(toFlat(evaluatePiecewise(sw.spline, ts)), F.smoothing.vw, 1e-8)
  })
  it('the interpolating polynomial matches scipy and interpolates', () => {
    const p = interpolatingPolynomial(x, y)
    close(toFlat(p.evaluate(tensor(F.ts))), F.polynomial.v, 1e-7)
    close(toFlat(p.evaluate(tensor(F.ts), 1)), F.polynomial.d1, 1e-6)
    close(toFlat(p.evaluate(x)), F.y, 1e-12)
  })
  it('Chebyshev nodes have a small Lebesgue constant; equispaced ones do not', () => {
    const grid = linspace(-1, 1, 2001)
    const cheb = Math.max(...toFlat(lebesgueFunction(chebyshevNodes(15), grid)))
    const equi = Math.max(...toFlat(lebesgueFunction(linspace(-1, 1, 15), grid)))
    expect(cheb).toBeLessThan(3)
    expect(equi).toBeGreaterThan(200)
  })
  it('integrates piecewise polynomials exactly', () => {
    const pp = linearInterpolant(tensor([0, 1, 3]), tensor([0, 2, 2]))
    expect(integratePiecewise(pp, 0, 3)).toBeCloseTo(1 + 4, 12)
  })
})

describe('B-splines', () => {
  const knots = tensor(F.bspline.knots)
  const xb = tensor(F.bspline.x)
  it('the basis and its derivatives match scipy', () => {
    for (const nu of [0, 1, 2]) {
      const B = toRows(bsplineBasis(xb, knots, 3, { derivative: nu }))
      close(B.flat(), F.bspline.design[String(nu)].flat(), 1e-9)
    }
  })
  it('least-squares splines match scipy', () => {
    const s = leastSquaresSpline(tensor(F.lsq.x), tensor(F.lsq.y), knots, 3)
    close(toFlat(s.coefficients), F.lsq.coef, 1e-8)
    close(toFlat(s.evaluate(xb)), F.lsq.v, 1e-8)
  })
  it('difference matrices and the derivative penalty', () => {
    expect(toRows(differenceMatrix(4, 2))).toEqual([
      [1, -2, 1, 0],
      [0, 1, -2, 1],
    ])
    expect(toRows(differenceMatrix(3, 1))[0]).toEqual([-1, 1, 0])
    // ∫(f″)² for f = x² on [0, 1]: f″ = 2, so 4. Represent x² exactly in a clamped quadratic basis.
    const k = tensor([0, 0, 0, 1, 1, 1])
    const S = toRows(derivativePenalty(k, 2, 2))
    const c = [0, 0, 1] // x² in the Bernstein basis of degree 2
    let q = 0
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) q += c[a] * S[a][b] * c[b]
    expect(q).toBeCloseTo(4, 10)
    expect(toFlat(bspline(k, 2, tensor(c)).evaluate(tensor([0.5])))[0]).toBeCloseTo(0.25, 12)
    const cyc = toRows(cyclicDifferencePenalty(5, 2))
    expect(cyc.map((r) => r.reduce((a, b) => a + b, 0))).toEqual([0, 0, 0, 0, 0])
  })
  it('cyclic bases wrap and sum to one', () => {
    const B = toRows(cyclicBsplineBasis(tensor([0, 0.3, 1, 2]), 0, 1, 6))
    B.forEach((r) => expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12))
    close(B[0], B[2], 1e-12)
  })
  it('tensor products', () => {
    const A = tensor([[1, 2]])
    const B = tensor([[3, 4, 5]])
    expect(toRows(tensorProductBasis(A, B))).toEqual([[3, 4, 5, 6, 8, 10]])
    const [S1, S2] = tensorProductPenalties(
      tensor([
        [1, 0],
        [0, 2],
      ]),
      tensor([[1]]),
    )
    expect(S1.shape).toEqual([2, 2])
    expect(S2.shape).toEqual([2, 2])
  })
})

describe('P-splines', () => {
  it('match a direct penalised least-squares solve', () => {
    const P = F.pspline
    const fit = pspline(tensor(F.lsq.x), tensor(F.lsq.y), { segments: P.segments, lambda: P.lam, range: [0, 10] })
    close(toFlat(fit.coefficients), P.coef, 1e-8)
    expect(fit.edf).toBeCloseTo(P.edf, 8)
    expect(fit.gcv).toBeCloseTo(P.gcv, 8)
  })
  it('GCV picks the minimum of the GCV path', () => {
    const xs = tensor(F.lsq.x)
    const ys = tensor(F.lsq.y)
    const fit = pspline(xs, ys, { lambda: 'gcv' })
    const path = toFlat(psplineGcvPath(xs, ys, linspace(-6, 6, 49)).gcv)
    expect(fit.gcv).toBeLessThanOrEqual(Math.min(...path) + 1e-12)
    expect(toFlat(fit.standardError(tensor([5])))[0]).toBeGreaterThan(0)
  })
})

describe('thin-plate splines', () => {
  it('match scipy RBFInterpolator, interpolating and smoothing', () => {
    const x2 = tensor(F.tps.x)
    const y2 = tensor(F.tps.y)
    close(toFlat(thinPlateSpline(x2, y2).evaluate(tensor(F.tps.g))), F.tps.v, 1e-8)
    close(toFlat(thinPlateSpline(x2, y2, { smoothing: 0.1 }).evaluate(tensor(F.tps.g))), F.tps.vs, 1e-8)
  })
  it('the regression basis reproduces its design at the data and leaves affine functions unpenalised', () => {
    const x2 = tensor(F.tps.x)
    const b = thinPlateRegressionBasis(x2, 8)
    close(toRows(b.evaluate(x2)).flat(), toRows(b.design).flat(), 1e-9)
    const S = toRows(b.penalty)
    expect(S[7].every((v) => v === 0)).toBe(true)
    expect(b.nullSpace).toBe(3)
  })
  it('uniform knots give a partition of unity', () => {
    const B = toRows(bsplineBasis(linspace(0, 1, 7), uniformKnots(0, 1, 5), 3))
    B.forEach((r) => expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12))
  })
})
