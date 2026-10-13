import { describe, expect, it } from 'vitest'
import {
  differenceOperator,
  discrepancyPrinciple,
  filterFactors,
  lCurve,
  landweberSteps,
  tikhonov,
  tikhonovPath,
} from 'aifn-compute/numerics/regularisation'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

type Standard = {
  lambda: number
  x: number[]
  residualNorm: number
  solutionNorm: number
  filterFactors: number[]
  singularValues: number[]
  ridge?: number[]
}
type Curve = {
  lambdas: number[]
  residualNorm: number[]
  solutionNorm: number[]
  curvature: number[]
  cornerIndex: number
}
type Fixture = {
  A: number[][]
  b: number[]
  noiseNorm: number
  tikhonov: Standard[]
  general: { lambda: number; x: number[]; residualNorm: number; solutionNorm: number }[]
  wide: { A: number[][]; b: number[]; cases: Standard[] }
  lCurve: Curve
  lCurveDefault: Curve
  discrepancy: Record<'tau1' | 'tau1.5' | 'sigma', Standard & { target: number }>
  landweber: {
    step: number
    iterates: Record<string, { x: number[]; residualNorm: number; solutionNorm: number; filterFactors: number[] }>
  }
}
const F = fixture<Fixture>('numerics/regularisation')

/** Every entry within `rel` of the expected, relative to the largest expected magnitude. */
const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, rel: number) => {
  expect(actual.length).toBe(expected.length)
  const scale = Math.max(...Array.from(expected, Math.abs), 1e-300)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(rel * scale)
}
const relClose = (actual: number, expected: number, rel: number) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected))

describe('tikhonov', () => {
  it('matches the SVD filter of numpy and scikit-learn Ridge(alpha = lambda, fit_intercept=False)', () => {
    for (const c of F.tikhonov) {
      const r = tikhonov(F.A, F.b, { lambda: c.lambda })
      // At lambda = 0 the solution carries the condition number (about 1e5) into its error.
      const tol = c.lambda === 0 ? 1e-9 : 1e-11
      close(toFlat(r.x), c.x, tol)
      if (c.ridge) close(toFlat(r.x), c.ridge, 1e-9)
      relClose(r.residualNorm, c.residualNorm, 1e-9)
      relClose(r.solutionNorm, c.solutionNorm, tol)
      // LAPACK's smallest singular values carry an absolute error near eps * s_1 (relative 1e-11 here).
      close(toFlat(r.filterFactors!), c.filterFactors, 1e-10)
      close(toFlat(r.singularValues!), c.singularValues, 1e-13)
    }
  })

  it('handles a wide matrix', () => {
    for (const c of F.wide.cases) {
      const r = tikhonov(F.wide.A, F.wide.b, { lambda: c.lambda })
      close(toFlat(r.x), c.x, 1e-9)
      expect(Math.abs(r.residualNorm - c.residualNorm)).toBeLessThan(1e-12)
    }
  })

  it('with a regularisation matrix matches the stacked least-squares solution', () => {
    const L = differenceOperator(F.A[0].length)
    for (const c of F.general) {
      const r = tikhonov(F.A, F.b, { lambda: c.lambda, L })
      close(toFlat(r.x), c.x, 1e-9)
      relClose(r.residualNorm, c.residualNorm, 1e-8)
      relClose(r.solutionNorm, c.solutionNorm, 1e-8)
      expect(r.filterFactors).toBeNull()
    }
    // L = I through the stacked route agrees with the SVD route.
    const I = Array.from({ length: 3 }, (_, i) => [0, 1, 2].map((j) => (i === j ? 1 : 0)))
    const A = [
      [1, 2, 0],
      [0, 1, 1],
      [1, 0, 1],
      [2, 1, 0],
    ]
    const b = [1, 2, 3, 4]
    close(toFlat(tikhonov(A, b, { lambda: 0.3, L: I }).x), toFlat(tikhonov(A, b, { lambda: 0.3 }).x), 1e-13)
  })

  it('the path agrees with single solves', () => {
    const lambdas = F.tikhonov.map((c) => c.lambda)
    const path = tikhonovPath(F.A, F.b, lambdas)
    const rows = toRows(path.x)
    F.tikhonov.forEach((c, j) => {
      close(rows[j], c.x, c.lambda === 0 ? 1e-9 : 1e-11)
      relClose(toFlat(path.residualNorm)[j], c.residualNorm, 1e-9)
      close(toRows(path.filterFactors)[j], c.filterFactors, 1e-10)
    })
    expect(path.x.shape).toEqual([lambdas.length, F.A[0].length])
  })

  it('filter factors, difference operators and errors', () => {
    close(toFlat(filterFactors([2, 1, 0], 1)), [0.8, 0.5, 0], 1e-15)
    close(toFlat(filterFactors([2, 0], 0)), [1, 0], 0)
    expect(toRows(differenceOperator(4, 2))).toEqual([
      [1, -2, 1, 0],
      [0, 1, -2, 1],
    ])
    expect(toRows(differenceOperator(3, 3 - 1)).length).toBe(1)
    expect(toRows(differenceOperator(2, 0))).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(() => tikhonov(F.A, F.b, { lambda: -1 })).toThrow(DomainError)
    expect(() => tikhonov(F.A, [1, 2], { lambda: 1 })).toThrow(ShapeError)
    expect(() => tikhonov(F.A, F.b, { lambda: 1, L: [[1, 2]] })).toThrow(ShapeError)
    expect(() => differenceOperator(3, 3)).toThrow(DomainError)
  })
})

describe('choosing the weight', () => {
  it('the L-curve curvature matches the parametric formula, and the corner its maximum', () => {
    for (const c of [F.lCurve, F.lCurveDefault]) {
      const curve = lCurve(F.A, F.b, c === F.lCurve ? c.lambdas : undefined)
      close(toFlat(curve.lambdas), c.lambdas, 1e-10)
      close(toFlat(curve.residualNorm), c.residualNorm, 1e-9)
      close(toFlat(curve.solutionNorm), c.solutionNorm, 1e-9)
      const kappa = toFlat(curve.curvature)
      c.curvature.forEach((k, j) => expect(Math.abs(kappa[j] - k)).toBeLessThanOrEqual(1e-7 * Math.max(1, Math.abs(k))))
      expect(curve.corner.index).toBe(c.cornerIndex)
      expect(curve.corner.lambda).toBe(toFlat(curve.lambdas)[c.cornerIndex])
    }
    expect(() => lCurve(F.A, F.b, [0, 1])).toThrow(DomainError)
  })

  it('the discrepancy principle matches brentq', () => {
    const cases = [
      [F.discrepancy.tau1, { delta: F.noiseNorm }],
      [F.discrepancy['tau1.5'], { delta: F.noiseNorm, tau: 1.5 }],
      [F.discrepancy.sigma, { sigma: 1e-3 }],
    ] as const
    for (const [c, options] of cases) {
      const d = discrepancyPrinciple(F.A, F.b, options)
      expect(d.attained).toBe(true)
      relClose(d.lambda, c.lambda, 1e-10)
      relClose(d.target, c.target, 1e-14)
      relClose(d.residualNorm, c.target, 1e-12)
      close(toFlat(d.x), c.x, 1e-10)
    }
  })

  it('reports a target that no weight attains', () => {
    const big = discrepancyPrinciple(F.A, F.b, { delta: 1e3 })
    expect(big.attained).toBe(false)
    expect(big.lambda).toBe(Infinity)
    expect(toFlat(big.x).every((v) => v === 0)).toBe(true)
    const A = [
      [1, 0],
      [0, 1],
      [0, 0],
    ]
    const small = discrepancyPrinciple(A, [1, 1, 0.5], { delta: 0.1 })
    expect(small.attained).toBe(false)
    expect(small.lambda).toBe(0)
    expect(small.residualNorm).toBeCloseTo(0.5, 14)
    expect(() => discrepancyPrinciple(A, [1, 1, 0.5], {})).toThrow(DomainError)
    expect(() => discrepancyPrinciple(A, [1, 1, 0.5], { delta: 1, tau: 0.5 })).toThrow(DomainError)
  })
})

describe('landweberSteps', () => {
  it('matches the plain recursion and its filter factors', () => {
    const alg = landweberSteps({ A: F.A, b: F.b })
    for (const [k, c] of Object.entries(F.landweber.iterates)) {
      const s = run(alg, undefined, Number(k))
      expect(s.t).toBe(Number(k))
      expect(s.step).toBeCloseTo(F.landweber.step, 10)
      close(toFlat(s.x), c.x, 1e-11)
      relClose(s.residualNorm, c.residualNorm, 1e-11)
      relClose(s.solutionNorm, c.solutionNorm, 1e-11)
      close(toFlat(s.filterFactors), c.filterFactors, 1e-11)
      expect(s.diverged).toBe(false)
    }
  })

  it('converges to the least-squares solution and flags a step too large', () => {
    const A = [
      [2, 0],
      [0, 1],
      [1, 1],
    ]
    const b = [1, 2, 3]
    const s = run(landweberSteps({ A, b }), undefined, 2000)
    expect(s.converged).toBe(true)
    close(toFlat(s.x), toFlat(tikhonov(A, b, { lambda: 0 }).x), 1e-9)
    const bad = run(landweberSteps({ A, b, step: 0.5 }), { x0: [0, 0] }, 200)
    expect(bad.diverged).toBe(true)
    expect(() => landweberSteps({ A, b, step: 0 })).toThrow(DomainError)
    expect(() => landweberSteps({ A, b: [1, 2] })).toThrow(ShapeError)
  })

  it('satisfies the Algorithm protocol', () => {
    checkProtocol(landweberSteps({ A: F.A, b: F.b }), undefined, { steps: 8 })
  })
})
