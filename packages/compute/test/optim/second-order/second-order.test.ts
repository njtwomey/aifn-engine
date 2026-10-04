import { describe, expect, it } from 'vitest'
import {
  bfgs,
  gaussNewton,
  lbfgs,
  leastSquares,
  levenbergMarquardt,
  newton,
  trustRegion,
  type ResidualFunction,
} from 'aifn-compute/optim/second-order'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import type { Algorithm } from 'aifn-compute/foundation/contracts'
import type { IterateState, StartOptions } from 'aifn-compute/optim'
import { checkProtocol } from '../../protocol'
import { rosenbrock } from '../problems'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}
const rosen = rosenbrock()

describe('second-order and quasi-Newton methods reach the Rosenbrock minimum', () => {
  it.each([
    ['newton', () => run(newton(rosen.objective, { hessian: rosen.hessian }), { x0: rosen.start }, 200)],
    ['trust region', () => run(trustRegion(rosen.objective, { hessian: rosen.hessian }), { x0: rosen.start }, 500)],
    ['bfgs', () => run(bfgs(rosen.objective), { x0: rosen.start }, 500)],
    ['lbfgs', () => run(lbfgs(rosen.objective), { x0: rosen.start }, 500)],
  ])('%s', (_, go) => {
    const s = go()
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
  })

  it('pure Newton converges quadratically near the minimum', () => {
    const t = trace(
      newton(rosen.objective, { hessian: rosen.hessian, lineSearch: 'none', tolerance: 0 }),
      { x0: [1.1, 1.2] },
      8,
      { record: { error: (s) => Math.hypot(toFlat(s.x)[0] - 1, toFlat(s.x)[1] - 1) } },
    )
    const e = toFlat(t.series.error)
    // e_{k+1} ≤ C e_k² with a moderate C once close.
    expect(e[5] / e[4] ** 2).toBeLessThan(1e2)
    expect(e[6] / e[5] ** 2).toBeLessThan(1e2)
    expect(e[6]).toBeLessThan(1e-10)
  })
})

describe('least squares', () => {
  // Fit y = a·exp(b·t) to exact data from a = 2, b = −0.5.
  const ts = [0, 0.5, 1, 1.5, 2, 3]
  const residuals: ResidualFunction = (x) => {
    const [a, b] = toFlat(x)
    return {
      residuals: ts.map((t) => a * Math.exp(b * t) - 2 * Math.exp(-0.5 * t)),
      jacobian: ts.map((t) => [Math.exp(b * t), a * t * Math.exp(b * t)]),
    }
  }
  it('Gauss–Newton and Levenberg–Marquardt recover the parameters', () => {
    for (const method of ['gauss-newton', 'levenberg-marquardt'] as const) {
      const r = leastSquares(residuals, [1, 0], { method, tolerance: 1e-12 })
      expect(r.converged).toBe(true)
      close(toFlat(r.x), [2, -0.5], 1e-8)
    }
    expect(run(gaussNewton(residuals), { x0: [1, 0] }, 50).converged).toBe(true)
    expect(run(levenbergMarquardt(residuals, { scaling: 'marquardt' }), { x0: [1, 0] }, 200).converged).toBe(true)
  })

  it('satisfy the Algorithm protocol', () => {
    checkProtocol(gaussNewton(residuals), { x0: [1, 0] }, { steps: 6 })
    checkProtocol(levenbergMarquardt(residuals), { x0: [1, 0] }, { steps: 6 })
  })
})

describe('protocol', () => {
  const record = { value: (s: { value: number }) => s.value }
  it.each<[string, Algorithm<StartOptions, IterateState>]>([
    ['newton', newton(rosen.objective, { hessian: rosen.hessian })],
    ['trustRegion', trustRegion(rosen.objective, { hessian: rosen.hessian })],
    ['bfgs', bfgs(rosen.objective)],
    ['lbfgs', lbfgs(rosen.objective, { memory: 3 })],
  ])('%s satisfies the Algorithm protocol', (_, alg) => {
    checkProtocol(alg, { x0: rosen.start }, { steps: 12, record })
  })
})
