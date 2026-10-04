import { describe, expect, it } from 'vitest'
import { lbfgs, owlqn, pseudoGradient } from 'aifn-compute/optim/second-order'
import { fista, proxL1 } from 'aifn-compute/optim/proximal'
import { minimize } from 'aifn-compute/optim/minimize'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import type { ObjectiveFn } from 'aifn-compute/foundation/contracts'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkProtocol } from '../../protocol'
import { fixture } from '../../fixtures'
import { rosenbrock } from '../problems'

type Golden = {
  x: number[][]
  y: number[]
  lasso: { alpha: number; coef: number[] }[]
  elastic_net: { alpha: number; l1Ratio: number; coef: number[] }[]
}
const golden = fixture<Golden>('optim/second-order')

/** (1/(2n))‖y − Xw‖² + ½·l2‖w‖², with its gradient. */
function leastSquares(X: number[][], y: number[], l2 = 0): ObjectiveFn {
  const n = X.length
  const d = X[0].length
  return (w) => {
    const v = toFlat(w)
    const g = new Float64Array(d)
    let value = 0
    for (let i = 0; i < n; i++) {
      let r = -y[i]
      for (let j = 0; j < d; j++) r += X[i][j] * v[j]
      value += (r * r) / (2 * n)
      for (let j = 0; j < d; j++) g[j] += (r * X[i][j]) / n
    }
    for (let j = 0; j < d; j++) {
      value += 0.5 * l2 * v[j] * v[j]
      g[j] += l2 * v[j]
    }
    return { value, grad: fromData(g, [d]) }
  }
}

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

describe('owlqn', () => {
  it('the pseudo-gradient picks the one-sided derivative that descends, else 0', () => {
    const v = pseudoGradient([1, -1, 0, 0, 0], [0.5, 0.5, 2, -2, 0.3], 1)
    expect(Array.from(v)).toEqual([1.5, -0.5, 1, -1, 0])
  })

  it.each(golden.lasso.map((c) => [c.alpha, c] as const))('matches scikit-learn Lasso (α = %s)', (_, c) => {
    const s = run(
      owlqn(leastSquares(golden.x, golden.y), { l1: c.alpha, tolerance: 1e-10 }),
      { x0: Array(8).fill(0) },
      500,
    )
    expect(s.converged).toBe(true)
    close(toFlat(s.x), c.coef, 1e-6)
    // Exact zeros where the lasso solution is zero.
    c.coef.forEach((w, j) => {
      if (w === 0) expect(toFlat(s.x)[j]).toBe(0)
    })
    expect(s.nonzero).toBe(c.coef.filter((w) => w !== 0).length)
  })

  it.each(golden.elastic_net.map((c) => [c.alpha, c.l1Ratio, c] as const))(
    'matches scikit-learn ElasticNet (α = %s, ρ = %s)',
    (alpha, rho, c) => {
      const f = leastSquares(golden.x, golden.y, alpha * (1 - rho))
      const s = run(owlqn(f, { l1: alpha * rho, tolerance: 1e-10 }), { x0: Array(8).fill(0) }, 500)
      expect(s.converged).toBe(true)
      close(toFlat(s.x), c.coef, 1e-6)
    },
  )

  it('agrees with FISTA (proximal gradient) on a lasso and satisfies the KKT conditions', () => {
    const f = leastSquares(golden.x, golden.y)
    const lambda = 0.2
    const a = run(owlqn(f, { l1: lambda, tolerance: 1e-11 }), { x0: Array(8).fill(1) }, 500)
    const b = run(fista(f, proxL1(lambda), { stepSize: 0.2, tolerance: 1e-12 }), { x0: Array(8).fill(0) }, 20000)
    close(toFlat(a.x), toFlat(b.x), 1e-6)
    const g = toFlat(f(a.x).grad as Tensor)
    toFlat(a.x).forEach((w, j) => {
      if (w !== 0) expect(Math.abs(g[j] + lambda * Math.sign(w))).toBeLessThan(1e-8)
      else expect(Math.abs(g[j])).toBeLessThanOrEqual(lambda + 1e-12)
    })
  })

  it('takes exactly the steps of L-BFGS when l1 = 0', () => {
    const rosen = rosenbrock()
    const a = trace(lbfgs(rosen.objective), { x0: rosen.start }, 30, { record: { x: (s) => s.x, f: (s) => s.value } })
    const b = trace(owlqn(rosen.objective), { x0: rosen.start }, 30, { record: { x: (s) => s.x, f: (s) => s.value } })
    expect(Array.from(toFlat(b.series.f))).toEqual(Array.from(toFlat(a.series.f)))
    expect(Array.from(toFlat(b.series.x))).toEqual(Array.from(toFlat(a.series.x)))
  })

  it('sparsifies as C grows, and minimize dispatches to it', () => {
    const f = leastSquares(golden.x, golden.y)
    const nnz = [0.01, 0.3, 1, 5].map((l1) => minimize(f, Array(8).fill(0), { method: 'owlqn', l1 }).state)
    const counts = nnz.map((s) => (s as unknown as { nonzero: number }).nonzero)
    for (let k = 1; k < counts.length; k++) expect(counts[k]).toBeLessThanOrEqual(counts[k - 1])
    expect(counts.at(-1)).toBe(0)
  })

  it('follows the Algorithm protocol', () => {
    checkProtocol(owlqn(leastSquares(golden.x, golden.y), { l1: 0.1 }), { x0: Array(8).fill(0.5) }, { steps: 10 })
  })
})
