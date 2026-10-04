import { describe, expect, it } from 'vitest'
import {
  fista,
  ista,
  projectBall,
  projectBox,
  projectSimplex,
  projectedGradient,
  proxL1,
  proxL2,
  proxSquaredL2,
} from 'aifn-compute/optim/proximal'
import type { ObjectiveFn } from 'aifn-compute/optim'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

describe('proximal and projected methods', () => {
  // Lasso: ½‖x − c‖² + λ‖x‖₁ has the soft-threshold solution.
  const c = [3, -0.5, 1.2]
  const f: ObjectiveFn = (x) => {
    const d = toFlat(x).map((vi, i) => vi - c[i])
    return { value: 0.5 * d.reduce((s, di) => s + di * di, 0), grad: d }
  }
  it('ISTA and FISTA reach the soft-threshold solution', () => {
    for (const alg of [
      ista(f, proxL1(1), { stepSize: 0.5 }),
      fista(f, proxL1(1), { stepSize: 0.5, backtracking: true }),
    ]) {
      const s = run(alg, { x0: [0, 0, 0] }, 500)
      expect(s.converged).toBe(true)
      close(toFlat(s.x), [2, 0, 0.2], 1e-6)
    }
    expect(toFlat(run(ista(f, proxSquaredL2(1), { stepSize: 0.5 }), { x0: [0, 0, 0] }, 500).x)[0]).toBeCloseTo(1.5, 5)
    expect(run(ista(f, proxL2(1), { stepSize: 0.5 }), { x0: [0, 0, 0] }, 500).converged).toBe(true)
  })

  it('projected gradient stays in the set', () => {
    const s = run(projectedGradient(f, projectBox(0, 1), { stepSize: 0.5 }), { x0: [0.5, 0.5, 0.5] }, 200)
    close(toFlat(s.x), [1, 0, 1], 1e-6)
    close(toFlat(projectSimplex()(tensor([0.5, 0.8, -1]))), [0.35, 0.65, 0], 1e-12)
    expect(Math.hypot(...toFlat(projectBall(1)(tensor([3, 4]))))).toBeCloseTo(1, 12)
  })

  it('satisfy the Algorithm protocol', () => {
    checkProtocol(ista(f, proxL1(1), { stepSize: 0.5 }), { x0: [0, 0, 0] }, { steps: 8 })
    checkProtocol(fista(f, proxL1(1), { stepSize: 0.5, backtracking: true }), { x0: [0, 0, 0] }, { steps: 8 })
    checkProtocol(projectedGradient(f, projectBox(0, 1), { stepSize: 0.5 }), { x0: [0.5, 0.5, 0.5] }, { steps: 8 })
  })
})
