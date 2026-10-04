import { describe, expect, it } from 'vitest'
import { backtracking, strongWolfe } from 'aifn-compute/optim/line-search'
import { tensor } from 'aifn-compute/foundation/tensor'
import { rosenbrock } from '../problems'

const rosen = rosenbrock()

describe('line searches', () => {
  it('backtracking satisfies Armijo and strong Wolfe satisfies both conditions', () => {
    const x = [-1.2, 1]
    const p = rosen.objective(tensor(x)).grad.map((v) => -v)
    const b = backtracking(rosen.objective, x, p)
    expect(b.converged && b.armijo).toBe(true)
    expect(b.trials.length).toBeGreaterThan(1)
    const w = strongWolfe(rosen.objective, x, p, { c2: 0.1 })
    expect(w.converged && w.armijo && w.curvature).toBe(true)
    expect(Math.abs(w.trials.at(-1)!.slope)).toBeLessThanOrEqual(0.1 * Math.abs(w.initialSlope))
  })

  it('fails visibly on an ascent direction', () => {
    const r = backtracking(rosen.objective, [0, 0], [-1, 0])
    expect(r.converged).toBe(false)
    expect(r.alpha).toBe(0)
  })
})
