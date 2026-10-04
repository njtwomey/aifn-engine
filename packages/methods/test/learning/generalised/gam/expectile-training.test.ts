import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { curve1d } from 'aifn-methods/data/synthetic'
import {
  expectileGam,
  expectileProblem,
  expectileTrainingRun,
  gamModel,
  gamProblem,
  s,
} from 'aifn-methods/learning/generalised/gam'

const d = curve1d(stream('train'), { case: 'skewed', n: 150 })
const data = { x: d.x as Tensor, y: d.y as Tensor }
const terms = [s(0, { k: 10 })]

describe('expectileProblem', () => {
  const problem = expectileProblem(gamProblem({ terms, method: 'fixed', lambda: 0.3 }, data), 0.8)
  it('autodiff of J_τ equals the closed-form gradient', () => {
    const beta = Float64Array.from({ length: problem.design.P }, (_, i) => 0.1 * Math.sin(i + 1))
    const { value, grad } = valueAndGrad((b: Value) => problem.objective(b))(fromData(beta, [beta.length]))
    const closed = problem.gradient(beta)
    toFlat(grad as Tensor).forEach((v, i) => expect(v).toBeCloseTo(closed[i], 10))
    const v = typeof value === 'number' ? value : toFlat(value as Tensor)[0]
    expect(v).toBeCloseTo(problem.evaluate(beta).objective, 12)
  })

  it('its optimum is the expectile GAM at the same λ, with zero gradient', () => {
    const model = expectileGam({ terms, method: 'fixed', lambda: 0.3, tau: 0.8 }).fit(data)
    const beta = problem.optimum.beta
    toFlat(model.coefficients).forEach((v, i) => expect(beta[i]).toBeCloseTo(v, 8))
    expect(Math.hypot(...problem.gradient(beta))).toBeLessThan(1e-8)
    // gamModel reads the asymmetric weights at β: the EDF of the weighted fit.
    expect(gamModel(problem, beta).edf).toBeCloseTo(model.edf, 6)
  })
})

describe('expectileTrainingRun', () => {
  it('Adam and L-BFGS reach the LAWS minimiser of J_τ at the λ LAWS chose', () => {
    const run = expectileTrainingRun(
      { terms },
      data,
      [0.2, 0.9],
      [
        { method: 'lbfgs', steps: 300 },
        { method: 'adam', options: { stepSize: 0.05 }, steps: 3000 },
        { method: 'p-irls', steps: 0 },
      ],
    )
    for (const r of run.runs) {
      const [lb, ad, laws] = r.fits
      const best = r.optimum.objective
      expect(lb.states.at(-1)!.objective - best).toBeLessThan(1e-9)
      expect(ad.states.at(-1)!.objective - best).toBeLessThan(1e-4)
      expect(laws.states.at(-1)!.objective).toBeCloseTo(best, 10)
      expect(laws.states.at(-1)!.converged).toBe(true)
      expect(ad.states[0].t).toBe(0)
      expect(lb.states.at(-1)!.below).toBeCloseTo(laws.states.at(-1)!.below, 2)
      expect(Number.isFinite(r.criterion)).toBe(true)
    }
  })
})
