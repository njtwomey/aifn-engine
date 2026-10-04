import { describe, expect, it } from 'vitest'
import { ravel } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { moons, regression1d } from 'aifn-methods/data/synthetic'
import { comparisonModel, fullBatchComparison } from 'aifn-methods/neural/full-batch'

describe('fullBatchComparison', () => {
  const data = moons(stream('moons'), { n: 64, noise: 0.15 })
  const network = { width: 6, depth: 2, activation: 'tanh' } as const

  it('trains every optimiser from the same weights and counts work in full-data gradient evaluations', () => {
    const snaps = [
      ...fullBatchComparison(data, { task: 'classification', network, iterations: 40, batchSize: 16, l2: 1e-3 }),
    ]
    const last = snaps.at(-1)!
    expect(last.runs.map((r) => r.optimiser)).toEqual(['lbfgs', 'gradient-descent', 'adam', 'sgd'])
    expect(last.parameterCount).toBe(comparisonModel({ ...network, inputs: 2 }).size)
    // One initial point: every run starts at the same objective and the same θ.
    const first = last.runs.map((r) => r.loss[0])
    for (const v of first) expect(v).toBeCloseTo(first[0], 12)
    for (const r of last.runs) expect(r.checkpoints[0].theta).toEqual(last.runs[0].checkpoints[0].theta)
    const [lbfgs, gd, adam] = last.runs
    // L-BFGS: one record per iteration, each with its step length, line-search evaluations and sᵀy.
    const steps = lbfgs.iteration.length - 1
    expect(lbfgs.stepSize.length).toBe(steps)
    expect(lbfgs.lineEvaluations.reduce((a, b) => a + b, 1)).toBe(lbfgs.evaluations.at(-1))
    for (let k = 0; k < steps; k++) if (!lbfgs.skipped[k]) expect(lbfgs.curvature[k]).toBeGreaterThan(0)
    // Gradient descent: one evaluation per step; Adam: a 16-example minibatch is a quarter of one.
    expect(gd.evaluations.slice(0, 3)).toEqual([1, 2, 3])
    expect(adam.evaluations.slice(0, 3)).toEqual([0.25, 0.5, 0.75])
    expect(lbfgs.loss.at(-1)).toBeLessThan(gd.loss.at(-1)!)
    expect(lbfgs.score).toBeGreaterThan(0.9)
  })

  it('trains one optimiser alone and fits a 1-d regression', () => {
    const sine = regression1d(stream('sine'), { n: 40, fn: 'sine', noise: 0.05, range: [-3, 3] })
    const [only] = [
      ...fullBatchComparison(sine, { task: 'regression', network, optimisers: ['lbfgs'], iterations: 60 }),
    ].at(-1)!.runs
    expect(only.optimiser).toBe('lbfgs')
    expect(only.score).toBeLessThan(0.15)
    const { unravel } = comparisonModel({ ...network, inputs: 1 })
    expect(ravel(unravel(only.checkpoints.at(-1)!.theta)).vector).toEqual(only.checkpoints.at(-1)!.theta)
  })
})
