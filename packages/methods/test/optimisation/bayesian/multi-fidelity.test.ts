import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { matern52 } from 'aifn-compute/learning/kernels'
import { forrester } from 'aifn-methods/data/objectives'
import {
  bayesianAlgorithms,
  expectedImprovement,
  maxValueEntropySearch,
  multiFidelityBayesianOptimisationSteps,
  type MultiFidelityBayesianOptimisationState,
} from 'aifn-methods/optimisation/bayesian'
import { expectProtocol } from '../../protocol'

const flat = (t: Tensor) => Array.from(toFlat(t) as ArrayLike<number>)
const row = (t: Tensor, i: number) => {
  const [, m] = t.shape
  return flat(t).slice(i * m, i * m + m)
}

const hi = forrester()
const lo = forrester({ fidelity: 'low' })
const objective = (x: Tensor, z: number) => (z === 0 ? lo : hi).value(tensor([x.data[0] as number]))
const kernels = [matern52({ lengthscale: 0.2, variance: 1 }), matern52({ lengthscale: 0.5, variance: 0.5 })]
const base = { grid: 41, refit: false, kernels }

describe('multiFidelityBayesianOptimisationSteps', () => {
  it('the initial design is nested, the cost adds up, and every fidelity has a posterior and an acquisition', () => {
    const alg = multiFidelityBayesianOptimisationSteps(objective, hi, { cost: [1, 10] }, base)
    const tr = trace(alg, undefined, 3, { stream: stream(0), keep: 'all' })
    const s0 = tr.steps[0] as MultiFidelityBayesianOptimisationState
    // Defaults: (2d + 1) 2^(M - 1 - t) points, 6 low and 3 high; the high ones are the first three low ones.
    expect(s0.fidelity).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1])
    expect(flat(s0.x).slice(6)).toEqual(flat(s0.x).slice(0, 3))
    expect(s0.cost).toBe(6 + 30)
    expect(s0.mean.shape).toEqual([2, 41])
    expect(s0.acquisition.shape).toEqual([2, 41])
    expect(s0.best).toBe(Math.min(...flat(s0.y).slice(6)))
    for (const s of tr.steps.slice(1) as MultiFidelityBayesianOptimisationState[]) {
      const prev = tr.steps[s.t - 1] as MultiFidelityBayesianOptimisationState
      expect(s.cost).toBe(prev.cost + [1, 10][prev.nextFidelity])
      expect(s.fidelity.at(-1)).toBe(prev.nextFidelity)
      expect(flat(s.x).at(-1)).toBe(flat(prev.next)[0])
    }
  })

  it('mf-mes: the top row is max-value entropy search per cost, and no fidelity tells more than the top', () => {
    const alg = multiFidelityBayesianOptimisationSteps(
      objective,
      hi,
      { cost: [1, 4] },
      { ...base, acquisition: 'mf-mes' },
    )
    const s = run(alg, undefined, 2, { stream: stream(1) })
    const mes = flat(maxValueEntropySearch(tensor(row(s.mean, 1)), tensor(row(s.sd, 1)), { samples: s.maxValues! }))
    row(s.acquisition, 1).forEach((v, i) => expect(v).toBeCloseTo(mes[i] / 4, 12))
    // The low fidelity's information about y* (times its cost) never exceeds the top's (data processing).
    const top = row(s.acquisition, 1).map((v) => v * 4)
    row(s.acquisition, 0).forEach((v, i) => expect(v).toBeLessThanOrEqual(top[i] + 1e-6))
    expect(row(s.acquisition, 0).some((v) => v > 0)).toBe(true)
  })

  it('mf-mes: a fidelity that is an exact multiple of the top tells as much as the top', () => {
    // f_1 = 2 f_0 exactly; a discrepancy kernel of tiny variance makes the two almost perfectly correlated.
    const f = (x: Tensor, z: number) => (z === 0 ? 1 : 2) * Math.sin(5 * (x.data[0] as number))
    const k = [matern52({ lengthscale: 0.3, variance: 1 }), matern52({ lengthscale: 0.3, variance: 1e-10 })]
    const alg = multiFidelityBayesianOptimisationSteps(
      f,
      { lo: [0], hi: [1] },
      { cost: [1, 1] },
      { grid: 21, refit: false, kernels: k, acquisition: 'mf-mes', init: [5, 2] },
    )
    const s = run(alg, undefined, 0, { stream: stream(2) })
    expect(s.rho[0]).toBeCloseTo(2, 4)
    const [low, top] = [row(s.acquisition, 0), row(s.acquisition, 1)]
    low.forEach((v, i) => expect(Math.abs(v - top[i])).toBeLessThan(0.02 + 0.02 * top[i]))
  })

  it('ei-per-cost: the top row is EI per cost; lower rows are EI discounted by the correlation', () => {
    const alg = multiFidelityBayesianOptimisationSteps(
      objective,
      hi,
      { cost: [1, 10] },
      { ...base, acquisition: 'ei-per-cost' },
    )
    const s = run(alg, undefined, 2, { stream: stream(3) })
    const ei = flat(expectedImprovement(tensor(row(s.mean, 1)), tensor(row(s.sd, 1)), s.best))
    row(s.acquisition, 1).forEach((v, i) => expect(v).toBeCloseTo(ei[i] / 10, 12))
    row(s.acquisition, 0).forEach((v, i) => expect(v).toBeLessThanOrEqual(ei[i] + 1e-12))
    // The choice is the largest score over every fidelity and grid point.
    const all = [...row(s.acquisition, 0), ...row(s.acquisition, 1)]
    expect(row(s.acquisition, s.nextFidelity)[s.nextIndex]).toBe(Math.max(...all))
  })

  it('mf-gp-ucb: the score is the tightest bound, and the fidelity follows the thresholds gamma', () => {
    const make = (gamma: number) =>
      multiFidelityBayesianOptimisationSteps(
        objective,
        hi,
        { cost: [1, 10] },
        { ...base, acquisition: 'mf-gp-ucb', gamma: [gamma], zeta: [3, 0] },
      )
    const s = run(make(0), undefined, 1, { stream: stream(4) })
    const [a0, a1] = [row(s.acquisition, 0), row(s.acquisition, 1)]
    flat(s.combined).forEach((v, i) => expect(v).toBe(Math.min(a0[i], a1[i])))
    a0.forEach((v, i) => expect(v).toBeCloseTo(-(row(s.mean, 0)[i] - 2 * row(s.sd, 0)[i] - 3), 10))
    expect(s.nextIndex).toBe(flat(s.combined).indexOf(Math.max(...flat(s.combined))))
    // gamma = 0: the low fidelity is always uncertain enough; a huge gamma never is.
    expect(s.nextFidelity).toBe(0)
    expect(run(make(1e9), undefined, 1, { stream: stream(4) }).nextFidelity).toBe(1)
  })

  it('a short run on Forrester with refits: the incumbent at full fidelity improves on the initial design', () => {
    const alg = multiFidelityBayesianOptimisationSteps(
      objective,
      hi,
      { cost: [1, 5] },
      {
        acquisition: 'ei-per-cost',
        grid: 101,
      },
    )
    const tr = trace(alg, undefined, 8, { stream: stream(5), keep: 'all' })
    const first = tr.steps[0] as MultiFidelityBayesianOptimisationState
    expect(tr.final.best).toBeLessThanOrEqual(first.best)
    expect(tr.final.fidelity.slice(first.evaluations)).toContain(1)
    expect(tr.final.best).toBe(Math.min(...flat(tr.final.y).filter((_, i) => tr.final.fidelity[i] === 1)))
  })

  it('follows the trace protocol, and reports bad options', () => {
    for (const acquisition of ['mf-gp-ucb', 'mf-mes', 'ei-per-cost'] as const)
      expectProtocol(
        multiFidelityBayesianOptimisationSteps(objective, hi, { cost: [1, 10] }, { ...base, grid: 21, acquisition }),
        undefined,
        { n: 3 },
      )
    expect(() => multiFidelityBayesianOptimisationSteps(objective, hi, { cost: [1, 0] })).toThrow(/costs/)
    expect(() => multiFidelityBayesianOptimisationSteps(objective, hi, { cost: [1, 2] }, { init: [1] })).toThrow(/init/)
    expect(() =>
      multiFidelityBayesianOptimisationSteps(objective, hi, { cost: [1, 2] }, { acquisition: 'ei' as never }),
    ).toThrow(/unknown acquisition/)
    expect(bayesianAlgorithms.multiFidelityBayesianOptimisationSteps.info.problem).toBe('objective')
  })
})
