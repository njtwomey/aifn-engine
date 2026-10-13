import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  pbtToyProblem,
  populationAlgorithms,
  populationBasedTrainingSteps,
  populationFunctions,
  type PbtState,
} from 'aifn-methods/optimisation/population'
import { expectProtocol } from '../../protocol'
import { expectInfo } from '../../registry'

const flat = (x: Parameters<typeof toFlat>[0]) => Array.from(toFlat(x) as ArrayLike<number>)

describe('populationBasedTrainingSteps', () => {
  it('starts from the toy of Jaderberg et al. (2017), Figure 2', () => {
    const s = run(populationBasedTrainingSteps(), undefined, 0, { stream: stream(0) })
    expect(toArray(s.theta)).toEqual([
      [0.9, 0.9],
      [0.9, 0.9],
    ])
    expect(toArray(s.h)).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(flat(s.scores)[0]).toBeCloseTo(1.2 - 1.62, 14)
  })

  it('grid search stalls at Q = 0.39; PBT reaches the optimum 1.2', () => {
    for (let k = 0; k < 4; k++) {
      const grid = run(populationBasedTrainingSteps({ exploit: 'none', explore: 'none' }), undefined, 100, {
        stream: stream(k),
      })
      const pbt = run(populationBasedTrainingSteps(), undefined, 100, { stream: stream(k) })
      expect(grid.best).toBeCloseTo(0.39, 6)
      expect(pbt.best).toBeGreaterThan(1.199)
      expect(pbt.best).toBeGreaterThan(grid.best)
    }
  })

  it('exploit alone already gives most of the gain; inheriting the hyperparameters too undoes it', () => {
    const exploitOnly = run(populationBasedTrainingSteps({ explore: 'none' }), undefined, 40, { stream: stream(0) })
    const all = run(populationBasedTrainingSteps({ explore: 'none', inherit: 'all' }), undefined, 40, {
      stream: stream(0),
    })
    expect(exploitOnly.best).toBeGreaterThan(1.15)
    expect(all.best).toBeLessThan(0.4)
  })

  it('a gradient step on the surrogate, by hand', () => {
    const p = pbtToyProblem({ stepSize: 0.1 })
    // θ ← θ − 2αh⊙θ.
    expect(p.train([0.9, 0.5], [1, 0.5])).toEqual([0.9 - 0.18, 0.5 - 0.05])
    expect(p.surrogate?.([0.5, 0.5], [1, 0])).toBeCloseTo(0.95, 14)
  })

  it('ready steps carry the events: the weaker member copies the stronger and explores', () => {
    const tr = trace(populationBasedTrainingSteps({ readyInterval: 4 }), undefined, 8, {
      stream: stream(3),
      keep: 'all',
    })
    const steps = tr.steps as PbtState[]
    expect(steps.map((s) => s.ready)).toEqual([false, false, false, false, true, false, false, false, true])
    expect(steps.filter((s) => !s.ready).every((s) => s.events.length === 0)).toBe(true)
    const ev = steps[4].events
    expect(ev.map((e) => e.kind)).toEqual(['exploit', 'explore'])
    const [copy, explore] = ev
    expect(copy.source).not.toBe(copy.member)
    // After exploiting, the member's parameters are its source's.
    const theta = toArray(steps[4].theta) as number[][]
    expect(theta[copy.member]).toEqual(theta[copy.source as number])
    expect(explore.before).toEqual(copy.after)
    expect(explore.after).not.toEqual(explore.before)
  })

  it('truncation on a population of ten: the bottom two copy from the top two', () => {
    const tr = trace(populationBasedTrainingSteps({ population: 10, explore: 'resample' }), undefined, 4, {
      stream: stream(5),
      keep: 'all',
    })
    // The scores the ranking saw: one training step from the state before.
    const prev = tr.steps[3] as PbtState
    const toy = pbtToyProblem()
    const [th, hh] = [toArray(prev.theta) as number[][], toArray(prev.h) as number[][]]
    const before = th.map((x, i) => toy.score(toy.train(x, hh[i])))
    const after = tr.final as PbtState
    const order = before.map((_, i) => i).sort((a, b) => before[b] - before[a] || a - b)
    const exploits = after.events.filter((e) => e.kind === 'exploit')
    expect(exploits.length).toBe(2)
    for (const e of exploits) {
      expect(order.slice(8)).toContain(e.member)
      expect(order.slice(0, 2)).toContain(e.source)
    }
    // Resampled hyperparameters lie in the prior's [0, 1]².
    for (const e of after.events.filter((x) => x.kind === 'explore'))
      expect(e.after.every((v) => v >= 0 && v <= 1)).toBe(true)
  })

  it('rejects bad options', () => {
    expect(() => populationBasedTrainingSteps({ population: 1 })).toThrow(/at least 2/)
    expect(() => populationBasedTrainingSteps({ readyInterval: 0 })).toThrow(/readyInterval/)
    expect(() => populationBasedTrainingSteps({ fraction: 0.7 })).toThrow(/fraction/)
  })

  it('follows the trace protocol', () => {
    expectProtocol(populationBasedTrainingSteps({ population: 4 }), undefined, { n: 9 })
    expectProtocol(populationBasedTrainingSteps({ explore: 'resample', exploit: 'none' }), undefined, { n: 9 })
  })

  it('registers its entries', () => {
    expectInfo(populationAlgorithms, 'algorithm')
    expectInfo(populationFunctions, 'function')
  })
})
