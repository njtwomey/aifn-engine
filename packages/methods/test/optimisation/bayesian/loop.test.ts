/**
 * `bayesianOptimisationSteps` by behaviour: it finds the grid minimum of Forrester and Branin within a budget, a run is
 * fixed by its stream, every state is consistent with the evaluations (the incumbent never worsens, the surrogate is
 * the GP posterior at the reported hyperparameters, which the refit chose by marginal likelihood), and each
 * acquisition's choice is the argmax of the scores it reports.
 */
import { describe, expect, it } from 'vitest'
import { branin, forrester, svmTuningSurface } from 'aifn-methods/data/objectives'
import { gpPosterior, logMarginalLikelihood } from 'aifn-methods/learning/gaussian-processes'
import {
  bayesianAlgorithms,
  bayesianOptimisationSteps,
  ucbBeta,
  upperConfidenceBound,
  type BayesianOptimisationState,
} from 'aifn-methods/optimisation/bayesian'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, seek, trace } from 'aifn-compute/foundation/trace'
import { matern52, rbf } from 'aifn-compute/learning/kernels'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { expectInfo } from '../../registry'

const F = forrester()
const B = branin()

/** The smallest value of a function over a grid's rows. */
const gridMin = (f: (x: Tensor) => number, grid: Tensor): number => {
  const [m, d] = grid.shape
  const g = toFlat(grid)
  let best = Infinity
  for (let i = 0; i < m; i++) best = Math.min(best, f(fromData(Float64Array.from(g.slice(i * d, i * d + d)), [d])))
  return best
}

/** The index of the largest entry (the first of ties). */
const argmax = (t: Tensor): number => {
  const a = toFlat(t)
  return a.reduce((k, v, i) => (v > a[k] ? i : k), 0)
}

/** The surrogate rebuilt from a state: the GP posterior on the unit cube with the standardised targets. */
const rebuild = (s: BayesianOptimisationState, lo: number[], hi: number[], kernel = matern52()) => {
  const d = lo.length
  const u = toFlat(s.x).map((v, i) => (v - lo[i % d]) / (hi[i % d] - lo[i % d]))
  const X = fromData(Float64Array.from(u), [s.evaluations, d])
  const Z = fromData(
    Float64Array.from(toFlat(s.y), (v) => (v - s.shift) / s.scale),
    [s.evaluations],
  )
  const k = kernel.withParams(s.hyper as never)
  return { X, Z, k, post: gpPosterior(k, X, Z, { noiseVariance: s.noiseVariance }) }
}

describe('bayesianOptimisationSteps on Forrester', () => {
  const alg = bayesianOptimisationSteps(F.value, F)
  const tr = trace(alg, undefined, 12, { stream: stream(0), keep: 'all' })
  const states = tr.steps

  it('finds the grid minimum within 15 evaluations, next to the true minimiser', () => {
    expect(alg.grid.shape).toEqual([201, 1])
    expect(tr.final.evaluations).toBe(15)
    expect(tr.final.best).toBeCloseTo(gridMin(F.value, alg.grid), 12)
    expect(Math.abs(toFlat(tr.final.bestX)[0] - toFlat(F.minima[0])[0])).toBeLessThan(0.005)
    expect(tr.final.best - F.minimumValue).toBeLessThan(0.01)
  })

  it('keeps every state consistent with the evaluations', () => {
    for (let t = 0; t < states.length; t++) {
      const s = states[t]
      expect(s.t).toBe(t)
      expect(s.evaluations).toBe(3 + t)
      expect(s.x.shape).toEqual([s.evaluations, 1])
      toFlat(s.x).forEach((x, i) => expect(toFlat(s.y)[i]).toBe(F.value(tensor([x]))))
      const y = toFlat(s.y)
      expect(s.best).toBe(Math.min(...y))
      expect(y[s.bestIndex]).toBe(s.best)
      expect(toFlat(s.bestX)).toEqual(toFlat(s.x).slice(s.bestIndex, s.bestIndex + 1))
      expect(s.nextIndex).toBe(argmax(s.acquisition))
      expect(toFlat(s.next)).toEqual(toFlat(alg.grid).slice(s.nextIndex, s.nextIndex + 1))
      if (t > 0) {
        expect(s.best).toBeLessThanOrEqual(states[t - 1].best)
        expect(toFlat(s.x).at(-1)).toBe(toFlat(states[t - 1].next)[0])
      }
    }
  })

  it('reports the GP posterior at the refitted hyperparameters, which maximise the marginal likelihood', () => {
    for (const s of states.filter((_, t) => t % 4 === 0)) {
      const { X, Z, k, post } = rebuild(s, [0], [1])
      const p = post.predict(fromData(Float64Array.from(toFlat(alg.grid)), [201, 1]))
      toFlat(p.mean).forEach((v, i) => expect(s.shift + s.scale * v).toBeCloseTo(toFlat(s.mean)[i], 10))
      toFlat(p.variance).forEach((v, i) => expect(s.scale * Math.sqrt(v)).toBeCloseTo(toFlat(s.sd)[i], 10))
      expect(s.logMarginal).toBeCloseTo(
        logMarginalLikelihood(k, X, Z, { noiseVariance: s.noiseVariance }).value as number,
        10,
      )
      const atStart = logMarginalLikelihood(matern52({ lengthscale: 0.2 }), X, Z, { noiseVariance: s.noiseVariance })
      expect(s.logMarginal).toBeGreaterThanOrEqual((atStart.value as number) - 1e-9)
      // Nearly noise-free: the posterior interpolates the evaluations.
      const atData = post.predict(X)
      toFlat(atData.mean).forEach((v, i) => expect(s.shift + s.scale * v).toBeCloseTo(toFlat(s.y)[i], 2))
      toFlat(atData.variance).forEach((v) => expect(s.scale * Math.sqrt(v)).toBeLessThan(0.02))
    }
  })

  it('is fixed by its stream, and replays step by step', () => {
    const again = trace(alg, undefined, 12, { stream: stream(0) }).final
    expect(toFlat(again.x)).toEqual(toFlat(tr.final.x))
    expect(toFlat(again.acquisition)).toEqual(toFlat(tr.final.acquisition))
    expect(toFlat(run(alg, undefined, 5, { stream: stream(0) }).x)).toEqual(toFlat(states[5].x))
    expect(toFlat(seek(alg, undefined, 7, { checkpoints: tr, stream: stream(0) }).mean)).toEqual(toFlat(states[7].mean))
    const other = run(alg, undefined, 0, { stream: stream(1) })
    expect(toFlat(other.x)).not.toEqual(toFlat(states[0].x))
    expect(() => structuredClone(tr.final)).not.toThrow()
  })

  it('keeps the starting kernel without refit', () => {
    const kernel = rbf({ lengthscale: 0.1, variance: 1 })
    const s = run(bayesianOptimisationSteps(F.value, F, { kernel, refit: false, grid: 51 }), undefined, 3, {
      stream: stream(2),
    })
    expect(s.hyper).toEqual(kernel.params)
    expect(s.noiseVariance).toBe(1e-6)
  })
})

describe('bayesianOptimisationSteps on Branin', () => {
  it('finds a global minimum to within a grid cell in 30 evaluations', () => {
    const alg = bayesianOptimisationSteps(B.value, B)
    const s = run(alg, undefined, 25, { stream: stream(0) })
    expect(alg.grid.shape).toEqual([41 * 41, 2])
    expect(alg.shape).toEqual([41, 41])
    expect(s.evaluations).toBe(30)
    // The grid's spacing is 15 / 40 = 0.375; its best point is 0.419 against the minimum 0.398.
    expect(gridMin(B.value, alg.grid)).toBeCloseTo(0.4188, 4)
    expect(s.best - B.minimumValue).toBeLessThan(0.07)
    const [x1, x2] = toFlat(s.bestX)
    const nearest = Math.min(...B.minima.map((m) => Math.hypot(x1 - toFlat(m)[0], x2 - toFlat(m)[1])))
    expect(nearest).toBeLessThan(0.375 * Math.SQRT2)
    // The grid's last coordinate varies fastest.
    expect(toFlat(alg.grid).slice(0, 4)).toEqual([-5, 0, -5, 0.375])
  })

  it('takes explicit initial points and a given grid', () => {
    const init = tensor([
      [0, 5],
      [5, 10],
    ])
    const grid = tensor([
      [-3, 12],
      [3, 2],
      [9, 2.5],
      [0, 0],
    ])
    const s = run(bayesianOptimisationSteps(B.value, B, { init, grid, refit: false }), undefined, 1, {
      stream: stream(0),
    })
    expect(toFlat(s.x).slice(0, 4)).toEqual([0, 5, 5, 10])
    expect(s.acquisition.shape).toEqual([4])
  })
})

describe('the acquisition options', () => {
  it('thompson: the next point is the minimum of one joint posterior draw on the grid', () => {
    const alg = bayesianOptimisationSteps(F.value, F, { acquisition: 'thompson', grid: 101 })
    const tr = trace(alg, undefined, 5, { stream: stream(4), keep: 'all' })
    for (const s of tr.steps) {
      const draw = toFlat(s.draw!)
      expect(draw).toHaveLength(101)
      expect(draw[s.nextIndex]).toBe(Math.min(...draw))
      // The draw interpolates the (nearly noise-free) evaluations at grid points.
      expect(s.maxValues).toBeNull()
    }
    expect(toFlat(tr.steps[1].draw!)).not.toEqual(toFlat(tr.steps[2].draw!))
  })

  it('mes: samples of the minimum, at or below the incumbent, and the choice maximises the score', () => {
    const s = run(bayesianOptimisationSteps(F.value, F, { acquisition: 'mes', grid: 101 }), undefined, 4, {
      stream: stream(5),
    })
    expect(s.maxValues!.shape).toEqual([10])
    for (const v of toFlat(s.maxValues!)) expect(v).toBeLessThanOrEqual(s.best)
    expect(s.nextIndex).toBe(argmax(s.acquisition))
  })

  it("ucb with 'srinivas' uses the GP-UCB schedule over the grid", () => {
    const s = run(
      bayesianOptimisationSteps(F.value, F, { acquisition: 'ucb', beta: 'srinivas', grid: 101 }),
      undefined,
      2,
      {
        stream: stream(6),
      },
    )
    const expected = upperConfidenceBound(s.mean, s.sd, { beta: ucbBeta(3, { candidates: 101 }) })
    toFlat(expected).forEach((v, i) => expect(toFlat(s.acquisition)[i]).toBeCloseTo(v, 12))
  })

  it('pi, and maximising instead of minimising', () => {
    const neg = (x: Tensor) => -F.value(x)
    const s = run(bayesianOptimisationSteps(neg, F, { acquisition: 'pi', minimise: false, grid: 101 }), undefined, 3, {
      stream: stream(7),
    })
    expect(s.best).toBe(Math.max(...toFlat(s.y)))
    for (const p of toFlat(s.acquisition)) expect(p >= 0 && p <= 1).toBe(true)
  })
})

describe('a noisy objective', () => {
  it('draws the noise from the run, fits the noise variance, and replays exactly', () => {
    const f = svmTuningSurface({ noise: 0.01 })
    const alg = bayesianOptimisationSteps(f.value, f, { noise: 'fit', grid: 21, init: 6 })
    const a = run(alg, undefined, 6, { stream: stream(8) })
    const b = run(alg, undefined, 6, { stream: stream(8) })
    expect(toFlat(a.y)).toEqual(toFlat(b.y))
    expect(a.noiseVariance).toBeGreaterThan(0)
    expect(a.noiseVariance).not.toBe(0.1)
    // The values are the surface plus noise of the declared size.
    const residuals = toFlat(a.y).map((y, i) => y - f.mean(tensor(toFlat(a.x).slice(2 * i, 2 * i + 2))))
    expect(residuals.some((r) => r !== 0)).toBe(true)
    for (const r of residuals) expect(Math.abs(r)).toBeLessThan(0.05)
  })
})

describe('errors and registration', () => {
  it('reports bad boxes, grids and options', () => {
    expect(() => bayesianOptimisationSteps(F.value, { lo: [1], hi: [0] })).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(F.value, { lo: [0, 0], hi: [1] })).toThrow(ShapeError)
    expect(() => bayesianOptimisationSteps(F.value, F, { noise: 'fit', refit: false })).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(F.value, F, { grid: 1 })).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(F.value, F, { grid: tensor([[0, 1]]) })).toThrow(ShapeError)
  })

  it('flags a state whose evaluation is not finite', () => {
    const s = run(
      bayesianOptimisationSteps(() => NaN, F, { grid: 11 }),
      undefined,
      3,
      { stream: stream(0) },
    )
    expect(s.diverged).toBe(true)
    expect(s.t).toBe(0)
  })

  it('registers the loop with state fields for its roles', () => {
    expectInfo(bayesianAlgorithms, 'algorithm')
    const { info } = bayesianAlgorithms.bayesianOptimisationSteps
    const s = run(bayesianOptimisationSteps(F.value, F, { grid: 21 }), undefined, 1, { stream: stream(0) })
    expect(s).toHaveProperty(info.state.iterate!)
    expect(s).toHaveProperty(info.state.objective!)
    expect(info.random).toBe(true)
  })
})
