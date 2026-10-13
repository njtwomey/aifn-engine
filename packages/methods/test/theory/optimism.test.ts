/**
 * The optimism study against the laws it demonstrates: for least squares on a fixed design the optimism is
 * $2p\sigma^2/n$ and $\mathrm{df} = p$, for ridge and nearest neighbours $\mathrm{df} = \trace\Smat$, and selecting the
 * best of $M$ noise features has $\mathrm{df} = \expect\max_j \chi^2_{1,j}$; each within Monte Carlo error. Also the
 * fits against the shared least-squares layer, determinism, and a run through a worker task tree.
 */
import { describe, expect, it } from 'vitest'
import { child, standardNormals, stream, units } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import { applyWeights, leastSquaresWeights, targetFunction } from 'aifn-methods/theory'
import { optimismFunctions, optimismStudy, optimismSweep } from 'aifn-methods/theory/optimism'
import { evaluateTask } from './simulation-task'

/** Within `z` Monte Carlo standard errors. */
const near = (value: number, truth: number, se: number, z = 4) => expect(Math.abs(value - truth)).toBeLessThan(z * se)

describe('optimismStudy', () => {
  it('gives optimism 2 p sigma^2 / n and df p for least squares on a fixed design', () => {
    const n = 40
    const r = optimismStudy(stream(0), { model: { kind: 'polynomial', degree: 3 }, n, noise: 1, repeats: 1500 })
    expect(r.referenceDf).toBeCloseTo(4, 10)
    near(r.df, 4, r.dfSe)
    near(r.optimism, (2 * 4) / n, r.optimismSe)
    near(r.covariancePenalty, (2 * 4) / n, (2 * r.dfSe) / n)
    expect(r.inSampleError - r.trainError).toBeCloseTo(r.optimism, 12)
    // The expected training error is sigma^2 (n - p) / n plus the squared bias of a cubic fit to the sine.
    expect(r.trainError).toBeGreaterThan((n - 4) / n - 0.05)
  })

  it('gives df p for least squares on a random design too', () => {
    const r = optimismStudy(stream(1), {
      model: { kind: 'polynomial', degree: 2 },
      n: 30,
      noise: 0.5,
      design: 'random',
      repeats: 1500,
    })
    expect(r.referenceDf).toBeCloseTo(3, 10)
    near(r.df, 3, r.dfSe)
    near(r.optimism, (2 * 3 * 0.25) / 30, r.optimismSe)
  })

  it('gives df trace S for ridge and n / k for nearest neighbours', () => {
    const ridge = optimismStudy(stream(2), { model: { kind: 'polynomial', degree: 9, ridge: 1 }, n: 40, repeats: 1500 })
    expect(ridge.referenceDf).toBeGreaterThan(1)
    expect(ridge.referenceDf).toBeLessThan(10)
    near(ridge.df, ridge.referenceDf, ridge.dfSe)
    const knn = optimismStudy(stream(3), { model: { kind: 'knn', k: 5 }, n: 40, repeats: 1500, design: 'random' })
    expect(knn.referenceDf).toBeCloseTo(8, 10)
    near(knn.df, 8, knn.dfSe)
  })

  it('gives df E max chi^2 for the best of M noise features, far above one coefficient', () => {
    const expectedMax = (candidates: number) =>
      optimismStudy(stream(0), {
        target: 'zero',
        n: 10,
        model: { kind: 'best-of-noise-features', candidates },
        repeats: 2,
      }).referenceDf
    expect(expectedMax(1)).toBeCloseTo(1, 8)
    // E max(Z1^2, Z2^2) = 1 + 2 / pi.
    expect(expectedMax(2)).toBeCloseTo(1 + 2 / Math.PI, 8)
    const model = { kind: 'best-of-noise-features', candidates: 50 } as const
    const r = optimismStudy(stream(4), { target: 'zero', noise: 1, n: 50, model, repeats: 1500 })
    expect(r.referenceDf).toBeGreaterThan(expectedMax(10))
    near(r.df, r.referenceDf, r.dfSe)
    expect(r.df).toBeGreaterThan(4)
    // With f = 0 a fresh input changes nothing: the extra-sample error is the in-sample error.
    expect(r.extraSampleError).toBeCloseTo(r.inSampleError, 12)
    expect(optimismStudy(stream(4), { target: 'sine', n: 50, model, repeats: 2 }).referenceDf).toBeNaN()
  })

  it('fits as the shared least-squares layer does, and as nearest neighbours by hand', () => {
    const s = stream(5)
    const n = 12
    const sigma = 0.3
    const f = targetFunction('sine')
    const x = Float64Array.from(units(child(child(s, 'design'), 'x'), n), (u) => 2 * u - 1)
    // The mean training error of repeats 0 and 1, refitted here by `fit`.
    const byHand = (fit: (y: Float64Array) => ArrayLike<number>) => {
      let total = 0
      for (const r of [0, 1]) {
        const e = standardNormals(child(s, 'noise', r), n)
        const y = Float64Array.from(x, (v, i) => f(v) + sigma * e[i])
        const yHat = fit(y)
        total += y.reduce((a, v, i) => a + (v - yHat[i]) ** 2, 0) / n / 2
      }
      return total
    }
    for (const [degree, ridge] of [
      [3, 0],
      [5, 0.5],
      [14, 0],
    ]) {
      const p = degree + 1
      const phi = Float64Array.from({ length: n * p }, (_, k) => x[Math.floor(k / p)] ** (k % p))
      const expected = byHand((y) => applyWeights(phi, n, p, leastSquaresWeights(phi, n, p, y, ridge)))
      const r = optimismStudy(s, { model: { kind: 'polynomial', degree, ridge }, n, noise: sigma, repeats: 2 })
      expect(r.trainError).toBeCloseTo(expected, 9)
    }
    const k = 3
    const knn = (y: Float64Array) =>
      Float64Array.from(x, (q) => {
        const order = Array.from(x.keys()).sort((a, b) => Math.abs(x[a] - q) - Math.abs(x[b] - q) || a - b)
        return order.slice(0, k).reduce((a, j) => a + y[j], 0) / k
      })
    const r = optimismStudy(s, { model: { kind: 'knn', k }, n, noise: sigma, repeats: 2 })
    expect(r.trainError).toBeCloseTo(byHand(knn), 12)
    expect(r.referenceDf).toBeCloseTo(n / k, 12)
  })

  it('is fixed by its stream, and rejects bad options', () => {
    const o = { model: { kind: 'knn', k: 3 }, repeats: 20 } as const
    expect(optimismStudy(stream(6), o)).toEqual(optimismStudy(stream(6), o))
    expect(optimismStudy(stream(7), o).trainError).not.toBe(optimismStudy(stream(6), o).trainError)
    const tooMany = { kind: 'best-of-noise-features', candidates: 11 } as const
    expect(() => optimismStudy(stream(0), { model: tooMany, n: 10 })).toThrow(DomainError)
    expect(() => optimismStudy(stream(0), { ...o, repeats: 1 })).toThrow(DomainError)
  })
})

describe('optimismSweep', () => {
  it('runs one study per value on the same samples', () => {
    const s = stream(8)
    const sweep = optimismSweep(s, { family: 'ridge', values: [0.1, 10], degree: 6, n: 30, repeats: 50 })
    const one = optimismStudy(child(s, 'sweep'), {
      model: { kind: 'polynomial', degree: 6, ridge: 10 },
      n: 30,
      repeats: 50,
    })
    expect(sweep.values).toEqual([0.1, 10])
    expect(sweep.df[1]).toBe(one.df)
    expect(sweep.referenceDf[1]).toBe(one.referenceDf)
    // More penalty, fewer degrees of freedom.
    expect(sweep.referenceDf[1]).toBeLessThan(sweep.referenceDf[0])
    const best = optimismSweep(s, {
      family: 'best-of-noise-features',
      values: [1, 10],
      target: 'zero',
      n: 20,
      repeats: 20,
    })
    expect(best.referenceDf[0]).toBeCloseTo(1, 8)
    const degrees = optimismSweep(s, { family: 'polynomial', values: [1, 4], n: 20, repeats: 20 })
    expect(Array.from(degrees.referenceDf).map((v) => Math.round(v))).toEqual([2, 5])
    const k = optimismSweep(s, { family: 'knn', values: [1, 4], n: 20, repeats: 20 })
    expect(Array.from(k.referenceDf).map((v) => Math.round(v))).toEqual([20, 5])
  })

  it('runs as a worker task, with a plain, cloneable result', async () => {
    const task = {
      $call: 'theory/optimism/optimismSweep',
      args: [
        { $call: 'foundation/random/stream', args: [7] },
        { family: 'polynomial', values: [1, 3], repeats: 20 },
      ],
    }
    const result = await evaluateTask(task)
    expect(structuredClone(result)).toEqual(result)
    expect(result).toEqual(optimismSweep(stream(7), { family: 'polynomial', values: [1, 3], repeats: 20 }))
    const study = await evaluateTask({
      $call: 'theory/optimism/optimismStudy',
      args: [
        { $call: 'foundation/random/stream', args: [7] },
        { model: { kind: 'knn', k: 2 }, repeats: 5 },
      ],
    })
    expect(structuredClone(study)).toEqual(study)
  })

  it('registers both studies as simulations', () => {
    expect(Object.keys(optimismFunctions).sort()).toEqual(['optimismStudy', 'optimismSweep'])
    for (const e of Object.values(optimismFunctions)) expect(e.info.role).toBe('simulation')
  })
})
