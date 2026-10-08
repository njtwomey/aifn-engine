/**
 * Reward models: the best-of-n weights and curve against brute-force enumeration; the KL bound's values; the
 * Bradley–Terry fit recovering a linear gold reward; synthetic preferences' determinism and noise.
 */
import { describe, expect, it } from 'vitest'
import { toRows } from 'aifn-compute/foundation/tensor'
import {
  bestOfNCurve,
  bestOfNKl,
  bestOfNWeights,
  fitBradleyTerry,
  rewardModelFunctions,
  syntheticPreferences,
} from 'aifn-methods/neural/reward-models'
import { expectInfo } from '../../registry'

/** Every subset of size n of 0 … M − 1. */
function subsets(M: number, n: number): number[][] {
  if (n === 0) return [[]]
  const out: number[][] = []
  for (let first = 0; first <= M - n; first++)
    for (const rest of subsets(M - first - 1, n - 1)) out.push([first, ...rest.map((v) => v + first + 1)])
  return out
}

describe('bestOfNWeights', () => {
  it('sums to one, is zero below rank n, and is the probability of the maximum rank', () => {
    for (const [M, n] of [
      [7, 1],
      [7, 3],
      [7, 7],
      [3000, 40],
    ]) {
      const w = bestOfNWeights(M, n)
      expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      for (let i = 0; i < n - 1; i++) expect(w[i]).toBe(0)
      expect(w.every(Number.isFinite)).toBe(true)
    }
    const all = subsets(7, 3)
    const w = bestOfNWeights(7, 3)
    for (let i = 0; i < 7; i++)
      expect(w[i]).toBeCloseTo(all.filter((s) => Math.max(...s) === i).length / all.length, 12)
  })
})

describe('bestOfNCurve', () => {
  const proxy = Float64Array.from([0.3, -1, 2, 0.7, 1.1, -0.4, 0.05, 1.6])
  const gold = Float64Array.from([1, 0, -2, 0.5, 2, 0.2, 0.1, 1.5])

  it('gives the pool mean at n = 1 and the gold of the proxy maximum at n = M', () => {
    const [one, all] = bestOfNCurve({ proxy, gold, ns: [1, 8] })
    expect(one.proxy).toBeCloseTo(proxy.reduce((a, b) => a + b, 0) / 8, 12)
    expect(one.gold).toBeCloseTo(gold.reduce((a, b) => a + b, 0) / 8, 12)
    expect(all.gold).toBe(-2)
    expect(all.proxy).toBe(2)
    expect([one.kl, all.kl]).toEqual([0, bestOfNKl(8)])
  })

  it('matches brute-force enumeration of every subset', () => {
    for (const n of [2, 3, 5]) {
      const all = subsets(8, n)
      const best = all.map((s) => s.reduce((b, i) => (proxy[i] > proxy[b] ? i : b), s[0]))
      const [point] = bestOfNCurve({ proxy, gold, ns: [n] })
      expect(point.gold).toBeCloseTo(best.reduce((a, i) => a + gold[i], 0) / all.length, 12)
      expect(point.proxy).toBeCloseTo(best.reduce((a, i) => a + proxy[i], 0) / all.length, 12)
    }
  })

  it('shares the gold reward between tied proxies', () => {
    const [top] = bestOfNCurve({ proxy: Float64Array.from([1, 2, 2]), gold: Float64Array.from([0, 4, 6]), ns: [3] })
    expect(top.gold).toBe(5)
  })

  it('rejects n outside 1 … M', () => {
    expect(() => bestOfNCurve({ proxy, gold, ns: [9] })).toThrow(/from 1 to 8/)
  })
})

describe('bestOfNKl', () => {
  it('is 0 at n = 1 and increases', () => {
    expect(bestOfNKl(1)).toBe(0)
    expect(bestOfNKl(2)).toBeCloseTo(Math.log(2) - 0.5, 15)
    for (let n = 2; n < 300; n++) expect(bestOfNKl(n)).toBeGreaterThan(bestOfNKl(n - 1))
  })
})

describe('syntheticPreferences', () => {
  const gold = (x: Float64Array) => 2 * x[0] - x[1] + 0.5 * x[2]

  it('is deterministic from its seed', () => {
    const a = syntheticPreferences({ seed: 3, n: 50, dim: 3, gold, noise: 'gumbel', temperature: 1 })
    const b = syntheticPreferences({ seed: 3, n: 50, dim: 3, gold, noise: 'gumbel', temperature: 1 })
    const c = syntheticPreferences({ seed: 4, n: 50, dim: 3, gold, noise: 'gumbel', temperature: 1 })
    expect(toRows(a.chosen)).toEqual(toRows(b.chosen))
    expect(toRows(a.chosen)).not.toEqual(toRows(c.chosen))
  })

  it('always prefers the larger gold reward without noise, and usually with Gumbel noise', () => {
    const clean = syntheticPreferences({ seed: 1, n: 500, dim: 3, gold, noise: 'none', temperature: 1 })
    expect(clean.goldChosen.every((g, i) => g >= clean.goldRejected[i])).toBe(true)
    const noisy = syntheticPreferences({ seed: 1, n: 4000, dim: 3, gold, noise: 'gumbel', temperature: 1 })
    const agree = noisy.goldChosen.filter((g, i) => g > noisy.goldRejected[i]).length / 4000
    expect(agree).toBeGreaterThan(0.6)
    expect(agree).toBeLessThan(1)
    // The Bradley–Terry likelihood of the choices: the mean of σ(Δ) over pairs matches the agreement rate.
    const sigma = (z: number) => 1 / (1 + Math.exp(-z))
    let expected = 0
    noisy.goldChosen.forEach((g, i) => (expected += sigma(Math.abs(g - noisy.goldRejected[i]))))
    expect(agree).toBeCloseTo(expected / 4000, 1)
  })
})

describe('fitBradleyTerry', () => {
  it('recovers the direction of a linear gold reward', () => {
    const w0 = [2, -1, 0.5]
    const gold = (x: Float64Array) => w0.reduce((a, w, k) => a + w * x[k], 0)
    const prefs = syntheticPreferences({ seed: 5, n: 4000, dim: 3, gold, noise: 'gumbel', temperature: 1 })
    const { weights, score } = fitBradleyTerry(prefs, { l2: 1e-3 })
    const cos = weights.reduce((a, w, k) => a + w * w0[k], 0) / Math.hypot(...weights) / Math.hypot(...w0)
    expect(cos).toBeGreaterThan(0.995)
    // With Gumbel noise at temperature 1 the likelihood's scale is the gold reward's own.
    weights.forEach((w, k) => expect(w).toBeCloseTo(w0[k], 0))
    expect(score(Float64Array.from([1, 0, 0]))).toBeCloseTo(weights[0], 12)
  })

  it('uses the features it is given', () => {
    const gold = (x: Float64Array) => x[0] * x[0]
    const prefs = syntheticPreferences({ seed: 6, n: 2000, dim: 1, gold, noise: 'gumbel', temperature: 0.5 })
    const { weights } = fitBradleyTerry(prefs, { features: (x) => Float64Array.from([x[0], x[0] * x[0]]) })
    expect(Math.abs(weights[0])).toBeLessThan(0.3)
    expect(weights[1]).toBeGreaterThan(1)
  })

  it('is registered', () => {
    expectInfo(rewardModelFunctions, 'function')
  })
})
