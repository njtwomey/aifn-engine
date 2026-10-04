/**
 * Finite Markov chains against numpy (`fixtures/probability/markov.json`: eigenvectors, inv(I − Q), solves, matrix
 * powers), closed forms (gambler's ruin, Kac's formula) and laws (d(t) non-increasing, the relaxation-time bounds on
 * the mixing time, the simulator's frequencies).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  absorption,
  classifyStates,
  distanceToStationarity,
  distributionAfter,
  expectedHittingTimes,
  hittingProbabilities,
  isReversible,
  markovChainSteps,
  meanReturnTimes,
  mixingTime,
  nStepTransition,
  simulateChain,
  spectralGap,
  stationaryDistribution,
  stationaryDistributions,
  transitionMatrix,
} from 'aifn-compute/probability/markov'
import { fixture } from '../../fixtures'

type Ergodic = {
  P: number[][]
  stationary: number[]
  moduli: number[]
  target: number[]
  hitting: number[]
  worst: number[]
}
type Absorbing = {
  target: number
  win: number
  P: number[][]
  fundamental: number[][]
  probabilities: number[][]
  expectedSteps: number[]
  varianceSteps: number[]
}
const F = fixture<{ ergodic: Ergodic[]; absorbing: Absorbing[] }>('probability/markov')
const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t))
const close = (a: readonly number[], b: readonly number[], digits = 10) =>
  a.forEach((v, i) => (Number.isFinite(b[i]) ? expect(v).toBeCloseTo(b[i], digits) : expect(v).toBe(b[i])))

/** A lazy random walk on a path of n states: reversible, with a closed-form uniform-like stationary law. */
const lazyPath = (n: number): number[][] =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (j === i) return 0.5 + (i === 0 || i === n - 1 ? 0.25 : 0)
      return Math.abs(i - j) === 1 ? 0.25 : 0
    }),
  )

describe('validation', () => {
  it('rejects non-stochastic matrices', () => {
    expect(() =>
      transitionMatrix([
        [0.5, 0.4],
        [0, 1],
      ]),
    ).toThrow(/row 0/)
    expect(() =>
      transitionMatrix([
        [1.2, -0.2],
        [0, 1],
      ]),
    ).toThrow(/not a probability/)
    expect(() => transitionMatrix([[1, 0, 0]])).toThrow(/square/)
  })
})

describe('stationary distribution', () => {
  it.each(F.ergodic.map((c, i) => [i, c] as const))('matches numpy’s left eigenvector (case %i)', (_, c) => {
    close(flat(stationaryDistribution(c.P)), c.stationary, 10)
  })

  it('is fixed by P and has mean return times 1/π (Kac)', () => {
    const P = F.ergodic[1].P
    const pi = flat(stationaryDistribution(P))
    close(flat(distributionAfter(P, pi, 1)), pi, 12)
    close(
      flat(meanReturnTimes(P)),
      pi.map((v) => 1 / v),
      8,
    )
  })

  it('agrees with a long run of Pⁿ', () => {
    const P = F.ergodic[2].P
    const rows = toRows(nStepTransition(P, 400)) as number[][]
    for (const r of rows) close(r, F.ergodic[2].stationary, 8)
  })

  it('throws for two closed classes and lists one per class', () => {
    const P = [
      [1, 0, 0],
      [0.3, 0.4, 0.3],
      [0, 0, 1],
    ]
    expect(() => stationaryDistribution(P)).toThrow(/2 closed classes/)
    expect(toRows(stationaryDistributions(P))).toEqual([
      [1, 0, 0],
      [0, 0, 1],
    ])
  })
})

describe('classifyStates', () => {
  it('finds classes, closed classes, transient states and periods', () => {
    // 0 ↔ 1 (period 2, closed), 2 → {0, 3}, 3 → 3 (absorbing)
    const P = [
      [0, 1, 0, 0],
      [1, 0, 0, 0],
      [0.5, 0, 0, 0.5],
      [0, 0, 0, 1],
    ]
    const c = classifyStates(P)
    expect(c.classes).toEqual([[0, 1], [3], [2]])
    expect(c.closed).toEqual([true, true, false])
    expect(c.periods.slice(0, 2)).toEqual([2, 1])
    expect(c.transient).toEqual([2])
    expect(c.absorbing).toEqual([3])
    expect(c.irreducible).toBe(false)
    expect(c.aperiodic).toBe(false)
  })

  it('reports an irreducible, aperiodic chain', () => {
    const c = classifyStates(F.ergodic[0].P)
    expect(c.irreducible && c.aperiodic).toBe(true)
  })
})

describe('absorption (gambler’s ruin)', () => {
  it.each(F.absorbing.map((c) => [c.target, c.win, c] as const))('N = %i, p = %f matches numpy', (_, __, c) => {
    const a = absorption(c.P)
    expect(a.absorbing).toEqual([0, c.target])
    ;(toRows(a.fundamental) as number[][]).forEach((r, i) => close(r, c.fundamental[i], 9))
    ;(toRows(a.probabilities) as number[][]).forEach((r, i) => close(r, c.probabilities[i], 10))
    close(flat(a.expectedSteps), c.expectedSteps, 9)
    close(flat(a.varianceSteps), c.varianceSteps, 6)
  })

  it('has the closed forms i/N and i(N − i) in a fair game, and (1 − (q/p)^i)/(1 − (q/p)^N) otherwise', () => {
    const fair = F.absorbing[0]
    const a = absorption(fair.P)
    const N = fair.target
    a.transient.forEach((i, r) => {
      expect((toRows(a.probabilities) as number[][])[r][1]).toBeCloseTo(i / N, 12)
      expect(flat(a.expectedSteps)[r]).toBeCloseTo(i * (N - i), 10)
    })
    const biased = F.absorbing[1]
    const ratio = (1 - biased.win) / biased.win
    const h = flat(hittingProbabilities(biased.P, [biased.target]))
    h.forEach((v, i) => expect(v).toBeCloseTo((1 - ratio ** i) / (1 - ratio ** biased.target), 12))
  })
})

describe('hitting', () => {
  it.each(F.ergodic.map((c, i) => [i, c] as const))('expected hitting times match numpy (case %i)', (_, c) => {
    close(flat(expectedHittingTimes(c.P, c.target)), c.hitting, 8)
  })

  it('gives ∞ where the target may never be reached and 0 probability where it cannot', () => {
    const P = [
      [1, 0, 0],
      [0.5, 0, 0.5],
      [0, 0, 1],
    ]
    expect(flat(expectedHittingTimes(P, [2]))).toEqual([Infinity, Infinity, 0])
    expect(flat(hittingProbabilities(P, [2]))).toEqual([0, 0.5, 1])
  })

  it('a state whose only way to a lost state passes through the target hits it surely', () => {
    // 0 → 1 surely; 1 (the target) → 2, which is absorbing and cannot return. The chain stops on entering 1, so
    // k₀ = 1, though 0 can reach the lost state 2 through 1.
    const P = [
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1],
    ]
    expect(flat(expectedHittingTimes(P, [1]))).toEqual([1, 0, Infinity])
    expect(flat(hittingProbabilities(P, [1]))).toEqual([1, 1, 0])
  })
})

describe('structure, not tolerance', () => {
  it('a class that leaks a tiny probability is transient, and a leaking state is not absorbing', () => {
    const e = 1e-12
    const P = [
      [1 - e, e],
      [0, 1],
    ]
    const c = classifyStates(P)
    expect(c.transient).toEqual([0])
    expect(c.absorbing).toEqual([1])
    expect(flat(hittingProbabilities(P, [1]))).toEqual([1, 1])
  })
})

describe('convergence', () => {
  it.each(F.ergodic.map((c, i) => [i, c] as const))('d(t) matches numpy and never increases (case %i)', (_, c) => {
    const d = flat(distanceToStationarity(c.P, 12).worst)
    close(d, c.worst, 10)
    for (let t = 1; t < d.length; t++) expect(d[t]).toBeLessThanOrEqual(d[t - 1] + 1e-12)
  })

  it.each(F.ergodic.map((c, i) => [i, c] as const))('eigenvalue moduli match numpy (case %i)', (_, c) => {
    const g = spectralGap(c.P)
    const re = flat(g.real)
    const im = flat(g.imag)
    close(
      re.map((v, i) => Math.hypot(v, im[i])),
      c.moduli,
      8,
    )
    expect(g.secondModulus).toBeCloseTo(c.moduli[1], 8)
  })

  it('a reversible chain mixes within its relaxation-time bounds', () => {
    for (const n of [4, 8, 12]) {
      const P = lazyPath(n)
      expect(isReversible(P).reversible).toBe(true)
      const m = mixingTime(P)
      expect(m.lower).toBeDefined()
      expect(m.time).toBeGreaterThanOrEqual(m.lower!)
      expect(m.time).toBeLessThanOrEqual(m.upper!)
      const d = flat(m.distances)
      expect(d.at(-1)!).toBeLessThanOrEqual(0.25)
      expect(d.at(-2)!).toBeGreaterThan(0.25)
    }
  })

  it('a periodic chain never mixes and a non-reversible one is detected', () => {
    const flip = [
      [0, 1],
      [1, 0],
    ]
    expect(mixingTime(flip, { maxSteps: 50 }).time).toBe(Infinity)
    expect(spectralGap(flip).absoluteGap).toBeCloseTo(0, 12)
    const cycle = [
      [0.1, 0.9, 0],
      [0, 0.1, 0.9],
      [0.9, 0, 0.1],
    ]
    expect(isReversible(cycle).reversible).toBe(false)
  })
})

describe('simulation', () => {
  it('visit frequencies approach π (ergodic theorem) and the exact distribution is p₀Pᵗ', () => {
    const c = F.ergodic[1]
    const tr = trace(markovChainSteps(c.P, { start: 0 }), undefined, 30, { keep: 'all', stream: stream(3) })
    tr.steps.forEach((s) => close(flat(s.distribution), flat(distributionAfter(c.P, 0, s.t)), 12))
    const last = run(markovChainSteps(c.P), undefined, 20_000, { stream: stream(4) })
    const freq = flat(last.visits).map((v) => v / (last.t + 1))
    freq.forEach((f, i) => expect(Math.abs(f - c.stationary[i])).toBeLessThan(0.02))
    expect(last.occupationDistance).toBeLessThan(0.03)
  })

  it('simulateChain is reproducible and follows the transitions', () => {
    const P = lazyPath(5)
    const a = flat(simulateChain(stream(9), P, 2, 200))
    expect(flat(simulateChain(stream(9), P, 2, 200))).toEqual(a)
    expect(a[0]).toBe(2)
    for (let t = 1; t < a.length; t++) expect(P[a[t - 1]][a[t]]).toBeGreaterThan(0)
  })
})
