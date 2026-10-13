/**
 * Adaptive holdouts: the Ladder and its parameter-free variant against literal transcriptions of Blum and Hardt's
 * Figures 1 and 2 and their laws, Thresholdout against Dwork et al.'s Figure 1 on the same noise draws, the boosting
 * attack's bias, the expected Gaussian maximum against scipy (fixture `theory/adaptive-holdout`), and the worker path.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  adaptiveHoldoutFunctions,
  boostingAttack,
  expectedGaussianMaximum,
  ladder,
  ladderParameterFree,
  thresholdout,
} from 'aifn-methods/theory/adaptive-holdout'
import { fixture } from '../fixtures'

const fx = fixture<{ expected_maximum: { J: number; value: number }[] }>('theory/adaptive-holdout')

/** A k x n matrix of 0/1 losses, each row with its own error rate. */
function lossMatrix(s: Stream, k: number, n: number): number[][] {
  return Array.from({ length: k }, (_, t) => {
    const p = 0.3 + 0.2 * uniform(child(s, 'rate', t))
    const r = child(s, 'row', t)
    return Array.from({ length: n }, () => (uniform(r) < p ? 1 : 0))
  })
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
const roundTo = (x: number, step: number) => Math.round(x / step) * step

describe('ladder', () => {
  it('follows Figure 1 of Blum and Hardt (2015), within eta of the best holdout loss and with few updates', () => {
    const losses = lossMatrix(stream('ladder'), 200, 97)
    for (const eta of [0.0123, 0.0471, 0.1013]) {
      const { reported, updated } = ladder(losses, { eta })
      let R = Infinity
      let best = Infinity
      losses.forEach((row, t) => {
        const r = mean(row)
        const moves = r < R - eta
        if (moves) R = roundTo(r, eta)
        best = Math.min(best, r)
        expect(updated[t]).toBe(moves)
        expect(reported[t]).toBeCloseTo(R, 12)
        expect(Math.abs(reported[t] - best)).toBeLessThanOrEqual(eta + 1e-12)
      })
      expect(updated.filter(Boolean).length).toBeLessThanOrEqual(1 / eta + 1)
      // Reported values only fall, by at least eta at each update after the first.
      for (let t = 1; t < reported.length; t++)
        if (updated[t]) expect(reported[t - 1] - reported[t]).toBeGreaterThanOrEqual(eta - 1e-12)
    }
  })

  it('reports to the multiple of eta and accepts a tensor; rejects bad input', () => {
    const r = ladder(
      [
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
      ],
      { eta: 0.05 },
    )
    expect(Array.from(r.reported)).toEqual([0.4, 0.3])
    expect(() => ladder([[0.5, 1.5]], { eta: 0.1 })).toThrow(DomainError)
    expect(() => ladder([[0.5, 0.5]], { eta: 0 })).toThrow(DomainError)
  })
})

describe('ladderParameterFree', () => {
  it('follows Figure 2 of Blum and Hardt (2015) literally', () => {
    const losses = lossMatrix(stream('parameter-free'), 150, 80)
    const n = 80
    const { reported, updated } = ladderParameterFree(losses)
    let R = Infinity
    let previous = new Array<number>(n).fill(0)
    losses.forEach((row, t) => {
      const d = row.map((v, i) => v - previous[i])
      const md = mean(d)
      const s = Math.sqrt(d.reduce((a, v) => a + (v - md) ** 2, 0) / (n - 1))
      const moves = mean(row) < R - s / Math.sqrt(n)
      if (moves) {
        R = roundTo(mean(row), 1 / n)
        previous = row
      }
      expect(updated[t]).toBe(moves)
      expect(reported[t]).toBeCloseTo(R, 12)
    })
    expect(updated.filter(Boolean).length).toBeGreaterThan(1)
  })
})

describe('thresholdout', () => {
  it('answers from the training set until the gap crosses the threshold; the budget stops it (no noise)', () => {
    const train = [
      [1, 1, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ]
    const holdout = [
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]
    const r = thresholdout(stream(0), train, holdout, { threshold: 0.3, sigma: 0, budget: 2 })
    expect(Array.from(r.answers)).toEqual([0.5, 0.5, 0, 0, NaN])
    expect(r.revealed).toEqual([false, true, false, true, false])
    expect(r.spent).toBe(2)
  })

  it('matches the paper’s algorithm on the same Laplace draws', () => {
    const s = stream('thresholdout')
    const m = 60
    const train = lossMatrix(child(s, 'train'), m, 50)
    const holdout = lossMatrix(child(s, 'holdout'), m, 50)
    const [T, sigma, B] = [0.05, 0.02, 10]
    const r = thresholdout(child(s, 'noise'), train, holdout, { threshold: T, sigma, budget: B })
    // Figure 1 of Dwork et al., with Lap(b) drawn by inversion from one uniform, in the order of the figure.
    const noise = child(s, 'noise')
    const lap = (b: number) => {
      const c = uniform(noise) - 0.5
      return -b * Math.sign(c) * Math.log1p(-2 * Math.abs(c))
    }
    let budget = B
    let That = T + lap(2 * sigma)
    const expected: number[] = []
    for (let j = 0; j < m; j++) {
      if (budget < 1) {
        expected.push(NaN)
        continue
      }
      const eta = lap(4 * sigma)
      const [ht, hh] = [mean(train[j]), mean(holdout[j])]
      if (Math.abs(hh - ht) > That + eta) {
        const xi = lap(sigma)
        const gamma = lap(2 * sigma)
        budget -= 1
        That = T + gamma
        expected.push(hh + xi)
      } else expected.push(ht)
    }
    expect(Array.from(r.answers)).toEqual(expected)
    expect(r.spent).toBe(B - budget)
    expect(r.spent).toBeGreaterThan(0)
    expect(() => thresholdout(s, [[1]], [[1], [0]])).toThrow(/queries/)
  })
})

describe('boostingAttack', () => {
  it('biases a full-precision leaderboard by order sqrt(k / n), the Ladder much less; fresh losses stay at 1/2', () => {
    const s = stream('attack')
    const [n, k] = [1000, 400]
    const exact = boostingAttack(s, { n, k, mechanism: 'exact' })
    const rounded = boostingAttack(s, { n, k, mechanism: 'rounded', digits: 5 })
    const lad = boostingAttack(s, { n, k, mechanism: 'ladder', eta: 0.01 })
    const free = boostingAttack(s, { n, k, mechanism: 'parameter-free' })
    // The same random submissions: rounding to 5 decimals does not change any decision when n = 1000.
    expect(Array.from(rounded.kept)).toEqual(Array.from(exact.kept))
    expect(exact.kept[k - 1]).toBeGreaterThan(k / 3)
    // The bias of the vote (Blum and Hardt, Theorem 5.1); the fresh loss is 1/2 within sampling error.
    expect(exact.holdout[k - 1]).toBeLessThan(0.5 - 0.1)
    expect(exact.reported[k - 1]).toBe(exact.holdout[k - 1])
    for (const r of [exact, lad, free]) expect(Math.abs(r.fresh[k - 1] - 0.5)).toBeLessThan(4 * Math.sqrt(0.25 / n))
    for (const r of [lad, free]) {
      expect(r.kept[k - 1]).toBeLessThan(30)
      expect(r.holdout[k - 1]).toBeGreaterThan(0.5 - 0.08)
    }
    // Answers before any submission is kept: the vote is the tie-breaking bits, its loss near 1/2.
    expect(boostingAttack(s, { n, k: 3, mechanism: 'ladder', eta: 0.5 }).kept[2]).toBe(1)
    // Deterministic per stream.
    expect(Array.from(boostingAttack(s, { n, k, mechanism: 'exact' }).holdout)).toEqual(Array.from(exact.holdout))
  })

  it('against Thresholdout: a spent budget answers nothing', () => {
    const r = boostingAttack(stream('attack'), { n: 500, k: 100, mechanism: 'thresholdout', threshold: 0, budget: 3 })
    expect(r.spent).toBe(3)
    expect(r.answers.slice(-10).every(Number.isNaN)).toBe(true)
    expect(() => boostingAttack(stream(0), { n: 1, k: 1, mechanism: 'exact' })).toThrow(DomainError)
  })
})

describe('expectedGaussianMaximum', () => {
  it('matches scipy quadrature of the density of the maximum, and the closed form at J = 2', () => {
    for (const c of fx.expected_maximum) expect(expectedGaussianMaximum(c.J)).toBeCloseTo(c.value, 10)
    expect(expectedGaussianMaximum(2)).toBeCloseTo(1 / Math.sqrt(Math.PI), 13)
    expect(expectedGaussianMaximum(10)).toBeCloseTo(1.5388, 4)
    for (const J of [2, 10, 1000]) expect(expectedGaussianMaximum(J)).toBeLessThan(Math.sqrt(2 * Math.log(J)))
    expect(() => expectedGaussianMaximum(0.5)).toThrow(DomainError)
  })
})

/** Evaluate a worker task tree as `aifn-render`'s compute worker does: nested calls first, modules by import. */
async function evaluate(x: unknown): Promise<unknown> {
  if (typeof x === 'object' && x !== null && typeof (x as { $call?: unknown }).$call === 'string') {
    const { $call, args } = x as { $call: string; args: unknown[] }
    const cut = $call.lastIndexOf('/')
    const [module, key] = [$call.slice(0, cut), $call.slice(cut + 1)]
    const ns = (await import(`aifn-compute/${module}`).catch(() => import(`aifn-methods/${module}`))) as Record<
      string,
      unknown
    >
    const f = ns[key] as (...a: unknown[]) => unknown
    return f(...(await Promise.all(args.map(evaluate))))
  }
  if (Array.isArray(x)) return Promise.all(x.map(evaluate))
  return x
}

describe('worker path', () => {
  it('runs each function from a task tree, and its result survives structured cloning', async () => {
    const s = { $call: 'foundation/random/stream', args: ['worker'] }
    const tasks = [
      { $call: 'theory/adaptive-holdout/boostingAttack', args: [s, { n: 200, k: 20, mechanism: 'parameter-free' }] },
      { $call: 'theory/adaptive-holdout/boostingAttack', args: [s, { n: 200, k: 20, mechanism: 'thresholdout' }] },
      { $call: 'theory/adaptive-holdout/ladder', args: [lossMatrix(stream(1), 5, 10), { eta: 0.1 }] },
      { $call: 'theory/adaptive-holdout/ladderParameterFree', args: [lossMatrix(stream(1), 5, 10)] },
      { $call: 'theory/adaptive-holdout/thresholdout', args: [s, [[1, 0]], [[0, 0]], { budget: 1 }] },
      { $call: 'theory/adaptive-holdout/expectedGaussianMaximum', args: [10] },
    ]
    for (const task of tasks) {
      const result = await evaluate(task)
      expect(structuredClone(result)).toEqual(result)
    }
    const direct = boostingAttack(stream('worker'), { n: 200, k: 20, mechanism: 'parameter-free' })
    expect(await evaluate(tasks[0])).toEqual(direct)
  })

  it('registers every export', () => {
    expect(Object.keys(adaptiveHoldoutFunctions).sort()).toEqual(
      ['boostingAttack', 'expectedGaussianMaximum', 'ladder', 'ladderParameterFree', 'thresholdout'].sort(),
    )
  })
})
