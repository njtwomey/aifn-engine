/**
 * Convolutional sparse coding: synthesis against the explicit dictionary of every shift of every filter; the coding
 * gradient against finite differences; the worked example of the note (one filter, exact activations); learning that
 * never increases its objective and recovers a planted pulse.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  convolutionalDictionaryLearning,
  convolutionalDictionaryLearningSteps,
  convolutionalSparseCode,
  convolutionalSparseCodeSteps,
  convolutionalSynthesis,
} from 'aifn-compute/signal/sparse'

const D = [
  [1, -2, 0.5],
  [0.3, 1, 1, -1],
].map((f) => [...f, 0].slice(0, 4)) // two filters of length 4
const Z = [
  [0, 1, 0, 0, -0.5, 0, 2],
  [0.7, 0, 0, 0, 0, 1, 0],
]

describe('convolutionalSynthesis', () => {
  it('equals the banded Toeplitz dictionary of all shifts times the stacked maps', () => {
    const K = 2
    const L = 4
    const M = 7
    const N = M + L - 1
    // Column (k, m) of the dictionary is filter k placed at m.
    const columns = Array.from({ length: K * M }, (_, c) => {
      const [k, m] = [Math.floor(c / M), c % M]
      const col = new Array<number>(N).fill(0)
      D[k].forEach((v, l) => (col[m + l] = v))
      return col
    })
    const stacked = Z.flat()
    const expected = Array.from({ length: N }, (_, n) => columns.reduce((a, col, c) => a + col[n] * stacked[c], 0))
    const got = toFlat(convolutionalSynthesis(D, Z))
    expect(got.length).toBe(N)
    expected.forEach((v, n) => expect(got[n]).toBeCloseTo(v, 12))
  })
})

describe('convolutionalSparseCode', () => {
  it("matches the note's worked example: activations of plus or minus (1 − λ/6) and exact zeros elsewhere", () => {
    const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]
    const lambda = 0.1
    const { Z: maps, converged } = convolutionalSparseCode([[1, 2, 1]], y, { lambda, maxSteps: 2000, tolerance: 1e-12 })
    expect(converged).toBe(true)
    const z = toFlat(maps)
    z.forEach((v, m) => {
      if (m === 2) expect(v).toBeCloseTo(1 - lambda / 6, 9)
      else if (m === 8) expect(v).toBeCloseTo(-(1 - lambda / 6), 9)
      else expect(Math.abs(v)).toBe(0)
    })
  })

  it('has the gradient of the smooth part: finite differences agree', () => {
    const y = [0.3, -1, 2, 0.5, 0, 1, -0.4, 0.2, 1.5, -0.3]
    const alg = convolutionalSparseCodeSteps(D, y, { lambda: 0.1, init: [Array(7).fill(0.1), Array(7).fill(-0.2)] })
    const s0 = alg.init(undefined, stream(1))
    const x = Array.from(toFlat(s0.x))
    const fit = (v: number[]) => {
      const r = toFlat(convolutionalSynthesis(D, [v.slice(0, 7), v.slice(7)])).map((f, n) => y[n] - f)
      return 0.5 * r.reduce((a, b) => a + b * b, 0)
    }
    const g = toFlat(s0.gradY)
    for (let i = 0; i < x.length; i++) {
      const h = 1e-6
      const up = x.slice()
      const down = x.slice()
      up[i] += h
      down[i] -= h
      expect(g[i]).toBeCloseTo((fit(up) - fit(down)) / (2 * h), 6)
    }
  })
})

describe('convolutionalDictionaryLearning', () => {
  // Six copies of the pulse (1, 3, 1) at irregular places and scales.
  const y = new Array<number>(60).fill(0)
  for (const [t, a] of [
    [3, 1],
    [11, -0.5],
    [19, 2],
    [30, 1.5],
    [41, -1],
    [50, 0.8],
  ])
    [1, 3, 1].forEach((v, l) => (y[t + l] += a * v))

  it('never increases the objective, and keeps every filter in the unit ball', () => {
    const alg = convolutionalDictionaryLearningSteps(y, { filters: 2, length: 6, lambda: 0.1 })
    const t = trace(alg, undefined, 15, {
      record: { objective: (s) => s.objective },
    })
    const objectives = toFlat(t.series.objective)
    for (let i = 1; i < objectives.length; i++)
      expect(objectives[i]).toBeLessThanOrEqual(objectives[i - 1] * (1 + 1e-9))
    for (const f of toRows(run(alg, undefined, 15).D)) expect(Math.hypot(...f)).toBeLessThanOrEqual(1 + 1e-9)
  })

  it('recovers a planted pulse at some shift within the filter, on every seed', () => {
    const unit = [1, 3, 1].map((v) => v / Math.sqrt(11))
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const { D: learnt } = convolutionalDictionaryLearning(y, {
        filters: 1,
        length: 6,
        lambda: 0.1,
        maxSteps: 60,
        stream: stream(seed),
      })
      const f = toRows(learnt)[0]
      const best = Math.max(...[0, 1, 2, 3].map((s) => Math.abs(unit.reduce((a, v, l) => a + v * f[s + l], 0))))
      expect(best).toBeGreaterThan(0.999)
    }
  })

  it('rejects filters longer than the signal', () => {
    expect(() => convolutionalDictionaryLearningSteps([1, 2], { filters: 1, length: 3, lambda: 0.1 })).toThrow(/length/)
  })
})
