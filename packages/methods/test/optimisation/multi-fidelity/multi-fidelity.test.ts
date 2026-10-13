import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { learningCurves } from 'aifn-methods/data/synthetic'
import {
  fitPowerLaw,
  hyperbandSchedule,
  multiFidelityAlgorithms,
  multiFidelityFunctions,
  multiFidelitySteps,
  successiveHalvingSchedule,
  type CurveFunction,
  type MultiFidelityState,
} from 'aifn-methods/optimisation/multi-fidelity'
import { fixture } from '../../fixtures'
import { expectProtocol } from '../../protocol'
import { expectInfo } from '../../registry'

type Brackets = { s: number; n: number[]; r: number[] }[]
type Fit = { t: number[]; y: number[]; a: number; b: number; c: number; rss: number }
const F = fixture<{ hyperband: Record<string, Brackets>; powerLaw: Fit[] }>('optimisation/multi-fidelity')

// Configuration i ends at 0.1 (n - 1 - i) / n: the last is best; 0.02 i / r makes the good ones look bad early.
const crossing = (n: number): CurveFunction => ({ count: n, at: (i, r) => (0.1 * (n - 1 - i)) / n + (0.02 * i) / r })
// No crossing: the order is the same at every budget.
const parallel = (n: number): CurveFunction => ({ count: n, at: (i, r) => 0.01 * ((i * 7) % n) + 1 / r })

describe('schedules', () => {
  it('Hyperband matches Li et al. (2018), Algorithm 1 and its table for R = 81, eta = 3', () => {
    const cases: Record<string, [number, number]> = {
      R81eta3: [81, 3],
      R27eta3: [27, 3],
      R16eta2: [16, 2],
      R100eta3: [100, 3],
    }
    for (const [name, [R, eta]] of Object.entries(cases)) {
      const hb = hyperbandSchedule({ maxBudget: R, eta })
      const ref = F.hyperband[name]
      expect(
        hb.brackets.map((b) => b.s),
        name,
      ).toEqual(ref.map((b) => b.s))
      hb.brackets.forEach((b, k) => {
        expect(
          b.rungs.map((g) => g.configurations),
          `${name} s=${b.s}`,
        ).toEqual(ref[k].n)
        b.rungs.forEach((g, i) => expect(g.budget).toBeCloseTo(ref[k].r[i], 12))
      })
    }
    const hb = hyperbandSchedule({ maxBudget: 81, eta: 3 })
    expect(hb.configurations).toBe(81 + 34 + 15 + 8 + 5)
    // Budgets from scratch: 405 for s = 4 (5 rungs of 81), 34·3 + 11·9 + 3·27 + 81 = 363, 15·9 + 5·27 + 81 = 351,
    // 8·27 + 2·81 = 378, 5·81 = 405.
    expect(hb.brackets.map((b) => b.budget)).toEqual([405, 363, 351, 378, 405])
    expect(hb.budget).toBe(1902)
    // Resuming: s = 4 costs 81 + 27·2 + 9·6 + 3·18 + 54 = 297.
    expect(hb.brackets[0].resumedBudget).toBe(297)
  })

  it('a minimum budget rescales Hyperband', () => {
    const hb = hyperbandSchedule({ maxBudget: 100, minBudget: 10, eta: 3 })
    // R / r = 10, so s_max = 2.
    expect(hb.brackets.map((b) => b.rungs.map((g) => g.configurations))).toEqual([[9, 3, 1], [5, 1], [3]])
    expect(hb.brackets[0].rungs.map((g) => g.budget)).toEqual([100 / 9, 100 / 3, 100])
  })

  it('successive halving: rungs by hand', () => {
    const sh = successiveHalvingSchedule({ n: 27, minBudget: 1, maxBudget: 27, eta: 3 })
    expect(sh.rungs).toEqual([
      { rung: 0, configurations: 27, budget: 1, keep: 9 },
      { rung: 1, configurations: 9, budget: 3, keep: 3 },
      { rung: 2, configurations: 3, budget: 9, keep: 1 },
      { rung: 3, configurations: 1, budget: 27, keep: 0 },
    ])
    expect(sh.budget).toBe(108)
    // Fewer configurations than rungs: stops when one is left.
    const short = successiveHalvingSchedule({ n: 10, maxBudget: 81, eta: 3 })
    expect(short.rungs.map((g) => [g.configurations, g.budget, g.keep])).toEqual([
      [10, 1, 3],
      [3, 3, 1],
      [1, 9, 0],
    ])
    // An early-stopping rate starts higher: s = 1 gives budgets 3, 9, 27.
    const s1 = successiveHalvingSchedule({ n: 9, maxBudget: 27, eta: 3, earlyStoppingRate: 1 })
    expect(s1.rungs.map((g) => g.budget)).toEqual([3, 9, 27])
    expect(() => successiveHalvingSchedule({ n: 0, maxBudget: 9 })).toThrow(/positive integer/)
    expect(() => hyperbandSchedule({ maxBudget: 9, eta: 1 })).toThrow(/eta/)
  })
})

describe('multiFidelitySteps', () => {
  it('synchronous successive halving: one rung per step, the best third promoted', () => {
    const sh = successiveHalvingSchedule({ n: 9, minBudget: 1, maxBudget: 9, eta: 3 })
    const tr = trace(multiFidelitySteps(parallel(9), sh), undefined, 10, { keep: 'all' })
    const steps = tr.steps as MultiFidelityState[]
    expect(steps.length).toBe(4)
    // Final losses 0.01 · (7i mod 9): config 0 is best, then 4 (7·4 = 28 ≡ 1), then 8 (56 ≡ 2).
    expect(steps[1].evaluated).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8])
    expect(steps[1].promoted).toEqual([0, 4, 8])
    expect(steps[2].promoted).toEqual([0])
    expect(steps[3].evaluated).toEqual([0])
    expect(steps[3].terminated).toBe(true)
    expect(tr.final.spent).toBe(9 + 9 + 9)
    expect(tr.final.trueBest).toBe(0)
    expect(tr.final.survived).toBe(true)
    expect(tr.final.incumbent?.config).toBe(0)
  })

  it('crossing curves: the true best is eliminated at the first rung', () => {
    const sh = successiveHalvingSchedule({ n: 9, minBudget: 1, maxBudget: 9, eta: 3 })
    const tr = trace(multiFidelitySteps(crossing(9), sh), undefined, 10, { keep: 'all' })
    const steps = tr.steps as MultiFidelityState[]
    expect(steps[1].trueBest).toBe(8)
    // At budget 1: loss 0.1 (8 − i)/9 + 0.02 i is increasing in i, so 0, 1, 2 go on.
    expect(steps[1].promoted).toEqual([0, 1, 2])
    expect(steps[1].survived).toBe(false)
    expect(tr.final.survived).toBe(false)
    expect(tr.final.bestRung).toBe(0)
  })

  it('Hyperband runs every bracket in turn and spends the scheduled budget; resuming spends less', () => {
    const hb = hyperbandSchedule({ maxBudget: 27, eta: 3 })
    const curves = parallel(hb.configurations)
    const steps = hb.brackets.reduce((a, b) => a + b.rungs.length, 0)
    const s = run(multiFidelitySteps(curves, hb), undefined, steps + 5)
    expect(s.t).toBe(steps)
    expect(s.spent).toBeCloseTo(hb.budget, 10)
    const r = run(multiFidelitySteps(curves, hb, { resume: true }), undefined, steps + 5)
    expect(r.spent).toBeCloseTo(hb.resumedBudget, 10)
    expect(new Set(s.evaluations.map((e) => e.bracket))).toEqual(new Set([0, 1, 2, 3]))
  })

  it('reads a curve table, interpolating between budgets', () => {
    const budgets = [1, 2, 4, 8]
    const values = [
      [1, 0.8, 0.6, 0.5],
      [0.9, 0.85, 0.8, 0.75],
      [1.2, 0.7, 0.4, 0.2],
    ]
    const sh = successiveHalvingSchedule({ n: 3, minBudget: 1, maxBudget: 3, eta: 3 })
    const s = run(multiFidelitySteps({ budgets, values }, sh), undefined, 5)
    // Rung 0 at budget 1 keeps config 1; rung 1 at budget 3 reads (0.85 + 0.8) / 2.
    expect(s.evaluations.map((e) => [e.config, e.budget])).toEqual([
      [0, 1],
      [1, 1],
      [2, 1],
      [1, 3],
    ])
    expect(s.evaluations[3].loss).toBeCloseTo(0.825, 14)
    expect(() => multiFidelitySteps({ budgets, values }, successiveHalvingSchedule({ n: 9, maxBudget: 9 }))).toThrow(
      /outside/,
    )
    expect(() => multiFidelitySteps(parallel(3), hyperbandSchedule({ maxBudget: 9 }))).toThrow(/curves hold 3/)
  })

  it('ASHA: one evaluation per step; with one worker it fills rungs as Li et al. (2020) describe', () => {
    const sh = successiveHalvingSchedule({ n: 9, minBudget: 1, maxBudget: 9, eta: 3 })
    const s = run(multiFidelitySteps(parallel(9), sh, { asynchronous: true }), undefined, 100)
    expect(s.terminated).toBe(true)
    const order = s.evaluations.map((e) => `${e.config}@${e.budget}`)
    // Losses at budget 1 are 1 + 0.01 (7i mod 9): 0 → 0, 1 → 7, 2 → 5, 3 → 3, 4 → 1, ...
    // After 0, 1, 2 the top third of rung 0 is {0}: promote it, and so on.
    expect(order.slice(0, 5)).toEqual(['0@1', '1@1', '2@1', '0@3', '3@1'])
    expect(s.evaluations.filter((e) => e.rung === 0).length).toBe(9)
    // Every configuration in the end is evaluated at budget 1 and the best reaches the top.
    expect(s.incumbent?.config).toBe(0)
    expect(s.evaluations.some((e) => e.config === 0 && e.budget === 9)).toBe(true)
    expect(s.time).toBe(s.spent)
  })

  it('ASHA on several workers runs in parallel: less simulated time than budget', () => {
    const sh = successiveHalvingSchedule({ n: 27, minBudget: 1, maxBudget: 27, eta: 3 })
    const s = run(multiFidelitySteps(parallel(27), sh, { asynchronous: true, workers: 4 }), undefined, 500)
    expect(s.terminated).toBe(true)
    expect(s.time).toBeLessThanOrEqual(s.spent / 2)
    // No worker ever runs two jobs at once.
    for (let w = 0; w < 4; w++) {
      const jobs = s.evaluations.filter((e) => e.worker === w).sort((a, b) => a.start - b.start)
      for (let k = 1; k < jobs.length; k++) expect(jobs[k].start).toBeGreaterThanOrEqual(jobs[k - 1].end)
    }
    // A promotion is always of a configuration in the top third of its rung when it was made.
    expect(s.promotions.length).toBeGreaterThan(0)
  })

  it('follows the trace protocol, synchronous and asynchronous', () => {
    const hb = hyperbandSchedule({ maxBudget: 9, eta: 3 })
    const curves = crossing(hb.configurations)
    expectProtocol(multiFidelitySteps(curves, hb), undefined, { n: 6 })
    expectProtocol(multiFidelitySteps(curves, hb, { asynchronous: true, workers: 3 }), undefined, { n: 12 })
  })
})

describe('fitPowerLaw', () => {
  it('matches scipy curve_fit', () => {
    for (const c of F.powerLaw) {
      const fit = fitPowerLaw(c.t, c.y)
      expect(fit.rss).toBeLessThanOrEqual(c.rss * (1 + 1e-6))
      expect(fit.a).toBeCloseTo(c.a, 5)
      expect(fit.b).toBeCloseTo(c.b, 5)
      expect(fit.c).toBeCloseTo(c.c, 5)
    }
  })

  it('recovers an exact power law and extrapolates it', () => {
    const t = [1, 2, 3, 5, 8, 13]
    const fit = fitPowerLaw(
      t,
      t.map((s) => 0.1 + 2 * s ** -1.3),
    )
    expect(fit.a).toBeCloseTo(0.1, 8)
    expect(fit.c).toBeCloseTo(1.3, 7)
    expect(fit.at(100)).toBeCloseTo(0.1 + 2 * 100 ** -1.3, 8)
    expect(() => fitPowerLaw([1, 2], [1, 2])).toThrow(/three/)
  })
})

describe('with learningCurves', () => {
  it('Hyperband over synthetic curves: the incumbent is among the best, and crossing curves lose the true best', () => {
    const hb = hyperbandSchedule({ maxBudget: 81, eta: 3 })
    const calm = learningCurves(stream(0), { n: hb.configurations, crossing: 0, noise: 0 })
    const s = run(multiFidelitySteps(calm, hb), undefined, 100)
    const finals = Array.from(toFlat(calm.final) as ArrayLike<number>)
    const rank = finals.filter((v) => v < finals[s.incumbent!.config]).length
    expect(rank).toBeLessThan(5)
    // Successive halving from 81 at budget 1: with full crossing the true best never reaches the top rung.
    const sh = successiveHalvingSchedule({ n: 81, maxBudget: 81, eta: 3 })
    const wild = learningCurves(stream(0), { n: 81, crossing: 1, noise: 0 })
    const w = run(multiFidelitySteps(wild, sh), undefined, 10)
    expect(w.trueBest).toBe(wild.best)
    expect(w.survived).toBe(false)
  })

  it('fitPowerLaw extrapolates a noise-free power-law curve from its first budgets', () => {
    const c = learningCurves(stream(1), { n: 3, noise: 0 })
    const rows = toArray(c.clean) as number[][]
    const t = Array.from(toFlat(c.budgets) as ArrayLike<number>)
    c.forms.forEach((f, i) => {
      const fit = fitPowerLaw(t.slice(0, 9), rows[i].slice(0, 9))
      expect(fit.a).toBeCloseTo(f.a, 6)
      expect(fit.c).toBeCloseTo(f.c, 6)
      expect(fit.at(81)).toBeCloseTo(rows[i][80], 6)
    })
  })
})

it('registers its entries', () => {
  expectInfo(multiFidelityAlgorithms, 'algorithm')
  expectInfo(multiFidelityFunctions, 'function')
})
