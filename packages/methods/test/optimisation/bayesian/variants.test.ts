/**
 * The options of `bayesianOptimisationSteps` by behaviour: named objectives (the same run as the function and its box),
 * the random-forest surrogate (the state's mean and spread are those of its trees), interleaved random points, the log
 * cost (the log-normal expected improvement), batches (each strategy's points are the next step's evaluations, with
 * their fantasies or penalties), and constraints (expected improvement over the best feasible value times the
 * probability of feasibility); and `bayesianOptimisationStudy`. A worker task tree, evaluated as the compute worker of
 * aifn-render does (addresses resolved to module exports, nested calls first), runs a whole study and a whole trace,
 * and its result survives `structuredClone`.
 */
import { describe, expect, it } from 'vitest'
import { forrester } from 'aifn-methods/data/objectives'
import {
  bayesianOptimisationStudy,
  bayesianOptimisationSteps,
  expectedImprovement,
  expectedImprovementLogNormal,
  type BayesianObjective,
  type BayesianOptimisationState,
  type BatchStrategy,
} from 'aifn-methods/optimisation/bayesian'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { normalCdf } from 'aifn-compute/numerics/special'

const F = forrester()
const forr = (u: number) => (6 * u - 2) ** 2 * Math.sin(12 * u - 4)

/** The index of the largest entry (the first of ties). */
const argmax = (a: ArrayLike<number>): number => {
  let k = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i
  return k
}

/** Entries within `tol` of each other. */
const near = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i]), `entry ${i}`).toBeLessThanOrEqual(tol)
}

describe('named objectives', () => {
  it('a name runs the same loop as the function with its box, and carries the minimum', () => {
    const opts = { grid: 41, refit: false as const }
    const a = run(bayesianOptimisationSteps('forrester', opts), undefined, 3, { stream: stream(0) })
    const b = run(bayesianOptimisationSteps(F.value, F, opts), undefined, 3, { stream: stream(0) })
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
    expect(toFlat(a.y)).toEqual(toFlat(b.y))
    const alg = bayesianOptimisationSteps({ name: 'forrester' }, null, opts)
    expect(alg.minimum).toBe(F.minimumValue)
    expect(alg.objective).toBe('forrester')
    expect(Array.from(toFlat(alg.box.hi))).toEqual([1])
  })
  it('kebab-case names and parameters reach the factory, noise drawn from the evaluation streams', () => {
    const alg = bayesianOptimisationSteps(
      { name: 'svm-tuning-surface', noise: 0.02 },
      { grid: 6, refit: false, init: 4 },
    )
    expect(alg.objective).toBe('svmTuningSurface')
    const a = run(alg, undefined, 1, { stream: stream(5) })
    const b = run(alg, undefined, 1, { stream: stream(5) })
    expect(toFlat(a.y)).toEqual(toFlat(b.y))
    expect(a.evaluations).toBe(5)
  })
  it('throws for an unknown name or a function without a box', () => {
    expect(() => bayesianOptimisationSteps('nope')).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(F.value as BayesianObjective, null)).toThrow(DomainError)
  })
})

describe('random-forest surrogate, interleaved random points and the log cost', () => {
  const alg = bayesianOptimisationSteps('forrester', {
    surrogate: 'random-forest',
    interleaveRandom: 3,
    grid: 51,
  })
  const tr = trace(alg, undefined, 12, { stream: stream(1), keep: 'all' })

  it('reports the trees’ mean and spread on the grid, and chooses by the acquisition except every third time', () => {
    for (const s of tr.steps) {
      expect(s.treePredictions!.shape).toEqual([10, 51])
      const p = toFlat(s.treePredictions!)
      const mean = Array.from({ length: 51 }, (_, i) => p.filter((_, k) => k % 51 === i).reduce((a, b) => a + b) / 10)
      const sd = Array.from({ length: 51 }, (_, i) =>
        Math.sqrt(p.filter((_, k) => k % 51 === i).reduce((a, b) => a + (b - mean[i]) ** 2, 0) / 10),
      )
      near(toFlat(s.mean), mean, 1e-12)
      near(toFlat(s.sd), sd, 1e-12)
      expect(s.random).toBe((s.t + 1) % 3 === 0)
      if (!s.random) expect(s.nextIndex).toBe(argmax(toFlat(s.acquisition)))
      near(toFlat(s.acquisition), toFlat(expectedImprovement(s.mean, s.sd, s.best)), 0)
      expect(s.hyper).toEqual({})
      expect(Number.isNaN(s.logMarginal)).toBe(true)
    }
  })
  it('is reproducible from its stream and gets near the minimum', () => {
    const again = run(alg, undefined, 12, { stream: stream(1) })
    expect(toFlat(again.y)).toEqual(toFlat(tr.final.y))
    expect(tr.final.best).toBeLessThan(F.minimumValue + 0.5)
  })
  it('models the log of a positive cost and scores it by the log-normal expected improvement', () => {
    const cost = (x: Tensor) => forr(toFlat(x)[0]) + 7
    const s = run(bayesianOptimisationSteps(cost, F, { logCost: true, grid: 51 }), undefined, 4, { stream: stream(2) })
    near(toFlat(s.acquisition), toFlat(expectedImprovementLogNormal(s.mean, s.sd, s.best)), 0)
    // The GP interpolates the log cost at the evaluations that lie on the grid (the steps' points).
    const xs = toFlat(s.x)
    for (let i = s.evaluations - 4; i < s.evaluations; i++) {
      const g = Math.round(xs[i] * 50)
      expect(Math.abs(toFlat(s.mean)[g] - Math.log(toFlat(s.y)[i]))).toBeLessThan(1e-3)
    }
    expect(() =>
      run(
        bayesianOptimisationSteps(() => -1, F, { logCost: true, grid: 11 }),
        undefined,
        1,
        { stream: stream(0) },
      ),
    ).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(cost, F, { logCost: true, minimise: false })).toThrow(DomainError)
    expect(() => bayesianOptimisationSteps(cost, F, { surrogate: 'random-forest', acquisition: 'thompson' })).toThrow(
      DomainError,
    )
  })
})

describe('batches', () => {
  const strategies: BatchStrategy[] = [
    'kriging-believer',
    'constant-liar-min',
    'constant-liar-mean',
    'constant-liar-max',
    'local-penalisation',
    'thompson',
  ]
  it.each(strategies)('%s: each step evaluates the batch the last state chose', (strategy) => {
    const alg = bayesianOptimisationSteps('forrester', {
      batch: { size: 3, strategy },
      grid: 41,
      refit: false,
      init: 3,
    })
    const tr = trace(alg, undefined, 2, { stream: stream(7), keep: 'all' })
    const [s0, s1, s2] = tr.steps
    expect([s0.evaluations, s1.evaluations, s2.evaluations]).toEqual([3, 6, 9])
    for (const [a, b] of [
      [s0, s1],
      [s1, s2],
    ] as [BayesianOptimisationState, BayesianOptimisationState][]) {
      expect(a.batch!.shape).toEqual([3, 1])
      expect(Array.from(toFlat(b.x).slice(a.evaluations))).toEqual(Array.from(toFlat(a.batch!)))
      expect(toFlat(a.next)).toEqual(toFlat(a.batch!).slice(0, 1))
      const rows = toFlat(a.batchAcquisition!)
      a.batchIndices!.forEach((idx, j) => expect(idx).toBe(argmax(rows.slice(j * 41, j * 41 + 41))))
      near(toFlat(a.acquisition), rows.slice(0, 41), 0)
      if (strategy.startsWith('constant') || strategy === 'kriging-believer')
        expect(new Set(a.batchIndices).size).toBe(3)
      const ys = Array.from(toFlat(a.y))
      const z = toFlat(a.fantasies ?? fromData(new Float64Array(0), [0]))
      if (strategy === 'kriging-believer') expect(z[0]).toBeCloseTo(toFlat(a.mean)[a.nextIndex], 10)
      if (strategy === 'constant-liar-min') near(z, [Math.min(...ys), Math.min(...ys), Math.min(...ys)], 1e-12)
      if (strategy === 'constant-liar-max') near(z, [Math.max(...ys), Math.max(...ys), Math.max(...ys)], 1e-12)
      if (strategy === 'local-penalisation') {
        expect(a.fantasies).toBeNull()
        for (let i = 0; i < 41; i++) expect(rows[41 + i]).toBeLessThanOrEqual(rows[i] + 1e-15)
        expect(rows[41 + a.nextIndex]).toBeLessThan(rows[a.nextIndex])
      }
    }
    const again = run(alg, undefined, 2, { stream: stream(7) })
    expect(toFlat(again.y)).toEqual(toFlat(s2.y))
  })
  it('combines with the forest surrogate and with random interleaving', () => {
    const forest = bayesianOptimisationSteps('branin', {
      surrogate: 'random-forest',
      batch: { size: 2, strategy: 'constant-liar-mean' },
      grid: 11,
      init: 5,
    })
    const s = run(forest, undefined, 2, { stream: stream(4) })
    expect(s.evaluations).toBe(9)
    expect(s.batch!.shape).toEqual([2, 2])
    const random = bayesianOptimisationSteps('forrester', { batch: { size: 2 }, interleaveRandom: 1, grid: 21 })
    const r = run(random, undefined, 1, { stream: stream(4) })
    expect(r.random).toBe(true)
    expect(r.fantasies).toBeNull()
    expect(r.batchIndices!.length).toBe(2)
  })
  it('throws for a batch size below 1 or constraints with a batch', () => {
    expect(() => bayesianOptimisationSteps('forrester', { batch: { size: 0 } })).toThrow(DomainError)
    expect(() =>
      bayesianOptimisationSteps('forrester', { batch: { size: 2 }, constraints: [{ threshold: 0 }] }),
    ).toThrow(DomainError)
  })
})

describe('constraints', () => {
  const objective = (x: Tensor) => ({ value: forr(toFlat(x)[0]), constraints: [Math.sin(6 * toFlat(x)[0])] })
  const alg = bayesianOptimisationSteps(objective, F, { constraints: [{ threshold: 0 }], grid: 51 })
  const tr = trace(alg, undefined, 8, { stream: stream(0), keep: 'all' })

  it('scores the expected improvement over the best feasible value times the probability of feasibility', () => {
    for (const s of tr.steps) {
      const c = toFlat(s.constraintValues!)
      const y = toFlat(s.y)
      let fb = -1
      for (let i = 0; i < s.evaluations; i++) if (c[i] <= 0 && (fb < 0 || y[i] < y[fb])) fb = i
      expect(s.feasibleBestIndex).toBe(fb)
      const cm = toFlat(s.constraintMean!)
      const cs = toFlat(s.constraintSd!)
      const pf = Array.from(cm, (m, i) => (cs[i] > 0 ? normalCdf(-m / cs[i]) : m <= 0 ? 1 : 0))
      near(toFlat(s.feasibility!), pf, 1e-15)
      const ei = fb >= 0 ? toFlat(expectedImprovement(s.mean, s.sd, y[fb])) : null
      near(toFlat(s.acquisition), ei ? pf.map((p, i) => p * ei[i]) : pf, 1e-15)
      expect(s.nextIndex).toBe(argmax(toFlat(s.acquisition)))
    }
  })
  it('finds the constrained minimum on the grid', () => {
    const grid = toFlat(alg.grid)
    let best = Infinity
    for (const u of grid) if (Math.sin(6 * u) <= 0) best = Math.min(best, forr(u))
    expect(tr.final.feasibleBest).toBeLessThan(best + 0.05)
    expect(tr.final.best).toBeLessThanOrEqual(tr.final.feasibleBest)
  })
  it('scores the probability of feasibility alone while nothing is feasible', () => {
    const none = bayesianOptimisationSteps(objective, F, { constraints: [{ threshold: -2 }], grid: 21, refit: false })
    const s = run(none, undefined, 1, { stream: stream(0) })
    expect(s.feasibleBestIndex).toBe(-1)
    expect(Number.isNaN(s.feasibleBest)).toBe(true)
    near(toFlat(s.acquisition), toFlat(s.feasibility!), 0)
  })
  it('throws when the objective leaves the constraints out, or for another acquisition', () => {
    const bad = bayesianOptimisationSteps(F.value, F, { constraints: [{ threshold: 0 }], grid: 11 })
    expect(() => run(bad, undefined, 0, { stream: stream(0) })).toThrow(ShapeError)
    expect(() =>
      bayesianOptimisationSteps(objective, F, { constraints: [{ threshold: 0 }], acquisition: 'ucb' }),
    ).toThrow(DomainError)
  })
})

describe('bayesianOptimisationStudy', () => {
  const opts = { objective: 'forrester', seeds: [0, 1], iterations: 3, grid: 31, refit: false as const }
  const study = bayesianOptimisationStudy({ ...opts, acquisitions: ['ei'], betas: [1, 9] })

  it('returns arms by seeds by evaluations of regret curves', () => {
    expect(study.labels).toEqual(['ei', 'ucb (beta 1)', 'ucb (beta 9)'])
    expect(study.initial).toBe(3)
    expect(study.simpleRegret.shape).toEqual([3, 2, 6])
    expect(study.meanCumulativeRegret.shape).toEqual([3, 6])
    expect(Array.from(toFlat(study.evaluations))).toEqual([1, 2, 3, 4, 5, 6])
    const simple = toFlat(study.simpleRegret)
    const cumulative = toFlat(study.cumulativeRegret)
    const values = toFlat(study.values)
    for (let r = 0; r < 6; r++)
      for (let i = 0; i < 6; i++) {
        const o = r * 6 + i
        expect(simple[o]).toBeGreaterThanOrEqual(0)
        if (i > 0) expect(simple[o]).toBeLessThanOrEqual(simple[o - 1])
        if (i > 0) expect(cumulative[o]).toBeCloseTo(cumulative[o - 1] + values[o] - F.minimumValue, 12)
      }
  })
  it('is the loop run from stream(seed), with one initial design per seed across the arms', () => {
    const s = run(bayesianOptimisationSteps('forrester', { grid: 31, refit: false }), undefined, 3, {
      stream: stream(1),
    })
    expect(Array.from(toFlat(study.observed).slice(6, 12))).toEqual(Array.from(toFlat(s.y)))
    const obs = toFlat(study.observed)
    for (const a of [1, 2]) expect(Array.from(obs.slice(a * 12, a * 12 + 3))).toEqual(Array.from(obs.slice(0, 3)))
  })
})

// ── The worker path ──────────────────────────────────────────────────────────────────────────────────────────────

/** A task as aifn-render's `call` makes it. */
const call = (address: string, ...args: unknown[]) => ({ $call: address, args })

/** An address resolved as the compute worker does: the module's export, or an entry of one of its tables. */
async function resolve(address: string): Promise<unknown> {
  const cut = address.lastIndexOf('/')
  const module = address.slice(0, cut)
  const key = address.slice(cut + 1)
  let ns: Record<string, unknown>
  try {
    ns = (await import(`aifn-compute/${module}`)) as Record<string, unknown>
  } catch {
    ns = (await import(`aifn-methods/${module}`)) as Record<string, unknown>
  }
  if (key in ns) return ns[key]
  for (const v of Object.values(ns))
    if (v !== null && typeof v === 'object' && key in v) {
      const entry = (v as Record<string, { info?: { key?: unknown } }>)[key]
      if (entry?.info?.key === key) return entry
    }
  throw new Error(`no export '${key}' in '${module}'`)
}

/** A task tree evaluated as the worker does: calls innermost first, arrays and plain objects walked. */
async function evaluate(x: unknown): Promise<unknown> {
  if (typeof x === 'object' && x !== null && typeof (x as { $call?: unknown }).$call === 'string') {
    const t = x as { $call: string; args: unknown[] }
    const f = await resolve(t.$call)
    const args = await Promise.all(t.args.map(evaluate))
    return (f as (...a: unknown[]) => unknown)(...args)
  }
  if (Array.isArray(x)) return Promise.all(x.map(evaluate))
  if (typeof x === 'object' && x !== null && Object.getPrototypeOf(x) === Object.prototype) {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(x)) out[k] = await evaluate(v)
    return out
  }
  return x
}

/** A result made safe for `postMessage` as the worker does: functions dropped. */
function toMessage(x: unknown): unknown {
  if (typeof x === 'function') return undefined
  if (typeof x !== 'object' || x === null || ArrayBuffer.isView(x)) return x
  if (Array.isArray(x)) return x.map(toMessage)
  return Object.fromEntries(
    Object.entries(x)
      .filter(([, v]) => typeof v !== 'function')
      .map(([k, v]) => [k, toMessage(v)]),
  )
}

describe('the compute worker’s task path', () => {
  it('runs a named trace end to end, and the result survives structuredClone', async () => {
    // The task is plain data: it survives structuredClone itself.
    const task = structuredClone(
      call(
        'foundation/trace/trace',
        call('optimisation/bayesian/bayesianOptimisationSteps', 'branin', { grid: 9, refit: false, init: 4 }),
        undefined,
        3,
        { stream: call('foundation/random/stream', 3) },
      ),
    )
    const result = structuredClone(toMessage(await evaluate(task))) as { final: { y: { data: ArrayLike<number> } } }
    const direct = trace(bayesianOptimisationSteps('branin', { grid: 9, refit: false, init: 4 }), undefined, 3, {
      stream: stream(3),
    })
    expect(Array.from(result.final.y.data)).toEqual(Array.from(toFlat(direct.final.y)))
  })
  it('runs a multi-seed study', async () => {
    const args = { objective: 'forrester', seeds: 2, iterations: 2, grid: 21, refit: false, betas: [1, 4] }
    const task = call('optimisation/bayesian/bayesianOptimisationStudy', args)
    const out = structuredClone(toMessage(await evaluate(task))) as {
      labels: string[]
      meanSimpleRegret: { shape: number[]; data: ArrayLike<number> }
    }
    expect(out.labels).toEqual(['ucb (beta 1)', 'ucb (beta 4)'])
    expect(out.meanSimpleRegret.shape).toEqual([2, 5])
    expect(Array.from(out.meanSimpleRegret.data)).toEqual(
      Array.from(toFlat(bayesianOptimisationStudy(args as never).meanSimpleRegret)),
    )
  })
})
