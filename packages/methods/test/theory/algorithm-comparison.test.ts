/**
 * The Type I error study: per-stream determinism, its streamed and one-data-set forms against the study, rejection
 * rates near or below the nominal level for the calibrated tests when the learners are equally good, power when one is
 * handicapped, and the worker path.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  algorithmComparisonFunctions,
  TYPE_I_TESTS,
  typeIErrorReplicate,
  typeIErrorStudy,
  typeIErrorStudySteps,
} from 'aifn-methods/theory/algorithm-comparison'

describe('typeIErrorStudy', () => {
  it('is fixed by its stream, and agrees with its streamed and one-data-set forms', () => {
    const s = stream('type-i')
    const options = { n: 40, repeats: 2, replicates: 3 }
    const study = typeIErrorStudy(s, options)
    expect(typeIErrorStudy(s, options)).toEqual(study)
    const steps = [...typeIErrorStudySteps(s, options)]
    expect(steps.map((t) => t.replicates)).toEqual([1, 2, 3])
    expect(steps[2]).toEqual(study)
    const one = typeIErrorReplicate(child(s, 1), options)
    for (const t of TYPE_I_TESTS) expect(study.pValues[t][1]).toBe(one.p[t])
    expect(one.p.cvT).toBe(one.naiveByRepeats[0])
    expect(one.p.corrected).toBe(one.correctedByRepeats[1])
    for (const t of TYPE_I_TESTS)
      for (const p of study.pValues[t]) {
        expect(p).toBeGreaterThanOrEqual(0)
        expect(p).toBeLessThanOrEqual(1)
      }
    expect(study.byRepeats.repeats).toEqual([1, 2])
  })

  it('calibrated tests reject at about the nominal rate when the learners are equal; a handicap gives power', () => {
    const s = stream('calibration')
    const R = 60
    const study = typeIErrorStudy(s, { n: 40, repeats: 1, replicates: R })
    // Within three binomial standard errors of 5%.
    const tol = 0.05 + 3 * Math.sqrt((0.05 * 0.95) / R)
    for (const t of ['mcnemar', 'combinedF', 'corrected'] as const) expect(study.rates[t]).toBeLessThanOrEqual(tol)
    expect(Math.abs(study.trueDifference)).toBeLessThan(0.03)
    const power = typeIErrorStudy(s, { n: 40, repeats: 1, replicates: 20, handicap: 1.5 })
    expect(power.trueDifference).toBeGreaterThan(0.05)
    expect(power.rates.cvT).toBeGreaterThan(study.rates.cvT)
  })

  it('rejects bad options', () => {
    expect(() => typeIErrorStudy(stream(0), { repeats: 11 })).toThrow(DomainError)
    expect(() => typeIErrorReplicate(stream(0), { n: 5 })).toThrow(DomainError)
    expect(() => typeIErrorStudy(stream(0), { replicates: 0 })).toThrow(DomainError)
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
  it('runs the study from a task tree; the result and every streamed value survive structured cloning', async () => {
    const options = { n: 40, repeats: 2, replicates: 2 }
    const task = {
      $call: 'theory/algorithm-comparison/typeIErrorStudy',
      args: [{ $call: 'foundation/random/stream', args: ['comparing/type-i'] }, options],
    }
    const result = await evaluate(task)
    expect(structuredClone(result)).toEqual(result)
    expect(result).toEqual(typeIErrorStudy(stream('comparing/type-i'), options))
    const steps = (await evaluate({
      ...task,
      $call: 'theory/algorithm-comparison/typeIErrorStudySteps',
    })) as Iterable<unknown>
    for (const value of steps) expect(structuredClone(value)).toEqual(value)
    expect(Object.keys(algorithmComparisonFunctions).sort()).toEqual(
      ['typeIErrorReplicate', 'typeIErrorStudy', 'typeIErrorStudySteps'].sort(),
    )
  })
})
