import { describe, expect, it } from 'vitest'
import { metricRegistry } from 'aifn-compute/learning/metrics'
import { Normal } from 'aifn-compute/probability/distributions'
import { tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

/**
 * Every registered metric against scikit-learn, scipy, published worked examples or a direct numpy computation
 * (gen/learning/_metric_cases.py). Each case holds the metric's positional arguments, options last.
 */
type Case = { args: unknown[]; value: number }
const F = fixture<{ metrics: Record<string, Case[]> }>('learning/metrics')

/** Arguments the JSON cannot carry: logScore takes a Normal predictive built from means and sds. */
function argumentsOf(key: string, args: unknown[]): unknown[] {
  if (key !== 'logScore') return args
  const { mean, sd } = args[1] as { mean: number[]; sd: number[] }
  return [args[0], Normal(tensor(mean), tensor(sd))]
}

describe('metrics match their references', () => {
  it('every registered metric has a reference case', () => {
    expect(Object.keys(metricRegistry).filter((k) => !(k in F.metrics))).toEqual([])
  })
  for (const [key, cases] of Object.entries(F.metrics))
    it.each(cases.map((c, i) => [i, c] as const))(`${key} #%i`, (_i, c) => {
      const metric = metricRegistry[key] as unknown as (...a: unknown[]) => number
      const got = metric(...argumentsOf(key, c.args))
      expect(Math.abs(got - c.value), `${key}: ${got} vs ${c.value}`).toBeLessThanOrEqual(
        1e-10 * (1 + Math.abs(c.value)),
      )
    })
})
