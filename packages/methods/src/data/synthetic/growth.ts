/**
 * A seeded synthetic growth-chart data set: a positive measurement against age whose median, spread and skewness all
 * change with age, drawn from a Box–Cox Cole–Green (LMS) law, y | age ~ BCCG(μ(age), σ(age), ν(age)) (Cole and Green,
 * 1992). μ is close to the median (an infant phase, steady growth and a pubertal spurt), σ to the coefficient of
 * variation (rising through childhood) and ν is the Box–Cox power (1 at birth, symmetric; negative in adolescence,
 * right-skewed). The truth (`GROWTH_TRUTH`) gives the parameter curves and every centile.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'
import { boxCoxColeGreenDistributional } from 'aifn-compute/probability/likelihoods'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'

const logistic = (t: number) => 1 / (1 + Math.exp(-t))
const BCCG = boxCoxColeGreenDistributional()

/** The law of `growthChart`: BCCG parameter curves over age in years, and the centiles they imply. */
export const GROWTH_TRUTH = {
  ages: [0, 18] as const,
  /** μ(age): close to the median. */
  mu: (a: number) => 3.4 + 8.6 * (1 - Math.exp(-a)) + 2.1 * a + 22 * logistic((a - 13) / 1.5),
  /** σ(age): close to the coefficient of variation. */
  sigma: (a: number) => 0.1 + 0.08 * logistic((a - 9) / 2),
  /** ν(age): the Box–Cox power; 1 is symmetric, below 1 right-skewed. */
  nu: (a: number) => 1 - 2 * logistic((a - 9) / 2.5),
  /** The p-centile at age a. */
  centile(a: number, p: number): number {
    return BCCG.quantile(p, [this.mu(a), this.sigma(a), this.nu(a)])
  },
}

/** Options of `growthChart`. */
export interface GrowthChartOptions {
  /** Rows. Default 1000. */
  n?: number
  /** The largest age (years); ages are uniform on [0, maxAge]. Default 18. */
  maxAge?: number
}

/** The growth-chart data set (see the module comment); x is age (sorted), y the measurement, `f` the true median. */
export function growthChart(s: Stream, options: GrowthChartOptions = {}): Dataset {
  const { n = 1000, maxAge = 18 } = options
  checkCount(n, 'growthChart')
  if (!(maxAge > 0 && maxAge <= 18)) throw new DomainError('growthChart', 'growthChart: maxAge must be in (0, 18]')
  const ages = Float64Array.from(toFlat(uniform(child(s, 'age'), 0, maxAge, { shape: [n] }) as Tensor)).sort()
  const u = toFlat(uniform(child(s, 'y'), 0, 1, { shape: [n] }) as Tensor)
  const y = Float64Array.from(ages, (a, i) => GROWTH_TRUTH.centile(a, u[i]))
  return {
    kind: 'dataset',
    x: matrix(ages, n, 1),
    y: vector(y),
    f: vector(Float64Array.from(ages, (a) => GROWTH_TRUTH.centile(a, 0.5))),
    meta: {
      name: 'growth chart',
      description: `${n} synthetic measurements against age uniform on [0, ${maxAge}] years, drawn from BCCG(μ(age), σ(age), ν(age)) with median, spread and skewness all changing with age.`,
      task: 'regression',
      featureNames: ['age'],
      targetName: 'measurement',
      key: s.key,
      recipe: generatorRecipe('growthChart', s.key, { n, maxAge }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'growthChart',
    name: 'Growth chart',
    summary:
      'A positive measurement against age from a Box–Cox Cole–Green law whose median, spread and skewness all change with age: the centile-curve problem.',
    task: 'regression',
    output: 'dataset',
    knobs: space({ n: int(10, 20000, { default: 1000 }), maxAge: real(1, 18, { default: 18 }) }),
    truth: false,
    random: true,
    notes: ['generalised-additive-models-for-location-scale-and-shape'],
    cite: ['cole1992'],
  },
  growthChart,
)
