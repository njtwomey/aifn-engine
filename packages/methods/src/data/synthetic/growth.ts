/**
 * A seeded synthetic growth-chart data set: a positive measurement against age whose median, spread and skewness all
 * change with age, drawn from a Box–Cox Cole–Green (LMS) law,
 * $y \mid a \sim \operatorname{BCCG}(\mu(a), \sigma(a), \nu(a))$ at age $a$ (Cole and Green, 1992). $\mu$ is close to
 * the median (an infant phase, steady growth and a pubertal spurt), $\sigma$ to the coefficient of variation (rising
 * through childhood) and $\nu$ is the Box–Cox power (1 at birth, symmetric; negative in adolescence, right-skewed). The
 * truth (`GROWTH_TRUTH`) gives the parameter curves and every centile.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'
import { boxCoxColeGreenDistributional } from 'aifn-compute/probability/likelihoods'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'

/**
 * The logistic function $1 / (1 + e^{-t})$, the smooth step of the parameter curves.
 *
 * @param t The point at which to evaluate it.
 * @returns The value, in $(0, 1)$.
 */
const logistic = (t: number) => 1 / (1 + Math.exp(-t))
/** The BCCG distribution, whose quantile function gives the centiles and draws the measurements. */
const BCCG = boxCoxColeGreenDistributional()

/** The law of `growthChart`: BCCG parameter curves over age in years, and the centiles they imply. */
export const GROWTH_TRUTH = {
  /** The range of ages, in years, over which the curves are defined. */
  ages: [0, 18] as const,
  /** $\mu(a)$: close to the median. */
  mu: (a: number) => 3.4 + 8.6 * (1 - Math.exp(-a)) + 2.1 * a + 22 * logistic((a - 13) / 1.5),
  /** $\sigma(a)$: close to the coefficient of variation. */
  sigma: (a: number) => 0.1 + 0.08 * logistic((a - 9) / 2),
  /** $\nu(a)$: the Box–Cox power; 1 is symmetric, below 1 right-skewed. */
  nu: (a: number) => 1 - 2 * logistic((a - 9) / 2.5),
  /** The quantile at level $p \in (0, 1)$ (the $100p$th centile) at age $a$ in years. */
  centile(a: number, p: number): number {
    return BCCG.quantile(p, [this.mu(a), this.sigma(a), this.nu(a)])
  },
}

/** Options of `growthChart`. */
export interface GrowthChartOptions {
  /** Rows. Default 1000. */
  n?: number
  /** The largest age (years); ages are uniform from 0 to `maxAge`, which must be in $(0, 18]$. Default 18. */
  maxAge?: number
}

/**
 * The growth-chart data set (see the file comment): ages uniform from 0 to `maxAge`, sorted, and each measurement
 * drawn by inverting the BCCG distribution function at a uniform level. Throws `DomainError` when $n$ is not a
 * non-negative integer or `maxAge` is not in $(0, 18]$.
 *
 * @param s The stream the ages (child `'age'`) and the levels of the measurements (child `'y'`) are drawn from.
 * @param options The number of rows and the largest age.
 * @returns A regression dataset: `x` the ages ($n \times 1$, sorted), `y` the measurements, and `f` the true median at
 *   each age.
 *
 * @example Half the measurements lie below the true median
 * const d = growthChart(stream(1), { n: 1000 })
 * const age = toArray(d.x).map((r) => r[0])
 * const y = toArray(d.y)
 * print('x:', d.x.shape, ' y:', d.y.shape)
 * print('first rows:', age.slice(0, 3), y.slice(0, 3))
 * const median = toArray(d.f)
 * print('share below the median:', y.filter((v, i) => v < median[i]).length / y.length)
 * print('share below the 90th centile:', y.filter((v, i) => v < GROWTH_TRUTH.centile(age[i], 0.9)).length / y.length)
 */
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
