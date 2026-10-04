/**
 * The test protocol of `aifn-compute/probability/tests`: what every hypothesis test returns, how a p-value is read from a null
 * law, the rejection region at a level α, confidence intervals from a pivot, and null laws that no registered family
 * covers (built with the distributions module's `univariate` builder).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { units } from 'aifn-compute/foundation/random'
import { dense, fromData, isTensor, toFlat, unwrap, type Value, type VectorLike } from 'aifn-compute/foundation/tensor'
import { univariate, type Support, type Univariate } from 'aifn-compute/probability/distributions'

/** The alternative hypothesis: the parameter differs from (two-sided), is below (less) or is above (greater) the null. */
export type Alternative = 'two-sided' | 'less' | 'greater'

/**
 * How a p-value is read from the null law of the statistic T at the observed t:
 *
 * - `upper`: P(T ≥ t) (χ², F, a two-sided Kolmogorov–Smirnov distance: large values are evidence against the null);
 * - `lower`: P(T ≤ t) (Shapiro–Wilk's W: small values are evidence);
 * - `both`: min(1, 2 min(P(T ≤ t), P(T ≥ t))), the doubled smaller tail (t and z tests, rank tests);
 * - `likelihood`: the total null probability of outcomes no more likely than t (exact binomial and Fisher tests, as
 *   scipy's two-sided `binomtest` and `fisher_exact`).
 */
export type Tail = 'upper' | 'lower' | 'both' | 'likelihood'

/** A confidence interval with its level (one-sided intervals have an infinite end). */
export type Interval = { readonly lower: number; readonly upper: number; readonly level: number }

/** An effect size: a scale-free measure of how far the data are from the null, optionally with an interval. */
export type EffectSize = { readonly name: string; readonly value: number; readonly ci?: Interval }

/**
 * The result of a hypothesis test. `statistic` follows `null` under the null hypothesis, and `pValue` is the `tail`
 * probability of `null` at `statistic` (up to a continuity correction, which `method` names). `estimate` and `ci` are
 * for the parameter named by `estimand` (a mean, a difference of means, a proportion, an odds ratio); `n` is the
 * number of observations used (after dropping zero differences, for example).
 */
export type TestResult = {
  readonly kind: 'test-result'
  /** The registry key of the test (`welchTTest`). */
  readonly test: string
  /** A display name, with the variant used (`Welch two-sample t-test`, `Mann–Whitney U (exact)`). */
  readonly method: string
  readonly statistic: number
  /** The statistic's symbol in TeX (`t`, `z`, `\\chi^2`, `U`). */
  readonly symbol: string
  /** Degrees of freedom: one number, or [numerator, denominator] for an F statistic. */
  readonly df?: number | readonly [number, number]
  readonly pValue: number
  readonly alternative: Alternative
  readonly tail: Tail
  /** The law of the statistic under the null hypothesis. */
  readonly null: Univariate
  readonly estimand?: string
  readonly estimate?: number
  /** The estimand's value under the null hypothesis (μ₀, p₀, an odds ratio of 1). */
  readonly nullValue?: number
  readonly ci?: Interval
  readonly effectSize?: EffectSize
  readonly n: number
}

/** Options shared by the tests: the alternative and the confidence level of the interval. */
export type TestOptions = { alternative?: Alternative; level?: number }

/** The protocol's object for one test, with `kind` set. */
export const result = (r: Omit<TestResult, 'kind'>): TestResult => ({ kind: 'test-result', ...r })

// ── Reading the null law ─────────────────────────────────────────────────────────────────────────────────────────────

const num = (v: Value): number => unwrap(v) as number

/** P(T ≥ t): the survival function just below t for an integer-valued law, so the observed value counts. */
export function upperTail(law: Univariate, t: number): number {
  if (!law.discrete) return num(law.survival(t))
  return num(law.survival(Math.ceil(t - 1e-9) - 1))
}

/** P(T ≤ t). */
export function lowerTail(law: Univariate, t: number): number {
  return num(law.cdf(law.discrete ? Math.floor(t + 1e-9) : t))
}

/** The integers a discrete law puts mass on, between its 10⁻¹⁴ and 1 − 10⁻¹⁴ quantiles (at most 10⁶). */
export function supportValues(law: Univariate): number[] {
  const lo = Math.max(num(law.quantile(1e-14)), -1e6)
  const hi = Math.min(num(law.isf(1e-14)), lo + 1e6)
  const out: number[] = []
  for (let k = Math.ceil(lo); k <= hi; k++) out.push(k)
  return out
}

/**
 * The p-value of an observed statistic under a null law, by the rule `tail`. `likelihood` sums the mass of every
 * outcome whose probability is at most the observed one's × (1 + 10⁻⁷), the relative tolerance scipy uses so that
 * outcomes equally likely in exact arithmetic count.
 */
export function pValueOf(law: Univariate, t: number, tail: Tail): number {
  let p: number
  if (tail === 'upper') p = upperTail(law, t)
  else if (tail === 'lower') p = lowerTail(law, t)
  else if (tail === 'both') p = 2 * Math.min(upperTail(law, t), lowerTail(law, t))
  else {
    if (!law.discrete) throw new DomainError('pValueOf', 'pValueOf: the likelihood rule needs a discrete null law')
    const d = num(law.prob(t)) * (1 + 1e-7)
    p = 0
    for (const k of supportValues(law)) {
      const q = num(law.prob(k))
      if (q <= d) p += q
    }
  }
  return Math.min(1, Math.max(0, p))
}

/** The tail rule of a test for an alternative: `both` when two-sided, else the side the alternative points to. */
export const tailOf = (alternative: Alternative): Tail =>
  alternative === 'two-sided' ? 'both' : alternative === 'greater' ? 'upper' : 'lower'

/**
 * The rejection region of a test at level α: the statistic values whose p-value is at most α, as disjoint closed
 * intervals [lower, upper] (infinite ends where the region is unbounded). Continuous laws use their quantiles;
 * discrete ones enumerate the support, so the region of an exact test has size at most α (usually less).
 */
export function rejectionRegion(test: Pick<TestResult, 'null' | 'tail'>, alpha: number): [number, number][] {
  const law = test.null
  if (!(alpha > 0 && alpha < 1)) throw new DomainError('rejectionRegion', 'rejectionRegion: α must be in (0, 1)')
  if (!law.discrete) {
    if (test.tail === 'upper') return [[num(law.isf(alpha)), Infinity]]
    if (test.tail === 'lower') return [[-Infinity, num(law.quantile(alpha))]]
    if (test.tail === 'both')
      return [
        [-Infinity, num(law.quantile(alpha / 2))],
        [num(law.isf(alpha / 2)), Infinity],
      ]
    throw new DomainError('rejectionRegion', 'rejectionRegion: the likelihood rule needs a discrete null law')
  }
  const out: [number, number][] = []
  for (const k of supportValues(law)) {
    if (pValueOf(law, k, test.tail) > alpha) continue
    const last = out[out.length - 1]
    if (last && last[1] === k - 1) last[1] = k
    else out.push([k, k])
  }
  return out
}

/** True when the statistic falls in the rejection region at level α, which for every rule is p ≤ α. */
export const rejects = (test: TestResult, alpha: number): boolean => test.pValue <= alpha

// ── Intervals from a pivot ───────────────────────────────────────────────────────────────────────────────────────────

/**
 * The interval estimate ± q · se for a pivot (estimate − θ)/se following `pivot` (a standard normal or Student t):
 * two-sided with q its 1 − (1 − level)/2 quantile; for `less`, (−∞, estimate + q · se] with q at `level`; for
 * `greater`, [estimate − q · se, ∞). The side follows the alternative, as scipy's `confidence_interval`.
 */
export function pivotInterval(
  estimate: number,
  se: number,
  pivot: Univariate,
  level: number,
  alternative: Alternative = 'two-sided',
): Interval {
  checkLevel(level, 'pivotInterval')
  if (alternative === 'two-sided') {
    const q = num(pivot.isf((1 - level) / 2))
    return { lower: estimate - q * se, upper: estimate + q * se, level }
  }
  const q = num(pivot.isf(1 - level))
  return alternative === 'less'
    ? { lower: -Infinity, upper: estimate + q * se, level }
    : { lower: estimate - q * se, upper: Infinity, level }
}

export function checkLevel(level: number, where: string): void {
  if (!(level > 0 && level < 1)) throw new DomainError(where, `${where}: the confidence level must be in (0, 1)`)
}

// ── Inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A sample as a Float64Array, with at least `min` finite values. NaN is an error, never dropped silently. */
export function sample(x: VectorLike, where: string, min = 1): Float64Array {
  const v = dense.toF64(x, where)
  if (v.length < min) throw new DomainError(where, `${where}: needs at least ${min} values, got ${v.length}`)
  for (const a of v)
    if (!Number.isFinite(a)) throw new DomainError(where, `${where}: the sample has a non-finite value`)
  return v
}

/** The mean and the sample variance (÷ (n − 1)) of a sample, by two passes. */
export function meanAndVariance(v: Float64Array): { mean: number; variance: number } {
  const n = v.length
  let m = 0
  for (const a of v) m += a
  m /= n
  let ss = 0
  for (const a of v) ss += (a - m) ** 2
  return { mean: m, variance: n > 1 ? ss / (n - 1) : NaN }
}

// ── Null laws that are not registered families ──────────────────────────────────────────────────────────────────────

/** Apply a scalar function elementwise to a number or a tensor (traced values are not differentiable here). */
function elementwiseOf(x: Value, f: (v: number) => number, where: string): Value {
  const r = unwrap(x)
  if (typeof r === 'number') return f(r)
  if (!isTensor(r)) throw new DomainError(where, `${where}: not differentiable`)
  return fromData(Float64Array.from(toFlat(r), f), r.shape)
}

/**
 * A continuous null law on [lower, upper] given by its cdf (and survival function, for accuracy in the upper tail),
 * for the statistics whose null laws are not families (Kolmogorov's Dₙ, a Bonferroni bound on a maximum). The density
 * is the central difference of the cdf; mean and variance are integrals of the survival function by Simpson's rule
 * on a finite support; entropy and mode throw. Quantiles invert the cdf numerically, and draws
 * invert uniforms through them.
 */
export function continuousLaw(spec: {
  name: string
  params: Record<string, number>
  lower: number
  upper: number
  cdf: (x: number) => number
  survival?: (x: number) => number
}): Univariate {
  const { lower, upper } = spec
  const cdf = (x: number) => (x <= lower ? 0 : x >= upper ? 1 : spec.cdf(x))
  const sf = (x: number) => (x <= lower ? 1 : x >= upper ? 0 : spec.survival ? spec.survival(x) : 1 - spec.cdf(x))
  const density = (x: number) => {
    if (!(x > lower && x < upper)) return 0
    const h = 1e-6 * Math.max(1, Math.abs(x), upper - lower)
    const a = Math.max(lower, x - h)
    const b = Math.min(upper, x + h)
    return Math.max(0, (cdf(b) - cdf(a)) / (b - a))
  }
  const moment = (k: 1 | 2) => {
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) throw new DomainError(spec.name, `${spec.name}: no moments`)
    // E[Xᵏ] = lowerᵏ + ∫ k xᵏ⁻¹ P(X > x) dx over the support.
    const m = 2000
    const h = (upper - lower) / m
    let s = 0
    for (let i = 0; i <= m; i++) {
      const x = lower + i * h
      const w = i === 0 || i === m ? 1 : i % 2 === 1 ? 4 : 2
      s += w * k * x ** (k - 1) * sf(x)
    }
    return lower ** k + (s * h) / 3
  }
  const where = spec.name
  const support: Support = { type: 'interval', lower, upper }
  const self: Univariate = univariate({
    name: spec.name,
    params: spec.params,
    batchShape: [],
    support,
    logProb: (x) => elementwiseOf(x, (v) => Math.log(density(v)), where),
    cdf: (x) => elementwiseOf(x, cdf, where),
    survival: (x) => elementwiseOf(x, sf, where),
    // Inverse-transform draws through the numerically inverted cdf (slow: one bisection per draw).
    sample: (s, shape) => {
      const u = units(
        s,
        shape.reduce((a, b) => a * b, 1),
      )
      return fromData(
        Float64Array.from(u, (p) => num(self.quantile(p))),
        shape,
      )
    },
    mean: () => moment(1),
    variance: () => moment(2) - moment(1) ** 2,
    entropy: () => {
      throw new DomainError(where, `${where}: no closed-form entropy`)
    },
    mode: () => {
      throw new DomainError(where, `${where}: no closed-form mode`)
    },
  })
  return self
}
