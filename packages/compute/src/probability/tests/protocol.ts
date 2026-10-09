/**
 * The test protocol of `aifn-compute/probability/tests`: what every hypothesis test returns, how a p-value is read from
 * a null law, the rejection region at a level $\alpha$, confidence intervals from a pivot, and null laws that no
 * registered family covers (built with the distributions module's `univariate` builder).
 *
 * A test reduces its data to a statistic $T$ whose law under $H_0$ is known, and reports the probability under that
 * law of a value at least as extreme as the observed $t$. Which values count as more extreme is the test's `Tail`
 * rule, so one function, `pValueOf`, reads every test's p-value, and `rejectionRegion` inverts it. Every test rejects
 * at level $\alpha$ exactly when $p \le \alpha$. The input checks shared by the tests (`sample`, `checkLevel`) are
 * here too: they throw `DomainError` rather than drop or clamp a bad value.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { units } from 'aifn-compute/foundation/random'
import { dense, fromData, isTensor, toFlat, unwrap, type Value, type VectorLike } from 'aifn-compute/foundation/tensor'
import { univariate, type Support, type Univariate } from 'aifn-compute/probability/distributions'

/**
 * The alternative hypothesis: the parameter differs from (`two-sided`), is below (`less`) or is above (`greater`) its
 * value under the null.
 */
export type Alternative = 'two-sided' | 'less' | 'greater'

/**
 * How a p-value is read from the null law of the statistic $T$ at the observed $t$:
 *
 * - `upper`: $\pr(T \ge t)$ ($\chi^2$, $F$, a two-sided Kolmogorov–Smirnov distance: large values are evidence
 *   against the null);
 * - `lower`: $\pr(T \le t)$ (Shapiro–Wilk's $W$: small values are evidence);
 * - `both`: $\min(1, 2\min(\pr(T \le t), \pr(T \ge t)))$, the doubled smaller tail ($t$ and $z$ tests, rank tests);
 * - `likelihood`: the total null probability of outcomes no more likely than $t$ (exact binomial and Fisher tests, as
 *   scipy's two-sided `binomtest` and `fisher_exact`).
 */
export type Tail = 'upper' | 'lower' | 'both' | 'likelihood'

/**
 * A confidence interval: `lower` and `upper` are its ends and `level` its confidence level, such as 0.95. A one-sided
 * interval has an infinite end (or the end of the parameter's range, 0 or 1 for a proportion).
 */
export type Interval = { readonly lower: number; readonly upper: number; readonly level: number }

/**
 * An effect size: a scale-free measure of how far the data are from the null. `name` says which measure it is
 * (`Cohen's d`, `odds ratio (conditional MLE)`), `value` is its estimate and `ci`, when given, its confidence interval.
 */
export type EffectSize = { readonly name: string; readonly value: number; readonly ci?: Interval }

/**
 * The result of a hypothesis test. `statistic` follows `null` under the null hypothesis, and `pValue` is the `tail`
 * probability of `null` at `statistic` (up to a continuity correction, which `method` names). `estimate` and `ci` are
 * for the parameter named by `estimand` (a mean, a difference of means, a proportion, an odds ratio); `n` is the
 * number of observations used (after dropping zero differences, for example).
 */
export type TestResult = {
  /** Always `'test-result'`, so that a result can be told from other objects. */
  readonly kind: 'test-result'
  /** The registry key of the test (`welchTTest`). */
  readonly test: string
  /** A display name, with the variant used (`Welch two-sample t-test`, `Mann–Whitney U (exact)`). */
  readonly method: string
  /** The observed value of the test statistic. */
  readonly statistic: number
  /** The statistic's symbol in TeX (`t`, `z`, `\\chi^2`, `U`). */
  readonly symbol: string
  /** Degrees of freedom: one number, or [numerator, denominator] for an $F$ statistic. */
  readonly df?: number | readonly [number, number]
  /** The p-value, in $[0, 1]$ (NaN when the statistic is). */
  readonly pValue: number
  /** The alternative hypothesis tested. */
  readonly alternative: Alternative
  /** The rule by which `pValue` is read from `null`. */
  readonly tail: Tail
  /** The law of the statistic under the null hypothesis. */
  readonly null: Univariate
  /** What `estimate` and `ci` are about (`mean`, `difference of means`, `proportion`, `odds ratio`). */
  readonly estimand?: string
  /** The point estimate of the estimand. */
  readonly estimate?: number
  /** The estimand's value under the null hypothesis ($\mu_0$, $p_0$, an odds ratio of 1). */
  readonly nullValue?: number
  /** The confidence interval for the estimand, one-sided when the alternative is. */
  readonly ci?: Interval
  /** The effect size the test reports, if any. */
  readonly effectSize?: EffectSize
  /** The number of observations used. */
  readonly n: number
}

/**
 * Options shared by the tests: `alternative`, the alternative hypothesis (default `two-sided`), and `level`, the
 * confidence level of the interval reported (default 0.95).
 */
export type TestOptions = { alternative?: Alternative; level?: number }

/**
 * The protocol's object for one test, with `kind` set.
 *
 * @param r Every field of the result but `kind`.
 * @returns `r` with `kind: 'test-result'`.
 */
export const result = (r: Omit<TestResult, 'kind'>): TestResult => ({ kind: 'test-result', ...r })

// ── Reading the null law ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A distribution method's value as a number: the tests call them on numbers, which come back as numbers.
 *
 * @param v The value a method of a `Univariate` returned for a number.
 * @returns It as a number.
 */
const num = (v: Value): number => unwrap(v) as number

/**
 * $\pr(T \ge t)$ under a null law: its survival function at $t$ for a continuous law, and just below $t$ for an
 * integer-valued one, so that the observed value counts. A non-integer $t$ of a discrete law is rounded up (within
 * $10^{-9}$).
 *
 * @param law The null law of the statistic $T$.
 * @param t The observed value of the statistic.
 * @returns The upper-tail probability, including the mass at $t$.
 *
 * @example The upper tail of a fair coin's heads, and of the standard normal
 * const coin = binomialTest(0, 10).null // Binomial(10, 1/2), the null law of 10 tosses
 * print('P(T >= 8) =', upperTail(coin, 8))
 * const z = zTest([0], { sigma: 1 }).null // the standard normal
 * print('P(Z >= 1.96) =', upperTail(z, 1.96))
 */
export function upperTail(law: Univariate, t: number): number {
  if (!law.discrete) return num(law.survival(t))
  return num(law.survival(Math.ceil(t - 1e-9) - 1))
}

/**
 * $\pr(T \le t)$ under a null law: its cdf at $t$, with $t$ rounded down (within $10^{-9}$) for a discrete law.
 *
 * @param law The null law of the statistic $T$.
 * @param t The observed value of the statistic.
 * @returns The lower-tail probability, including the mass at $t$.
 *
 * @example The lower tail of a fair coin's heads, and of the standard normal
 * const coin = binomialTest(0, 10).null
 * print('P(T <= 2) =', lowerTail(coin, 2))
 * const z = zTest([0], { sigma: 1 }).null
 * print('P(Z <= -1.96) =', lowerTail(z, -1.96))
 */
export function lowerTail(law: Univariate, t: number): number {
  return num(law.cdf(law.discrete ? Math.floor(t + 1e-9) : t))
}

/**
 * The integers a discrete law puts mass on, between its $10^{-14}$ and $1 - 10^{-14}$ quantiles (at most $10^6$ of
 * them), which `pValueOf` and `rejectionRegion` enumerate.
 *
 * @param law A discrete law on the integers.
 * @returns The integers from its lower to its upper quantile, ascending.
 *
 * @example The supports of two exact null laws
 * print('Binomial(6, 1/2):', supportValues(binomialTest(0, 6).null))
 * // Fisher's test of a table with first row total 5 and first column total 4, out of 9.
 * print('Hypergeometric:', supportValues(fisherExact([[3, 2], [1, 3]]).null))
 */
export function supportValues(law: Univariate): number[] {
  const lo = Math.max(num(law.quantile(1e-14)), -1e6)
  const hi = Math.min(num(law.isf(1e-14)), lo + 1e6)
  const out: number[] = []
  for (let k = Math.ceil(lo); k <= hi; k++) out.push(k)
  return out
}

/**
 * The p-value of an observed statistic under a null law, by the rule `tail`. `likelihood` sums the mass of every
 * outcome whose probability is at most the observed one's times $1 + 10^{-7}$, the relative tolerance scipy uses so
 * that outcomes equally likely in exact arithmetic count. Throws `DomainError` for `likelihood` on a continuous law.
 *
 * @param law The null law of the statistic.
 * @param t The observed value of the statistic.
 * @param tail How the p-value is read from `law` (see `Tail`).
 * @returns The p-value, clamped to $[0, 1]$.
 *
 * @example The three two-sided rules can differ for a skewed law
 * // 6 successes in 10 trials when p = 0.3; scipy's two-sided binomtest(6, 10, 0.3) uses the likelihood rule.
 * const law = binomialTest(0, 10, { p: 0.3 }).null
 * print('upper:', pValueOf(law, 6, 'upper'))
 * print('both (doubled tail):', pValueOf(law, 6, 'both'))
 * print('likelihood:', pValueOf(law, 6, 'likelihood'))
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

/**
 * The tail rule of a test for an alternative: `both` when two-sided, else the side the alternative points to.
 *
 * @param alternative The alternative hypothesis.
 * @returns `both`, `lower` (for `less`) or `upper` (for `greater`).
 *
 * @example The rule for each alternative
 * print('two-sided:', tailOf('two-sided'), ' less:', tailOf('less'), ' greater:', tailOf('greater'))
 */
export const tailOf = (alternative: Alternative): Tail =>
  alternative === 'two-sided' ? 'both' : alternative === 'greater' ? 'upper' : 'lower'

/**
 * The rejection region of a test at level $\alpha$: the statistic values whose p-value is at most $\alpha$, as
 * disjoint closed intervals $[l, u]$ (infinite ends where the region is unbounded). Continuous laws use their
 * quantiles; discrete ones enumerate the support, so the region of an exact test has size at most $\alpha$ (usually
 * less). Throws `DomainError` when $\alpha$ is not in $(0, 1)$, or for `likelihood` on a continuous law.
 *
 * @param test A test result, or anything with its `null` law and `tail` rule.
 * @param alpha The significance level $\alpha$, in $(0, 1)$.
 * @returns The region as `[lower, upper]` pairs, ascending.
 *
 * @example The critical values of a t-test, and the region of an exact test
 * const r = oneSampleTTest([5.1, 4.9, 5.6, 5.8, 6.0, 5.3], { mu: 5 })
 * print('t =', r.statistic, ' region at 0.05:', rejectionRegion(r, 0.05))
 * // Ten tosses of a fair coin: 0, 1, 9 or 10 heads, whose total probability is 0.021.
 * print('exact binomial test, n = 10:', rejectionRegion(binomialTest(5, 10), 0.05))
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

/**
 * True when the statistic falls in the rejection region at level $\alpha$, which for every rule is $p \le \alpha$.
 *
 * @param test The test result.
 * @param alpha The significance level $\alpha$.
 * @returns Whether `test.pValue` is at most $\alpha$.
 *
 * @example A p-value of 0.048 rejects at 5% but not at 1%
 * const r = oneSampleTTest([5.1, 4.9, 5.6, 5.8, 6.0, 5.3], { mu: 5 })
 * print('p =', r.pValue)
 * print('rejects at 0.05:', rejects(r, 0.05), ' at 0.01:', rejects(r, 0.01))
 */
export const rejects = (test: TestResult, alpha: number): boolean => test.pValue <= alpha

// ── Intervals from a pivot ───────────────────────────────────────────────────────────────────────────────────────────

/**
 * The interval $\hat\theta \pm q \cdot \mathrm{se}$ for a pivot $(\hat\theta - \theta)/\mathrm{se}$ following
 * `pivot` (a standard normal or Student's $t$): two-sided with $q$ its $1 - (1 - \text{level})/2$ quantile; for `less`,
 * $(-\infty, \hat\theta + q \cdot \mathrm{se}]$ with $q$ at `level`; for `greater`,
 * $[\hat\theta - q \cdot \mathrm{se}, \infty)$. The side follows the alternative, as scipy's `confidence_interval`.
 * Throws `DomainError` when `level` is not in $(0, 1)$.
 *
 * @param estimate The point estimate $\hat\theta$.
 * @param se Its standard error.
 * @param pivot The law of the pivot, symmetric about 0.
 * @param level The confidence level, in $(0, 1)$.
 * @param alternative Which interval: two-sided, an upper bound (`less`) or a lower bound (`greater`).
 * @returns The interval.
 *
 * @example A 95% interval from an estimate of 10 with standard error 2
 * const z = zTest([0], { sigma: 1 }).null // the standard normal
 * print('two-sided:', pivotInterval(10, 2, z, 0.95))
 * print('greater:', pivotInterval(10, 2, z, 0.95, 'greater'))
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

/**
 * Check a confidence level: throws `DomainError` unless it is in $(0, 1)$.
 *
 * @param level The confidence level.
 * @param where The caller's name, for the error message.
 *
 * @example A level of 95 (rather than 0.95) is refused
 * checkLevel(0.95, 'demo')
 * try {
 *   checkLevel(95, 'demo')
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function checkLevel(level: number, where: string): void {
  if (!(level > 0 && level < 1)) throw new DomainError(where, `${where}: the confidence level must be in (0, 1)`)
}

// ── Inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A sample as a Float64Array, with at least `min` finite values. NaN is an error, never dropped silently: throws
 * `DomainError` for too few values or a non-finite one.
 *
 * @param x The sample: an array or a tensor of numbers.
 * @param where The caller's name, for the error messages.
 * @param min The fewest values accepted.
 * @returns A new Float64Array of the values, which the caller may sort or modify.
 */
export function sample(x: VectorLike, where: string, min = 1): Float64Array {
  const v = dense.toF64(x, where)
  if (v.length < min) throw new DomainError(where, `${where}: needs at least ${min} values, got ${v.length}`)
  for (const a of v)
    if (!Number.isFinite(a)) throw new DomainError(where, `${where}: the sample has a non-finite value`)
  return v
}

/**
 * The mean and the sample variance (divided by $n - 1$) of a sample, by two passes.
 *
 * @param v The sample, not modified.
 * @returns `mean` and `variance` (NaN for a single value).
 */
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

/**
 * Apply a scalar function elementwise to a number or a tensor. Traced values are not differentiable here: they throw
 * `DomainError`.
 *
 * @param x A number or a tensor (a traced value is unwrapped first).
 * @param f The scalar function.
 * @param where The law's name, for the error message.
 * @returns `f` of `x`, a number for a number and a tensor of the same shape for a tensor.
 */
function elementwiseOf(x: Value, f: (v: number) => number, where: string): Value {
  const r = unwrap(x)
  if (typeof r === 'number') return f(r)
  if (!isTensor(r)) throw new DomainError(where, `${where}: not differentiable`)
  return fromData(Float64Array.from(toFlat(r), f), r.shape)
}

/**
 * A continuous null law on $[l, u]$ given by its cdf (and survival function, for accuracy in the upper tail), for the
 * statistics whose null laws are not families (Kolmogorov's $D_n$, a Bonferroni bound on a maximum). The density is
 * the central difference of the cdf; mean and variance are integrals of the survival function by Simpson's rule on a
 * finite support (they throw on an infinite one); entropy and mode throw. Quantiles invert the cdf numerically, and
 * draws invert uniforms through them. The density's step is relative to the support's width, so the support should be
 * finite.
 *
 * @param spec The law: `name` and `params` (shown when the law is printed), the support's ends `lower` and `upper`,
 *   its `cdf`, and optionally its `survival` function (default $1 - \text{cdf}$). Both are called only inside the
 *   support; outside it the law returns 0 and 1 itself.
 * @returns The law, as a `Univariate`.
 *
 * @example The law of the larger of two uniforms
 * const law = continuousLaw({ name: 'MaxOfTwoUniforms', params: {}, lower: 0, upper: 1, cdf: (x) => x * x })
 * print('mean =', law.mean(), ' median =', law.quantile(0.5))
 * print('P(T >= 0.9) =', pValueOf(law, 0.9, 'upper'))
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
