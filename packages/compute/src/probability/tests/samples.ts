/**
 * Tests on one series or several groups: the Ljung–Box (and Box–Pierce) portmanteau test of autocorrelation,
 * Grubbs' test for one outlier, and the one-way analysis of variance (scipy's `f_oneway`). Each has a large-is-extreme
 * statistic read from the upper tail of its null law ($\chi^2$, a Bonferroni bound, $F$), and each assumes normal
 * data.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { toFlat, type VectorLike } from 'aifn-compute/foundation/tensor'
import { ChiSquare, FisherSnedecor, StudentT } from 'aifn-compute/probability/distributions'
import { autocorrelation } from 'aifn-compute/probability/stats'
import { continuousLaw, meanAndVariance, pValueOf, result, sample, type TestResult } from './protocol'

/**
 * The Ljung–Box test (Ljung and Box, 1978) of $H_0$: the first $h$ autocorrelations of a series of length $n$ are
 * zero: $Q = n(n + 2)\sum_{k=1}^{h} \hat\rho_k^2/(n - k)$, approximately $\chi^2(h - \text{fitted})$ under the
 * null, where `fitted` counts the ARMA parameters estimated when the series are residuals. `boxPierce` uses
 * $Q = n\sum_{k=1}^{h} \hat\rho_k^2$ instead (Box and Pierce, 1970). $\hat\rho_k$ is the biased sample
 * autocorrelation (`aifn-compute/probability/stats`'s `autocorrelation`). As statsmodels' `acorr_ljungbox`.
 *
 * @param x The series, at least 3 finite values in time order.
 * @param options The test.
 * @param options.lags The number of lags $h$, an integer in $[1, n - 1]$.
 * @param options.fitted The number of parameters fitted to produce the series (0 for raw data); must be below $h$.
 * @param options.boxPierce Use the Box–Pierce statistic rather than Ljung–Box's.
 * @returns The test result, with $Q$ and its $h - \text{fitted}$ degrees of freedom.
 *
 * @example A series that flips sign each step, and white noise
 * const flips = [1.2, -0.8, 1.1, -1.3, 0.9, -1.0, 1.4, -0.7, 1.0, -1.2, 0.8, -0.9]
 * const r = ljungBox(flips, { lags: 3 })
 * print('Ljung-Box: Q =', r.statistic, ' p =', r.pValue)
 * print('Box-Pierce: Q =', ljungBox(flips, { lags: 3, boxPierce: true }).statistic)
 * const noise = normals(stream(1), 100)
 * print('white noise: p =', ljungBox(noise, { lags: 10 }).pValue)
 */
export function ljungBox(
  x: VectorLike,
  { lags, fitted = 0, boxPierce = false }: { lags: number; fitted?: number; boxPierce?: boolean },
): TestResult {
  const v = sample(x, 'ljungBox', 3)
  const n = v.length
  if (!(Number.isInteger(lags) && lags >= 1 && lags < n))
    throw new DomainError('ljungBox', 'ljungBox: lags must be an integer in [1, n − 1]')
  const df = lags - fitted
  if (!(df >= 1)) throw new DomainError('ljungBox', 'ljungBox: needs lags > fitted')
  const rho = toFlat(autocorrelation(v, { maxLag: lags }))
  let q = 0
  for (let k = 1; k <= lags; k++) q += boxPierce ? rho[k] ** 2 : rho[k] ** 2 / (n - k)
  const statistic = boxPierce ? n * q : n * (n + 2) * q
  const law = ChiSquare(df)
  return result({
    test: 'ljungBox',
    method: boxPierce ? 'Box–Pierce test' : 'Ljung–Box test',
    statistic,
    symbol: 'Q',
    df,
    pValue: pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    n,
  })
}

/**
 * The result of Grubbs' test: the protocol's fields and the suspected outlier, its position `index` in the input and
 * its `value`.
 */
export type GrubbsTest = TestResult & { index: number; value: number }

/**
 * Grubbs' test (Grubbs, 1950) for one outlier in a normal sample of size $n \ge 3$:
 * $G = \max_i \lvert x_i - \bar x \rvert/s$ (`two-sided`), or $(\max_i x_i - \bar x)/s$ (`greater`, the largest
 * value) or $(\bar x - \min_i x_i)/s$ (`less`, the smallest). With $t = \sqrt{n(n - 2)G^2/((n - 1)^2 - nG^2)}$,
 * $\pr(G \ge g) \approx c\, n \pr(T_{n-2} \ge t)$ ($c = 2$ two-sided, 1 one-sided), the Bonferroni bound that gives
 * Grubbs' critical values; the null law is that bound, capped at 1, on $[0, (n - 1)/\sqrt n]$. Throws `DomainError`
 * when every value is equal.
 *
 * @param x The sample, at least 3 finite values.
 * @param options Options of the test.
 * @param options.alternative Which extreme to test: either (`two-sided`), the largest (`greater`) or the smallest
 *   (`less`).
 * @returns The test result, with the suspected outlier's index and value.
 *
 * @example One value far above the rest
 * const r = grubbs([10.2, 9.8, 10.1, 10.4, 9.9, 10.0, 12.9, 10.3])
 * print('G =', r.statistic, ' p =', r.pValue)
 * print('outlier:', r.value, ' at index', r.index)
 *
 * @example The smallest value of the same sample is not an outlier
 * const r = grubbs([10.2, 9.8, 10.1, 10.4, 9.9, 10.0, 12.9, 10.3], { alternative: 'less' })
 * print('G =', r.statistic, ' p =', r.pValue, ' value:', r.value)
 */
export function grubbs(
  x: VectorLike,
  { alternative = 'two-sided' }: { alternative?: TestResult['alternative'] } = {},
): GrubbsTest {
  const v = sample(x, 'grubbs', 3)
  const n = v.length
  const { mean, variance } = meanAndVariance(v)
  const s = Math.sqrt(variance)
  if (!(s > 0)) throw new DomainError('grubbs', 'grubbs: every value is equal')
  let index = 0
  let best = -Infinity
  v.forEach((t, i) => {
    const g = alternative === 'greater' ? t - mean : alternative === 'less' ? mean - t : Math.abs(t - mean)
    if (g > best) {
      best = g
      index = i
    }
  })
  const G = best / s
  const sides = alternative === 'two-sided' ? 2 : 1
  const gMax = (n - 1) / Math.sqrt(n)
  const tOf = (g: number) => Math.sqrt((n * (n - 2) * g * g) / Math.max((n - 1) ** 2 - n * g * g, 1e-300))
  const T = StudentT(n - 2)
  const sf = (g: number) => Math.min(1, sides * n * (T.survival(tOf(g)) as number))
  const law = continuousLaw({
    name: 'GrubbsBonferroni',
    params: { n, sides },
    lower: 0,
    upper: gMax,
    cdf: (g) => 1 - sf(g),
    survival: sf,
  })
  return {
    ...result({
      test: 'grubbs',
      method: "Grubbs' test for one outlier",
      statistic: G,
      symbol: 'G',
      pValue: pValueOf(law, G, 'upper'),
      alternative,
      tail: 'upper',
      null: law,
      n,
    }),
    index,
    value: v[index],
  }
}

/**
 * The one-way analysis of variance of $k \ge 2$ groups (Fisher):
 * $F = (\mathrm{SS}_\text{between}/(k - 1))/(\mathrm{SS}_\text{within}/(N - k))$, which follows $F(k - 1, N - k)$
 * when every group is normal with one common mean and variance. The effect size is
 * $\eta^2 = \mathrm{SS}_\text{between}/\mathrm{SS}_\text{total}$. Throws `DomainError` for fewer than two groups or
 * no more values than groups.
 *
 * @param groups The groups' samples, each at least one finite value.
 * @returns The test result, with $F$ and its degrees of freedom $[k - 1, N - k]$.
 *
 * @example Three groups whose means differ, and three whose means do not
 * const r = oneWayAnova([[4.2, 4.8, 5.1, 4.5], [5.9, 6.3, 5.6, 6.1], [4.9, 5.3, 5.0, 5.6]])
 * print('F =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print(r.effectSize)
 * const same = oneWayAnova([[4.2, 4.8, 5.1, 4.5], [4.6, 4.4, 5.0, 4.9], [4.9, 4.3, 5.0, 4.6]])
 * print('F =', same.statistic, ' p =', same.pValue)
 */
export function oneWayAnova(groups: readonly VectorLike[]): TestResult {
  if (groups.length < 2) throw new DomainError('oneWayAnova', 'oneWayAnova: needs at least two groups')
  const gs = groups.map((g) => sample(g, 'oneWayAnova', 1))
  const N = gs.reduce((s, g) => s + g.length, 0)
  const k = gs.length
  if (N <= k) throw new DomainError('oneWayAnova', 'oneWayAnova: needs more values than groups')
  let grand = 0
  for (const g of gs) for (const v of g) grand += v
  grand /= N
  let between = 0
  let within = 0
  for (const g of gs) {
    const m = g.reduce((s, v) => s + v, 0) / g.length
    between += g.length * (m - grand) ** 2
    for (const v of g) within += (v - m) ** 2
  }
  const df: [number, number] = [k - 1, N - k]
  const statistic = between / df[0] / (within / df[1])
  const law = FisherSnedecor(df[0], df[1])
  return result({
    test: 'oneWayAnova',
    method: 'One-way analysis of variance',
    statistic,
    symbol: 'F',
    df,
    pValue: pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    effectSize: { name: 'η²', value: between / (between + within) },
    n: N,
  })
}
