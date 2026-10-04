/**
 * Tests on one series or several groups: the Ljung–Box (and Box–Pierce) portmanteau test of autocorrelation,
 * Grubbs' test for one outlier, and the one-way analysis of variance (scipy's `f_oneway`).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { toFlat, type VectorLike } from 'aifn-compute/foundation/tensor'
import { ChiSquare, FisherSnedecor, StudentT } from 'aifn-compute/probability/distributions'
import { autocorrelation } from 'aifn-compute/probability/stats'
import { continuousLaw, meanAndVariance, pValueOf, result, sample, type TestResult } from './protocol'

/**
 * The Ljung–Box test (Ljung and Box, 1978) of H₀: the first h autocorrelations of a series of length n are zero:
 * Q = n(n + 2) Σₖ₌₁ʰ ρ̂ₖ²/(n − k), approximately χ²(h − `fitted`) under the null, where `fitted` counts the ARMA
 * parameters estimated when the series are residuals. `boxPierce` uses Q = n Σ ρ̂ₖ² instead (Box and Pierce, 1970).
 * ρ̂ₖ is the biased sample autocorrelation (`aifn-compute/probability/stats`'s `autocorrelation`).
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

/** The result of Grubbs' test: the protocol's fields and the suspected outlier. */
export type GrubbsTest = TestResult & { index: number; value: number }

/**
 * Grubbs' test (Grubbs, 1950) for one outlier in a normal sample of size n ≥ 3: G = max |xᵢ − x̄|/s (`two-sided`), or
 * (max − x̄)/s (`greater`, the largest value) or (x̄ − min)/s (`less`, the smallest). With
 * t = √(n(n − 2)G²/((n − 1)² − nG²)), P(G ≥ g) ≈ c·n·P(T_{n−2} ≥ t) (c = 2 two-sided, 1 one-sided), the Bonferroni
 * bound that gives Grubbs' critical values; the null law is that bound, capped at 1, on [0, (n − 1)/√n].
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
 * The one-way analysis of variance of k ≥ 2 groups (Fisher): F = (SS_between/(k − 1))/(SS_within/(N − k)), which
 * follows F(k − 1, N − k) when every group is normal with one common mean and variance. The effect size is
 * η² = SS_between/SS_total.
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
