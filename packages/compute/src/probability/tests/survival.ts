/**
 * Survival estimators from right-censored times: the Kaplan–Meier estimate of the survival function with Greenwood's
 * variance and linear or log–log intervals (as scipy's `ecdf` on `CensoredData`), the Nelson–Aalen estimate of the
 * cumulative hazard with Aalen's variance, and the log-rank test that several groups share one survival function.
 *
 * The data are pairs of a time and an event flag: 1 when the event was observed at that time, 0 when the subject was
 * censored then (still event-free when last seen). Everything is built on the risk table of the distinct event times
 * $t_j$, with $n_j$ subjects at risk just before $t_j$ and $d_j$ events at it; a subject censored at $t_j$ counts as at
 * risk there. Curves are reported at the event times only, as step functions.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, tensor, toFlat, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import { ChiSquare, Normal } from 'aifn-compute/probability/distributions'
import { checkLevel, pValueOf, result, type TestResult } from './protocol'

/**
 * Times with event flags (1 an event, 0 censored at that time), checked: at least one time, equal lengths, finite
 * times and flags of 0 or 1 (a `DomainError` otherwise).
 *
 * @param time The times.
 * @param event The event flags, one per time.
 * @param where The caller's name, for error messages.
 * @returns The times `t` and flags `e` as new Float64Arrays.
 */
function censored(time: VectorLike, event: VectorLike, where: string): { t: Float64Array; e: Float64Array } {
  const t = dense.toF64(time, where)
  const e = dense.toF64(event, where)
  if (t.length === 0) throw new DomainError(where, `${where}: needs at least one time`)
  if (t.length !== e.length) throw new DomainError(where, `${where}: time and event lengths differ`)
  for (const v of t) if (!Number.isFinite(v)) throw new DomainError(where, `${where}: times must be finite`)
  for (const v of e) if (v !== 0 && v !== 1) throw new DomainError(where, `${where}: events must be 0 or 1`)
  return { t, e }
}

/**
 * The risk table at each distinct event time: the time, the number at risk just before it, and the events at it.
 * Times with only censorings have no row, but leave the risk set.
 *
 * @param t The times, in any order.
 * @param e The event flags, one per time.
 * @returns Arrays over the event times, ascending: `times`, `atRisk` ($n_j$), `events` ($d_j$) and `censoredAt`
 *   (the censorings at the same time).
 */
function riskTable(t: Float64Array, e: Float64Array) {
  const order = Array.from(t.keys()).sort((i, j) => t[i] - t[j])
  const times: number[] = []
  const atRisk: number[] = []
  const events: number[] = []
  const censoredAt: number[] = []
  let risk = t.length
  for (let k = 0; k < order.length;) {
    const v = t[order[k]]
    let d = 0
    let c = 0
    let j = k
    for (; j < order.length && t[order[j]] === v; j++) {
      if (e[order[j]] === 1) d++
      else c++
    }
    if (d > 0) {
      times.push(v)
      atRisk.push(risk)
      events.push(d)
      censoredAt.push(c)
    }
    risk -= j - k
    k = j
  }
  return { times, atRisk, events, censoredAt }
}

/**
 * A list of numbers as a rank-1 tensor.
 *
 * @param v The numbers.
 * @returns A new tensor of them.
 */
const vec = (v: readonly number[] | Float64Array) => fromData(Float64Array.from(v), [v.length])

/** A Kaplan–Meier estimate: the survival curve at each distinct event time, with pointwise intervals. */
export type KaplanMeier = {
  /** Always `'survival-curve'`. */
  readonly kind: 'survival-curve'
  /** Distinct event times, ascending. */
  readonly time: Tensor
  /** The number at risk just before each time. */
  readonly atRisk: Tensor
  /** The number of events at each time. */
  readonly events: Tensor
  /** $\hat S(t)$ just after each time (a step function, right-continuous). */
  readonly survival: Tensor
  /** Greenwood's standard error of $\hat S(t)$. */
  readonly standardError: Tensor
  /** The lower end of the pointwise interval at each time. */
  readonly lower: Tensor
  /** The upper end of the pointwise interval at each time. */
  readonly upper: Tensor
  /** The intervals' confidence level. */
  readonly level: number
  /** How the intervals were built. */
  readonly interval: 'linear' | 'log-log'
  /** The smallest time with $\hat S(t) \le \tfrac12$ (NaN when the curve stays above $\tfrac12$). */
  readonly median: number
}

/**
 * The Kaplan–Meier estimator (Kaplan and Meier, 1958): $\hat S(t) = \prod_{j: t_j \le t} (1 - d_j/n_j)$ over the
 * distinct event times $t_j$, with $d_j$ events among $n_j$ at risk. Greenwood's (1926) variance is
 * $\hat S^2 \sum_j d_j/(n_j(n_j - d_j))$. `linear` intervals are $\hat S \pm z\,\mathrm{se}$ cut to $[0, 1]$;
 * `log-log` (default) intervals transform through $\log(-\log \hat S)$, which keeps them in $[0, 1]$:
 * $\hat S^{\exp(\pm z\sigma)}$ with $\sigma = \sqrt{\sum_j d_j/(n_j(n_j - d_j))}/\lvert \log \hat S \rvert$. Where
 * $\hat S$ is 0 or 1 the log–log interval is undefined (NaN), as in scipy. Once $\hat S$ reaches 0 its standard error
 * is NaN.
 *
 * @param time The time of each subject's event or censoring.
 * @param event 1 where the event was observed, 0 where the time was censored.
 * @param options The intervals.
 * @param options.level The confidence level of the pointwise intervals, in $(0, 1)$.
 * @param options.interval `log-log` or `linear`.
 * @returns The curve at each distinct event time, with its standard errors, intervals and median.
 *
 * @example Eight subjects, three of them censored
 * const km = kaplanMeier([3, 5, 6, 8, 10, 12, 15, 18], [1, 1, 0, 1, 1, 0, 1, 0])
 * print('event times:', km.time, ' at risk:', km.atRisk)
 * print('S(t):', km.survival)
 * print('95% log-log interval:', km.lower, km.upper)
 * print('median survival:', km.median)
 */
export function kaplanMeier(
  time: VectorLike,
  event: VectorLike,
  { level = 0.95, interval = 'log-log' }: { level?: number; interval?: 'linear' | 'log-log' } = {},
): KaplanMeier {
  checkLevel(level, 'kaplanMeier')
  const { t, e } = censored(time, event, 'kaplanMeier')
  const r = riskTable(t, e)
  const z = Normal(0, 1).isf((1 - level) / 2) as number
  const k = r.times.length
  const S = new Float64Array(k)
  const se = new Float64Array(k)
  const lo = new Float64Array(k)
  const hi = new Float64Array(k)
  let s = 1
  let g = 0
  let median = NaN
  for (let j = 0; j < k; j++) {
    const [n, d] = [r.atRisk[j], r.events[j]]
    s *= 1 - d / n
    g += n > d ? d / (n * (n - d)) : Infinity
    S[j] = s
    se[j] = s * Math.sqrt(g)
    if (Number.isNaN(median) && s <= 0.5) median = r.times[j]
    if (interval === 'linear') {
      lo[j] = Math.max(0, s - z * se[j])
      hi[j] = Math.min(1, s + z * se[j])
    } else {
      const sigma = Math.sqrt(g) / Math.abs(Math.log(s))
      const ok = s > 0 && s < 1 && Number.isFinite(sigma)
      lo[j] = ok ? s ** Math.exp(z * sigma) : NaN
      hi[j] = ok ? s ** Math.exp(-z * sigma) : NaN
    }
  }
  return {
    kind: 'survival-curve',
    time: vec(r.times),
    atRisk: vec(r.atRisk),
    events: vec(r.events),
    survival: vec(S),
    standardError: vec(se),
    lower: vec(lo),
    upper: vec(hi),
    level,
    interval,
    median,
  }
}

/** A Nelson–Aalen estimate of the cumulative hazard at each distinct event time. */
export type NelsonAalen = {
  /** Always `'cumulative-hazard'`. */
  readonly kind: 'cumulative-hazard'
  /** Distinct event times, ascending. */
  readonly time: Tensor
  /** The number at risk just before each time. */
  readonly atRisk: Tensor
  /** The number of events at each time. */
  readonly events: Tensor
  /** $\hat H(t)$ just after each time. */
  readonly cumulativeHazard: Tensor
  /** Aalen's standard error $\sqrt{\sum_j d_j/n_j^2}$. */
  readonly standardError: Tensor
  /** The lower end of the pointwise interval at each time. */
  readonly lower: Tensor
  /** The upper end of the pointwise interval at each time. */
  readonly upper: Tensor
  /** The intervals' confidence level. */
  readonly level: number
}

/**
 * The Nelson–Aalen estimator (Nelson, 1972; Aalen, 1978) of the cumulative hazard:
 * $\hat H(t) = \sum_{j: t_j \le t} d_j/n_j$, with Aalen's variance $\sum_j d_j/n_j^2$ and the log-transformed
 * interval $\hat H \exp(\pm z\,\mathrm{se}/\hat H)$, which stays positive. $\exp(-\hat H)$ is the
 * Fleming–Harrington estimate of the survival function.
 *
 * @param time The time of each subject's event or censoring.
 * @param event 1 where the event was observed, 0 where the time was censored.
 * @param options The intervals.
 * @param options.level The confidence level of the pointwise intervals, in $(0, 1)$.
 * @returns The cumulative hazard at each distinct event time, with its standard errors and intervals.
 *
 * @example The cumulative hazard of the eight subjects, and the survival it implies
 * const na = nelsonAalen([3, 5, 6, 8, 10, 12, 15, 18], [1, 1, 0, 1, 1, 0, 1, 0])
 * print('H(t):', na.cumulativeHazard)
 * print('exp(-H):', exp(neg(na.cumulativeHazard)))
 * print('Kaplan-Meier:', kaplanMeier([3, 5, 6, 8, 10, 12, 15, 18], [1, 1, 0, 1, 1, 0, 1, 0]).survival)
 */
export function nelsonAalen(
  time: VectorLike,
  event: VectorLike,
  { level = 0.95 }: { level?: number } = {},
): NelsonAalen {
  checkLevel(level, 'nelsonAalen')
  const { t, e } = censored(time, event, 'nelsonAalen')
  const r = riskTable(t, e)
  const z = Normal(0, 1).isf((1 - level) / 2) as number
  const k = r.times.length
  const H = new Float64Array(k)
  const se = new Float64Array(k)
  const lo = new Float64Array(k)
  const hi = new Float64Array(k)
  let h = 0
  let v = 0
  for (let j = 0; j < k; j++) {
    const [n, d] = [r.atRisk[j], r.events[j]]
    h += d / n
    v += d / (n * n)
    H[j] = h
    se[j] = Math.sqrt(v)
    lo[j] = h * Math.exp((-z * se[j]) / h)
    hi[j] = h * Math.exp((z * se[j]) / h)
  }
  return {
    kind: 'cumulative-hazard',
    time: vec(r.times),
    atRisk: vec(r.atRisk),
    events: vec(r.events),
    cumulativeHazard: vec(H),
    standardError: vec(se),
    lower: vec(lo),
    upper: vec(hi),
    level,
  }
}

/**
 * The result of the log-rank test: the protocol's fields with the group labels in order (`groups`) and each group's
 * observed and expected numbers of events (`observed`, `expected`).
 */
export type LogRankTest = TestResult & { groups: number[]; observed: Tensor; expected: Tensor }

/**
 * The log-rank test (Mantel, 1966; Peto and Peto, 1972) of $H_0$: $G \ge 2$ groups share one survival function. At
 * each distinct event time $t_j$ with $n_j$ at risk and $d_j$ events, group $g$ expects $d_j n_{gj}/n_j$ of them; with
 * $\mathbf{O} - \mathbf{E}$ the observed minus expected events and $\Vmat$ their hypergeometric covariance,
 * $V_{gh} = \sum_j \frac{d_j(n_j - d_j)}{n_j - 1} \frac{n_{gj}}{n_j}\left(\delta_{gh} - \frac{n_{hj}}{n_j}\right)$,
 * the statistic $(\mathbf{O} - \mathbf{E})^\top \Vmat^{-1} (\mathbf{O} - \mathbf{E})$ over the first $G - 1$ groups
 * is $\chi^2(G - 1)$ under the null. With two groups it is the square of scipy's `logrank` statistic. `group` labels
 * each observation (any numbers); groups are ordered by label. Throws `DomainError` for fewer than two groups or
 * mismatched lengths.
 *
 * @param time The time of each subject's event or censoring.
 * @param event 1 where the event was observed, 0 where the time was censored.
 * @param group The group label of each subject.
 * @returns The test result, with each group's observed and expected events.
 *
 * @example Earlier events in one group, but too few subjects to reject at 5%
 * const time = [3, 5, 6, 8, 10, 12, 15, 18, 9, 11, 14, 16, 20, 22, 25, 30]
 * const event = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
 * const group = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
 * const r = logRankTest(time, event, group)
 * print('chi2 =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print('observed:', r.observed, ' expected:', r.expected)
 */
export function logRankTest(time: VectorLike, event: VectorLike, group: VectorLike): LogRankTest {
  const { t, e } = censored(time, event, 'logRankTest')
  const g = dense.toF64(group, 'logRankTest')
  if (g.length !== t.length) throw new DomainError('logRankTest', 'logRankTest: group and time lengths differ')
  const labels = [...new Set(g)].sort((a, b) => a - b)
  const G = labels.length
  if (G < 2) throw new DomainError('logRankTest', 'logRankTest: needs at least two groups')
  const index = new Map(labels.map((l, i) => [l, i]))
  const order = Array.from(t.keys()).sort((i, j) => t[i] - t[j])
  const risk = labels.map((l) => g.filter((v) => v === l).length)
  const O = new Float64Array(G)
  const E = new Float64Array(G)
  const V = new Float64Array(G * G)
  for (let k = 0; k < order.length;) {
    const v = t[order[k]]
    const dg = new Float64Array(G)
    let j = k
    const leaving = new Float64Array(G)
    for (; j < order.length && t[order[j]] === v; j++) {
      const gi = index.get(g[order[j]])!
      leaving[gi]++
      if (e[order[j]] === 1) dg[gi]++
    }
    const d = dg.reduce((a, b) => a + b, 0)
    const n = risk.reduce((a, b) => a + b, 0)
    if (d > 0) {
      const scale = n > 1 ? (d * (n - d)) / (n - 1) : 0
      for (let a = 0; a < G; a++) {
        O[a] += dg[a]
        E[a] += (d * risk[a]) / n
        for (let b = 0; b < G; b++) V[a * G + b] += scale * (risk[a] / n) * ((a === b ? 1 : 0) - risk[b] / n)
      }
    }
    for (let a = 0; a < G; a++) risk[a] -= leaving[a]
    k = j
  }
  // Drop the last group: V is singular (its rows sum to zero).
  const r = G - 1
  const diff = Array.from({ length: r }, (_, a) => O[a] - E[a])
  const Vr = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) => V[a * G + b]))
  const x = toFlat(solve(tensor(Vr), tensor(diff)))
  const statistic = diff.reduce((s, v, a) => s + v * x[a], 0)
  const law = ChiSquare(r)
  return {
    ...result({
      test: 'logRankTest',
      method: 'Log-rank test',
      statistic,
      symbol: '\\chi^2',
      df: r,
      pValue: pValueOf(law, statistic, 'upper'),
      alternative: 'two-sided',
      tail: 'upper',
      null: law,
      n: t.length,
    }),
    groups: labels,
    observed: vec(O),
    expected: vec(E),
  }
}
