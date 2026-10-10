/**
 * Uncertainty of a metric evaluated on a finite test set (metric-confidence-intervals): Wilson and Wald intervals for
 * proportions, the case (or cluster) bootstrap of any metric and of the difference between two models, and DeLong's
 * variance of AUROC and test for two correlated AUROCs.
 *
 * Intervals are two-sided at `level` (default 0.95), with the normal quantile $z_{1 - (1 - \mathit{level})/2}$ for the
 * closed forms. The bootstraps take a random `Stream` and draw resample $r$ from its child stream $r$, so a result
 * is reproducible from the seed and does not depend on the number of resamples drawn before it.
 */

import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import { quantile, standardDeviation } from 'aifn-compute/probability/stats'
import { isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  binaryTruth,
  dense,
  labelList,
  matrix,
  sameLength,
  values,
  vector,
  type Data,
  type Label,
  type Labels,
} from './core'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The Wilson score interval for a proportion of `successes` in `trials` (Wilson 1927): it inverts the score test, never
 * leaves $[0, 1]$ and keeps close to nominal coverage. Default level 0.95. As statsmodels' `proportion_confint` with
 * `method='wilson'`. NaN for no trials.
 *
 * @param successes The number of successes (for an accuracy, the correct cases).
 * @param trials The number of trials $m$.
 * @param options `level`, the coverage of the interval (default 0.95).
 * @returns The lower and upper ends of the interval.
 *
 * @example Eight of ten, and ten of ten
 * print('8 / 10', wilsonInterval(8, 10))
 * print('10 / 10', wilsonInterval(10, 10))
 */
export function wilsonInterval(successes: number, trials: number, options: { level?: number } = {}): [number, number] {
  const z = normalQuantile(1 - (1 - (options.level ?? 0.95)) / 2)
  const p = successes / trials
  const z2n = (z * z) / trials
  const centre = (p + z2n / 2) / (1 + z2n)
  const half = (z / (1 + z2n)) * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))
  return [centre - half, centre + half]
}

/**
 * The Wald interval $\hat p \pm z\sqrt{\hat p(1 - \hat p)/m}$: poor near 0 or 1 and for small $m$, where it can leave
 * $[0, 1]$ (it is not clipped) or have zero width. As statsmodels' `proportion_confint` with `method='normal'`.
 *
 * @param successes The number of successes.
 * @param trials The number of trials $m$.
 * @param options `level`, the coverage of the interval (default 0.95).
 * @returns The lower and upper ends of the interval.
 *
 * @example Past 1 for eight of ten, and no width for ten of ten
 * print('8 / 10', waldInterval(8, 10))
 * print('10 / 10', waldInterval(10, 10))
 */
export function waldInterval(successes: number, trials: number, options: { level?: number } = {}): [number, number] {
  const z = normalQuantile(1 - (1 - (options.level ?? 0.95)) / 2)
  const p = successes / trials
  const half = z * Math.sqrt((p * (1 - p)) / trials)
  return [p - half, p + half]
}

/**
 * Per-case data a bootstrap can resample: numbers, labels, strings, rows, or a rank-1 or rank-2 tensor (resampled
 * along axis 0).
 */
export type Cases = Data | Labels | ArrayLike<unknown> | Tensor

/**
 * Select cases (rows) by index, keeping the input's kind: a rank-1 tensor gives a rank-1 tensor, a rank-2 tensor a
 * matrix of the chosen rows, and an array a new array of the chosen elements. A tensor of higher rank throws
 * `ShapeError`.
 *
 * @param x The per-case data.
 * @param idx The indices of the cases to take, repeats allowed.
 * @returns The chosen cases, in the order of `idx`.
 */
function take<T extends Cases>(x: T, idx: ArrayLike<number>): T {
  if (isTensor(x)) {
    if (x.shape.length === 1) {
      const v = values(x)
      return vector(Float64Array.from(idx, (i) => v[i])) as T
    }
    const d = dense(x, 'bootstrap')
    const out = new Float64Array(idx.length * d.cols)
    for (let k = 0; k < idx.length; k++) out.set(d.data.subarray(idx[k] * d.cols, (idx[k] + 1) * d.cols), k * d.cols)
    return matrix(out, idx.length, d.cols) as T
  }
  const a = x as ArrayLike<unknown>
  return Array.from(idx, (i) => a[i]) as unknown as T
}

/**
 * The number of cases: a tensor's first dimension, or an array's length.
 *
 * @param x The per-case data.
 * @returns The number of cases.
 */
const caseCount = (x: Cases) => (isTensor(x) ? x.shape[0] : (x as ArrayLike<unknown>).length)

/** The result of a metric bootstrap. */
export type MetricBootstrap = {
  /** The metric on the full test set. */
  estimate: number
  /** The metric on each resample, in resample order (NaN where it was undefined). */
  replicates: Tensor
  /** The sample standard deviation of the defined replicates (NaN with fewer than two). */
  standardError: number
  /**
   * The interval at `level`: percentile (default), the replicates' quantiles $[q_{\alpha/2}, q_{1-\alpha/2}]$, or
   * basic, $[2\hat\theta - q_{1-\alpha/2}, 2\hat\theta - q_{\alpha/2}]$, with $\alpha = 1 - \mathit{level}$ and
   * $\hat\theta$ the estimate.
   */
  interval: [number, number]
  /** The coverage of the interval. */
  level: number
  /** Replicates that were NaN (for example a resample without positives) and were left out of the interval. */
  undefinedReplicates: number
}

/** Options of the bootstraps. */
export type BootstrapOptions = {
  /** Number of resamples $B$ (default 2000). */
  resamples?: number
  /** The coverage of the interval (default 0.95). */
  level?: number
  /** The kind of interval: `'percentile'` (default) or `'basic'` (see `MetricBootstrap`). */
  method?: 'percentile' | 'basic'
  /** A group per case: whole groups are resampled (a cluster bootstrap), so correlated cases stay together. */
  groups?: Labels
}

/**
 * Resampled case indices, whole groups at a time when groups are given. Throws `ShapeError` when there is not one
 * group per case.
 *
 * @param s The random stream; resample $r$ draws from its child stream $r$.
 * @param n The number of cases.
 * @param groups The group of each case; left out, cases are drawn one at a time.
 * @returns A function from the resample number $r$ to its case indices: $n$ draws with replacement, or as many
 *   groups as there are, drawn with replacement, with all their cases (so a resample's size varies).
 */
function resampler(s: Stream, n: number, groups?: Labels): (r: number) => Int32Array {
  if (!groups) {
    return (r) => {
      const t = child(s, r)
      return Int32Array.from({ length: n }, () => integers(t, n))
    }
  }
  const g = labelList(groups)
  if (g.length !== n) throw new ShapeError('metrics', 'metrics: bootstrap: one group per case')
  const members = new Map<Label, number[]>()
  g.forEach((v, i) => members.set(v, [...(members.get(v) ?? []), i]))
  const lists = [...members.values()]
  return (r) => {
    const t = child(s, r)
    const out: number[] = []
    for (let k = 0; k < lists.length; k++) out.push(...lists[integers(t, lists.length)])
    return Int32Array.from(out)
  }
}

/**
 * The bootstrap result from the estimate and its replicates: NaN replicates are counted and left out of the
 * interval and the standard error.
 *
 * @param estimate The metric on the full test set.
 * @param reps The metric on each resample.
 * @param o The bootstrap options; `level` and `method` are read.
 * @returns The result.
 */
function summarise(estimate: number, reps: Float64Array, o: BootstrapOptions): MetricBootstrap {
  const level = o.level ?? 0.95
  const finite = reps.filter((v) => !Number.isNaN(v))
  const a = (1 - level) / 2
  const [lo, hi] = finite.length ? [quantile(finite, a), quantile(finite, 1 - a)] : [NaN, NaN]
  return {
    estimate,
    replicates: vector(reps),
    standardError: finite.length > 1 ? standardDeviation(finite, { sample: true }) : NaN,
    interval: (o.method ?? 'percentile') === 'percentile' ? [lo, hi] : [2 * estimate - hi, 2 * estimate - lo],
    level,
    undefinedReplicates: reps.length - finite.length,
  }
}

/**
 * The case bootstrap of a metric (Efron 1979; metric-confidence-intervals): resample the $n$ cases with replacement
 * `resamples` times (each resample from its own child stream `child(s, r)`), recompute `metric(target, prediction)` on
 * each, and report the percentile (or basic) interval and the standard error. Targets and predictions are resampled
 * together; matrices and tensors are resampled by rows. Throws `ShapeError` when they differ in number of cases.
 *
 * @param s The random stream the resamples are drawn from.
 * @param metric The metric, called as `metric(target, prediction)` on the full set and on every resample.
 * @param target The per-case targets.
 * @param prediction The per-case predictions, one per target.
 * @param options `resamples`, `level`, `method` and `groups` (for a cluster bootstrap).
 * @returns The estimate, the replicates, their standard error and the interval.
 *
 * @example A 95% interval for an accuracy of 0.8
 * const y = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
 * const p = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
 * const b = bootstrapMetric(stream(0), accuracy, y, p, { resamples: 500 })
 * print('estimate', b.estimate, 'interval', b.interval, 'standard error', b.standardError)
 */
export function bootstrapMetric<T extends Cases, P extends Cases>(
  s: Stream,
  metric: (target: T, prediction: P) => number,
  target: T,
  prediction: P,
  options: BootstrapOptions = {},
): MetricBootstrap {
  const n = caseCount(target)
  sameLength({ length: n }, { length: caseCount(prediction) }, 'bootstrapMetric')
  const draw = resampler(s, n, options.groups)
  const B = options.resamples ?? 2000
  const reps = new Float64Array(B)
  for (let r = 0; r < B; r++) {
    const idx = draw(r)
    reps[r] = metric(take(target, idx), take(prediction, idx))
  }
  return summarise(metric(target, prediction), reps, options)
}

/**
 * The paired bootstrap of the difference `metric(target, predictionA) - metric(target, predictionB)` between two models
 * evaluated on the same cases: each resample is shared by both, which uses their correlation and gives a narrower,
 * correct interval. The predictions' lengths are not checked against the target's.
 *
 * @param s The random stream the resamples are drawn from.
 * @param metric The metric, called as `metric(target, prediction)` for each model.
 * @param target The per-case targets.
 * @param predictionA The first model's per-case predictions.
 * @param predictionB The second model's per-case predictions.
 * @param options `resamples`, `level`, `method` and `groups` (for a cluster bootstrap).
 * @returns The bootstrap of the difference: an interval that excludes 0 favours one model.
 *
 * @example Is model A's MAE lower than model B's?
 * const y = [1, 2, 3, 4, 5, 6, 7, 8]
 * const a = [1.1, 2.2, 2.9, 4.1, 5.2, 5.8, 7.1, 8.2]
 * const b = [1.5, 2.6, 3.5, 3.4, 5.6, 6.6, 6.5, 8.7]
 * const d = pairedBootstrap(stream(0), meanAbsoluteError, y, a, b, { resamples: 500 })
 * print('MAE(A) - MAE(B)', d.estimate, 'interval', d.interval)
 */
export function pairedBootstrap<T extends Cases, P extends Cases>(
  s: Stream,
  metric: (target: T, prediction: P) => number,
  target: T,
  predictionA: P,
  predictionB: P,
  options: BootstrapOptions = {},
): MetricBootstrap {
  const n = caseCount(target)
  const draw = resampler(s, n, options.groups)
  const B = options.resamples ?? 2000
  const reps = new Float64Array(B)
  for (let r = 0; r < B; r++) {
    const idx = draw(r)
    const t = take(target, idx)
    reps[r] = metric(t, take(predictionA, idx)) - metric(t, take(predictionB, idx))
  }
  return summarise(metric(target, predictionA) - metric(target, predictionB), reps, options)
}

// ── DeLong ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Structural components of AUROC (DeLong et al. 1988): $V_{10}(x_i)$ the mean over negatives of $\psi(x_i, y_j)$ for
 * each positive score $x_i$, and $V_{01}(y_j)$ the mean over positives for each negative score $y_j$, with
 * $\psi = 1, \tfrac{1}{2}, 0$ for greater, tied, smaller. Costs $O(mn)$.
 *
 * @param y The binary truth, 1 for a positive case.
 * @param s The score of each case, higher for positive.
 * @returns `v10` and `v01`, the components; `auc`, their mean (the AUROC); `m` and `n`, the numbers of positives and
 *   negatives.
 */
function components(y: Uint8Array, s: Float64Array) {
  const pos: number[] = []
  const neg: number[] = []
  y.forEach((v, i) => (v ? pos : neg).push(s[i]))
  const v10 = pos.map((x) => neg.reduce((a, yv) => a + (x > yv ? 1 : x === yv ? 0.5 : 0), 0) / neg.length)
  const v01 = neg.map((yv) => pos.reduce((a, x) => a + (x > yv ? 1 : x === yv ? 0.5 : 0), 0) / pos.length)
  const auc = v10.reduce((a, b) => a + b, 0) / v10.length
  return { v10, v01, auc, m: pos.length, n: neg.length }
}

/**
 * The sample covariance (divisor $m - 1$) of two equal-length sequences.
 *
 * @param a The first sequence.
 * @param b The second sequence, of the same length.
 * @returns The covariance; the variance when `a` and `b` are the same.
 */
const covariance = (a: number[], b: number[]) => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length
  const mb = b.reduce((s, v) => s + v, 0) / b.length
  return a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / (a.length - 1)
}

/**
 * AUROC with DeLong's variance (DeLong et al. 1988): $\var = S_{10}/m + S_{01}/n$ from the sample variances of the
 * structural components over the $m$ positives and $n$ negatives, and the normal interval at `level` (default 0.95),
 * not clipped to $[0, 1]$. Costs $O(mn)$. NaN without at least two positives and two negatives. As pROC's
 * `var(method = "delong")`.
 *
 * @param yTrue The true labels.
 * @param scores The score of each case, higher for the positive class.
 * @param options `positive`, the positive class (default `1` or `true` when present, else the last label), and
 *   `level`, the coverage of the interval (default 0.95).
 * @returns `auroc`, its `variance` and `standardError`, and the normal `interval`.
 *
 * @example An AUROC of 0.75 from four cases is very uncertain
 * print(aurocDeLong([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))
 */
export function aurocDeLong(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label; level?: number } = {},
): { auroc: number; variance: number; standardError: number; interval: [number, number] } {
  const { y } = binaryTruth(yTrue, options.positive)
  const s = values(scores)
  sameLength(y, s, 'aurocDeLong')
  const c = components(y, s)
  const variance = covariance(c.v10, c.v10) / c.m + covariance(c.v01, c.v01) / c.n
  const se = Math.sqrt(variance)
  const z = normalQuantile(1 - (1 - (options.level ?? 0.95)) / 2)
  return { auroc: c.auc, variance, standardError: se, interval: [c.auc - z * se, c.auc + z * se] }
}

/**
 * DeLong's test for two correlated AUROCs on the same cases: $z = (A_1 - A_2)/\sqrt{\var(A_1 - A_2)}$, with the
 * covariance of the two AUROCs from their structural components, and the two-sided p-value. As pROC's `roc.test`
 * with `method = "delong"`. The scores' lengths are not checked against the labels'.
 *
 * @param yTrue The true labels.
 * @param scoresA The first model's score for each case.
 * @param scoresB The second model's score for each case.
 * @param options `positive`, the positive class (default `1` or `true` when present, else the last label).
 * @returns `difference` $A_1 - A_2$, its `standardError`, the statistic `z` and the two-sided `pValue`.
 *
 * @example A perfect ranking against a weaker one on eight cases
 * const y = [0, 0, 0, 0, 1, 1, 1, 1]
 * const a = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * const b = [0.1, 0.6, 0.3, 0.8, 0.2, 0.7, 0.4, 0.9]
 * print(delongTest(y, a, b))
 */
export function delongTest(
  yTrue: Labels,
  scoresA: Data,
  scoresB: Data,
  options: { positive?: Label } = {},
): { difference: number; standardError: number; z: number; pValue: number } {
  const { y } = binaryTruth(yTrue, options.positive)
  const a = components(y, values(scoresA))
  const b = components(y, values(scoresB))
  const variance =
    (covariance(a.v10, a.v10) + covariance(b.v10, b.v10) - 2 * covariance(a.v10, b.v10)) / a.m +
    (covariance(a.v01, a.v01) + covariance(b.v01, b.v01) - 2 * covariance(a.v01, b.v01)) / a.n
  const difference = a.auc - b.auc
  const se = Math.sqrt(variance)
  const z = difference / se
  return { difference, standardError: se, z, pValue: 2 * (1 - normalCdf(Math.abs(z))) }
}
