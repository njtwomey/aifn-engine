/**
 * Uncertainty of a metric evaluated on a finite test set (metric-confidence-intervals): Wilson and Wald intervals for
 * proportions, the case (or cluster) bootstrap of any metric and of the difference between two models, and DeLong's
 * variance of AUROC and test for two correlated AUROCs.
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
 * leaves [0, 1] and keeps close to nominal coverage. Default level 0.95.
 */
export function wilsonInterval(successes: number, trials: number, options: { level?: number } = {}): [number, number] {
  const z = normalQuantile(1 - (1 - (options.level ?? 0.95)) / 2)
  const p = successes / trials
  const z2n = (z * z) / trials
  const centre = (p + z2n / 2) / (1 + z2n)
  const half = (z / (1 + z2n)) * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))
  return [centre - half, centre + half]
}

/** The Wald interval p̂ ± z√(p̂(1 − p̂)/m): poor near 0 or 1 and for small m. */
export function waldInterval(successes: number, trials: number, options: { level?: number } = {}): [number, number] {
  const z = normalQuantile(1 - (1 - (options.level ?? 0.95)) / 2)
  const p = successes / trials
  const half = z * Math.sqrt((p * (1 - p)) / trials)
  return [p - half, p + half]
}

/** Per-case data a bootstrap can resample: numbers, labels, strings, rows, or a tensor (resampled along axis 0). */
export type Cases = Data | Labels | ArrayLike<unknown> | Tensor

/** Select cases (rows) by index, keeping the input's kind. */
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

const caseCount = (x: Cases) => (isTensor(x) ? x.shape[0] : (x as ArrayLike<unknown>).length)

/** The result of a metric bootstrap. */
export type MetricBootstrap = {
  estimate: number
  replicates: Tensor
  standardError: number
  /** The interval at `level`: percentile (default) or basic. */
  interval: [number, number]
  level: number
  /** Replicates that were NaN (for example a resample without positives) and were left out of the interval. */
  undefinedReplicates: number
}

/** Options of the bootstraps. */
export type BootstrapOptions = {
  /** Number of resamples B (default 2000). */
  resamples?: number
  level?: number
  method?: 'percentile' | 'basic'
  /** A group per case: whole groups are resampled (a cluster bootstrap), so correlated cases stay together. */
  groups?: Labels
}

/** Resampled case indices, whole groups at a time when groups are given. */
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
 * The case bootstrap of a metric (Efron 1979; metric-confidence-intervals): resample the n cases with replacement
 * `resamples` times (each resample from its own child stream `s.child(r)`), recompute `metric(target, prediction)` on
 * each, and report the percentile (or basic) interval and the standard error. Targets and predictions are resampled
 * together; matrices and tensors are resampled by rows.
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
 * The paired bootstrap of the difference metric(target, A) − metric(target, B) between two models evaluated on the
 * same cases: each resample is shared by both, which uses their correlation and gives a narrower, correct interval.
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
 * Structural components of AUROC (DeLong et al. 1988): V₁₀(xᵢ) = mean over negatives of ψ(xᵢ, yⱼ) for each positive
 * and V₀₁(yⱼ) likewise, with ψ = 1, ½, 0 for greater, tied, smaller.
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

const covariance = (a: number[], b: number[]) => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length
  const mb = b.reduce((s, v) => s + v, 0) / b.length
  return a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / (a.length - 1)
}

/**
 * AUROC with DeLong's variance (DeLong et al. 1988): Var = S₁₀/m + S₀₁/n from the sample variances of the structural
 * components over the m positives and n negatives, and the normal interval at `level` (default 0.95). O(mn).
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
 * DeLong's test for two correlated AUROCs on the same cases: z = (A₁ − A₂)/√Var(A₁ − A₂), with the covariance of the
 * two AUROCs from their structural components, and the two-sided p-value.
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
