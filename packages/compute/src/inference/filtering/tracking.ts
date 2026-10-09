/**
 * How well a sequence of estimates tracks a known truth, and how quickly it follows a step: the lag to cover a fraction
 * of the step, the overshoot and the noise once settled, the root-mean-square error before and after the change, and
 * the coverage of an uncertainty band. For any filter or smoother run against simulated truth. Non-finite errors are
 * left out of the root-mean-square errors, and a metric with nothing to measure is NaN.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `trackingMetrics`. */
export interface TrackingOptions {
  /** The index of the first value after a step change in the truth; omit for a truth without one. */
  change?: number
  /** The fraction of the step that counts as followed (default 0.9). */
  level?: number
  /** The band $\pm z \cdot \text{sd}$ whose coverage is reported (default 1.96: a nominal 95 % band). */
  z?: number
  /** Indices before this are ignored by the errors and the coverage (default 0). */
  burnIn?: number
}

/** Tracking metrics of one estimate sequence. */
export interface TrackingMetrics {
  /**
   * Values from the change until the estimate first covers `level` of the step, counting the first value after the
   * change as 1: the smallest $k \ge 1$ with $(\text{estimate}[c + k - 1] - \text{truth}[c - 1])/\Delta \ge$ `level`,
   * $c$ the change and $\Delta$ the step. NaN without a change, for a step of 0, or when never reached.
   */
  lag: number
  /**
   * The largest excess of the estimate beyond the truth, in the step's direction, from the value that reached `level`
   * on ($\ge 0$; NaN when `lag` is).
   */
  overshoot: number
  /** The root-mean-square error once settled: from $c + \text{lag}$ to the end (NaN when `lag` is). */
  noise: number
  /** The root-mean-square error before the change, from `burnIn` (over every value from `burnIn` without a change). */
  rmseBefore: number
  /** The root-mean-square error from the change on (NaN without a change). */
  rmseAfter: number
  /**
   * The fraction of values from `burnIn` whose truth lies within $\text{estimate} \pm z \cdot \text{sd}$, over the
   * values with a finite estimate and sd (NaN without sds).
   */
  coverage: number
}

/**
 * The root-mean-square of the finite entries of `e` in a range.
 *
 * @param e The errors.
 * @param from The first index (clamped at 0).
 * @param to The index after the last.
 * @returns The root-mean-square, or NaN when the range has no finite entry.
 */
const rms = (e: Float64Array, from: number, to: number) => {
  let s = 0
  let n = 0
  for (let i = Math.max(0, from); i < to; i++)
    if (Number.isFinite(e[i])) {
      s += e[i] * e[i]
      n++
    }
  return n > 0 ? Math.sqrt(s / n) : NaN
}

/**
 * Lag, overshoot, settled noise, errors before and after a change, and band coverage of `estimate` (with optional
 * standard deviations `sd`) against `truth`, all of one length (else `DomainError`).
 *
 * @param estimate The estimates, one per time step (e.g. a filter's means).
 * @param truth The true values, as many.
 * @param sd The estimates' standard deviations, for the coverage; null (default) leaves `coverage` NaN.
 * @param options Where the step change is, what counts as following it, the band and the burn-in.
 * @returns The metrics.
 *
 * @example A step from 0 to 1 at index 3, followed in two values
 * const truth = [0, 0, 0, 1, 1, 1, 1, 1]
 * const estimate = [0, 0.1, -0.1, 0.5, 0.95, 1.1, 1, 1]
 * print(trackingMetrics(estimate, truth, new Array(8).fill(0.1), { change: 3 }))
 */
export function trackingMetrics(
  estimate: ArrayLike<number>,
  truth: ArrayLike<number>,
  sd: ArrayLike<number> | null = null,
  options: TrackingOptions = {},
): TrackingMetrics {
  const n = truth.length
  if (estimate.length !== n || (sd && sd.length !== n))
    throw new DomainError('trackingMetrics', 'trackingMetrics: estimate, truth and sd must have one length')
  const { change, level = 0.9, z = 1.96, burnIn = 0 } = options
  const error = Float64Array.from({ length: n }, (_, i) => estimate[i] - truth[i])
  const hasChange = change !== undefined && change > 0 && change < n
  let lag = NaN
  let overshoot = NaN
  let noise = NaN
  if (hasChange) {
    const before = truth[change - 1]
    const step = truth[change] - before
    if (step !== 0)
      for (let i = change; i < n; i++)
        if ((estimate[i] - before) / step >= level) {
          lag = i - change + 1
          break
        }
    if (Number.isFinite(lag)) {
      overshoot = 0
      for (let i = change + lag - 1; i < n; i++) overshoot = Math.max(overshoot, Math.sign(step) * error[i])
      noise = rms(error, change + lag, n)
    }
  }
  let coverage = NaN
  if (sd) {
    let inside = 0
    let count = 0
    for (let i = burnIn; i < n; i++) {
      if (!Number.isFinite(sd[i]) || !Number.isFinite(estimate[i])) continue
      count++
      if (Math.abs(error[i]) <= z * sd[i]) inside++
    }
    coverage = count > 0 ? inside / count : NaN
  }
  return {
    lag,
    overshoot,
    noise,
    rmseBefore: hasChange ? rms(error, burnIn, change) : rms(error, burnIn, n),
    rmseAfter: hasChange ? rms(error, change, n) : NaN,
    coverage,
  }
}
