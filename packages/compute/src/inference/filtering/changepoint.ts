/**
 * Bayesian online changepoint detection (Adams and MacKay, 2007, "Bayesian Online Changepoint Detection",
 * arXiv:0710.3742): the exact forward recursion over the run length $r_t$, the number of observations since the last
 * changepoint, for any segment model with a conjugate posterior.
 *
 * With a hazard $H(\tau)$ and each run's posterior predictive $\pi_t^{(r)} = p(x_t \mid r_{t-1} = r, x^{(r)})$, every
 * run either grows,
 *
 * $p(r_t = r + 1, x_{1:t}) = p(r_{t-1} = r, x_{1:t-1})\, \pi_t^{(r)} (1 - H(r + 1))$,
 *
 * or all runs feed one changepoint,
 *
 * $p(r_t = 0, x_{1:t}) = \sum_r p(r_{t-1} = r, x_{1:t-1})\, \pi_t^{(r)} H(r + 1)$,
 *
 * and dividing by the evidence $p(x_t \mid x_{1:t-1})$ gives the run-length posterior. Everything is kept in log space.
 * Each run carries the statistics of its conjugate posterior (a count and running sums), updated by one term per
 * observation; the run at $r = 0$ starts from the prior, so the observation after a changepoint is the first of the
 * new segment. Run lengths whose posterior mass falls below a threshold, or beyond the `maxRuns` most probable, are
 * discarded and the rest renormalised, which bounds the cost per step (Adams and MacKay, §2.4).
 *
 * The segment models here are the conjugate families of the paper and its usual companions: a normal with known
 * variance (normal prior on the mean), a normal with unknown mean and precision (normal–gamma, Student t predictive),
 * Poisson counts (gamma prior, negative binomial predictive), Bernoulli trials (beta prior) and an autoregression with
 * unknown coefficients and noise (normal–inverse-gamma, recursive least squares). Any other model can implement
 * `ConjugatePredictive`.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import {
  concat,
  fromData,
  isTensor,
  logsumexp,
  take,
  toFlat,
  type Tensor,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { Bernoulli, NegativeBinomial, Normal, StudentT, type Univariate } from 'aifn-compute/probability/distributions'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Protocol ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Per-run statistics of a conjugate posterior: named tensors, each with a leading axis of one entry per run. */
export type RunStats = { readonly [name: string]: Tensor }

/**
 * A segment model with a conjugate posterior, as BOCPD needs it. `O` is one observation: a number, or a record that
 * also holds what the next value is conditioned on (an autoregression's lags).
 */
export interface ConjugatePredictive<O = number, S extends RunStats = RunStats> {
  /** A readable name, for display (and in the name of `bocpd`'s algorithm). */
  readonly name: string
  /** The prior's statistics, as a single run (every field has a leading axis of length 1). */
  prior(): S
  /**
   * The posterior predictive of the next observed value under each run: a batch of $R$. `x` supplies what the value is
   * conditioned on, when the model needs it (lags); its own value is not used.
   */
  predictive(stats: S, x?: O): Univariate<Tensor>
  /** The statistics of every run after absorbing `x`. */
  update(stats: S, x: O): S
  /** The observed value of an observation (the number itself for scalar models). */
  value(x: O): number
}

/**
 * The hazard: a changepoint probability per step, constant (memoryless, geometric gaps with mean $1/H$), or a function
 * of the run length $\tau = r_{t-1} + 1$ giving $H(\tau)$ for each run (a float64 tensor in, one of the same length
 * out).
 */
export type Hazard = number | ((tau: Tensor) => Tensor)

/**
 * The constant hazard of geometric gaps with mean `meanGap` $\lambda$: $H = 1/\lambda$ (Adams and MacKay, §2.1).
 * A mean gap below 1 throws `DomainError`.
 *
 * @param meanGap The expected number of observations between changepoints $\lambda$, at least 1.
 * @returns The probability $H$ of a changepoint at each step.
 *
 * @example A change every 250 steps on average
 * print('H =', constantHazard(250))
 */
export function constantHazard(meanGap: number): number {
  if (!(meanGap >= 1))
    throw new DomainError('constantHazard', `constantHazard: the mean gap must be at least 1, got ${meanGap}`)
  return 1 / meanGap
}

// ── State ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The recursion's state after $t$ observations. */
export interface BocpdState<S extends RunStats = RunStats> extends Status {
  /** Observations absorbed. */
  t: number
  /** The run lengths still tracked, int32 `[R]` (ascending; not contiguous once pruning has removed some). */
  runLengths: Tensor
  /** $\log p(r_t = \text{runLengths}[i] \mid x_{1:t})$, float64 `[R]`, normalised. */
  logPosterior: Tensor
  /** The conjugate statistics of each tracked run, aligned with `runLengths`. */
  stats: S
  /** The most probable run length. */
  map: number
  /** $\log p(x_t \mid x_{1:t-1})$, the one-step predictive log density of the last observation (0 at $t = 0$). */
  logPredictive: number
  /** $\log p(x_{1:t})$, summed over steps (the evidence, up to the mass discarded by pruning). */
  logEvidence: number
  /** Posterior mass discarded by pruning at the last step, before renormalising. */
  pruned: number
  /** True once every observation has been absorbed. */
  terminated: boolean
}

/** Options of the recursion. */
export interface BocpdOptions {
  /** The hazard (default 1/100: geometric gaps with mean 100). */
  hazard?: Hazard
  /** Drop run lengths whose posterior probability falls below this (default 0: exact). */
  threshold?: number
  /** Keep at most this many run lengths, the most probable (default $\infty$). */
  maxRuns?: number
}

/**
 * $\log \sum_i \exp(v_i)$ of plain numbers, by the tensor reduction (one definition of the stable form).
 *
 * @param v The numbers $v_i$.
 * @returns Their log-sum-exp ($-\infty$ for none).
 */
const logSumExp = (v: ArrayLike<number>): number => logsumexp(fromData(Float64Array.from(v))) as number

/**
 * The hazard of each run. A constant outside $[0, 1]$ throws `DomainError`; a function's values are not checked.
 *
 * @param hazard The constant hazard, or the function of the run length.
 * @param tau The run length $\tau = r_{t-1} + 1$ of each run.
 * @returns $H(\tau)$ for each run.
 */
function hazards(hazard: Hazard, tau: Float64Array): Float64Array {
  if (typeof hazard === 'number') {
    if (!(hazard >= 0 && hazard <= 1))
      throw new DomainError('bocpd', `bocpd: the hazard must be in [0, 1], got ${hazard}`)
    return new Float64Array(tau.length).fill(hazard)
  }
  return Float64Array.from(toFlat(hazard(fromData(tau))))
}

/**
 * The statistics of the runs kept by pruning.
 *
 * @param stats The statistics of every run, each field with a leading axis of one entry per run.
 * @param keep The indices of the runs to keep, ascending.
 * @returns New statistics with only those runs, in that order.
 */
function takeRuns<S extends RunStats>(stats: S, keep: readonly number[]): S {
  const idx = fromData(Int32Array.from(keep))
  return Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, take(v, idx) as Tensor])) as S
}

/**
 * The statistics with the prior's run put first: the run of length 0 after a changepoint.
 *
 * @param prior The prior's statistics, one run.
 * @param stats The statistics of the grown runs, with the same fields.
 * @returns New statistics with one more run, the prior at index 0.
 */
function prepend<S extends RunStats>(prior: S, stats: S): S {
  return Object.fromEntries(Object.entries(prior).map(([k, v]) => [k, concat([v, stats[k]], 0)])) as S
}

/**
 * The state before any observation: all mass at $r_0 = 0$, a changepoint just before $x_1$.
 *
 * @param model The segment model, whose prior is the one run.
 * @returns The initial state, with $t = 0$.
 *
 * @example One run of length 0, with all the mass
 * const s = bocpdInit(normalKnownVariance())
 * print('run lengths', s.runLengths, 'log posterior', s.logPosterior)
 */
export function bocpdInit<O, S extends RunStats>(model: ConjugatePredictive<O, S>): BocpdState<S> {
  return {
    t: 0,
    runLengths: fromData(Int32Array.of(0)),
    logPosterior: fromData(Float64Array.of(0)),
    stats: model.prior(),
    map: 0,
    logPredictive: 0,
    logEvidence: 0,
    pruned: 0,
    terminated: false,
  }
}

/**
 * One step of the recursion: absorb observation `x` into `state`. Pure; the lab calls it directly to feed values one
 * at a time (the algorithm `bocpd` steps through a fixed series with it). An invalid constant hazard throws
 * `DomainError`.
 *
 * @param model The segment model.
 * @param state The state before `x` (not modified).
 * @param x The next observation.
 * @param options The hazard (default 0.01) and the pruning (default none).
 * @returns The state after `x`; `terminated` is false (the caller decides when a series ends).
 *
 * @example A jump starts a new run
 * const model = normalKnownVariance({ priorSd: 10 })
 * let s = bocpdInit(model)
 * for (const x of [0, 0.2, -0.1]) s = bocpdUpdate(model, s, x, { hazard: 0.1 })
 * print('after three values near 0, MAP run length', s.map)
 * const after = bocpdUpdate(model, s, 8, { hazard: 0.1 })
 * print('after a jump to 8, MAP run length', after.map)
 * print('p(r | x) for r =', after.runLengths, ':', exp(after.logPosterior))
 */
export function bocpdUpdate<O, S extends RunStats>(
  model: ConjugatePredictive<O, S>,
  state: BocpdState<S>,
  x: O,
  options: BocpdOptions = {},
): BocpdState<S> {
  const { hazard = 0.01, threshold = 0, maxRuns = Infinity } = options
  const r = toFlat(state.runLengths)
  const logP = toFlat(state.logPosterior)
  const R = r.length
  const logPi = toFlat(model.predictive(state.stats, x).logProb(model.value(x)))
  const H = hazards(
    hazard,
    Float64Array.from(r, (v) => v + 1),
  )
  // Joint log p(rₜ, x₁:ₜ) − log p(x₁:ₜ₋₁): index 0 is the changepoint, index i + 1 run i grown by one.
  const joint = new Float64Array(R + 1)
  const reset = new Float64Array(R)
  for (let i = 0; i < R; i++) {
    const a = logP[i] + logPi[i]
    joint[i + 1] = a + Math.log1p(-H[i])
    reset[i] = a + Math.log(H[i])
  }
  joint[0] = logSumExp(reset)
  const evidence = logSumExp(joint)
  for (let i = 0; i <= R; i++) joint[i] -= evidence
  const lengths = new Int32Array(R + 1)
  for (let i = 0; i < R; i++) lengths[i + 1] = r[i] + 1
  let stats = prepend(model.prior(), model.update(state.stats, x))

  // Pruning: keep runs above the threshold (always the most probable), then the `maxRuns` most probable.
  let best = 0
  for (let i = 1; i <= R; i++) if (joint[i] > joint[best]) best = i
  const logThreshold = threshold > 0 ? Math.log(threshold) : -Infinity
  let keep: number[] = []
  for (let i = 0; i <= R; i++) if (i === best || joint[i] >= logThreshold) keep.push(i)
  if (keep.length > maxRuns) {
    keep = [...keep]
      .sort((a, b) => joint[b] - joint[a])
      .slice(0, Math.max(1, maxRuns))
      .sort((a, b) => a - b)
  }
  let logPosterior = joint
  let runLengths = lengths
  let pruned = 0
  if (keep.length < R + 1) {
    const kept = Float64Array.from(keep, (i) => joint[i])
    const mass = logSumExp(kept)
    pruned = -Math.expm1(mass)
    logPosterior = kept.map((v) => v - mass)
    runLengths = Int32Array.from(keep, (i) => lengths[i])
    stats = takeRuns(stats, keep)
  }
  let map = 0
  for (let i = 1; i < logPosterior.length; i++) if (logPosterior[i] > logPosterior[map]) map = i
  return {
    t: state.t + 1,
    runLengths: fromData(runLengths),
    logPosterior: fromData(logPosterior),
    stats,
    map: runLengths[map],
    logPredictive: evidence,
    logEvidence: state.logEvidence + evidence,
    pruned,
    terminated: false,
  }
}

/**
 * Bayesian online changepoint detection over a fixed series, as an algorithm: step $t$ absorbs `data[t]` by
 * `bocpdUpdate`, and the run terminates after the last observation. `init` takes nothing.
 *
 * @param model The segment model.
 * @param data The series, one observation per step.
 * @param options The hazard (default 0.01) and the pruning, applied at every step.
 * @returns The algorithm; it takes no start and draws no random numbers.
 *
 * @example Step through a series with one change
 * const model = normalKnownVariance({ priorSd: 10 })
 * const data = [0, 0.1, -0.2, 5, 5.1, 4.9]
 * const s = run(bocpd(model, data, { hazard: 0.1 }), undefined, 100)
 * print('t =', s.t, 'terminated =', s.terminated)
 * print('MAP run length', s.map, '(the last three values are one segment)')
 */
export function bocpd<O, S extends RunStats>(
  model: ConjugatePredictive<O, S>,
  data: readonly O[],
  options: BocpdOptions = {},
): Algorithm<void, BocpdState<S>> {
  return {
    name: `bocpd(${model.name})`,
    init: () => ({ ...bocpdInit(model), terminated: data.length === 0 }),
    step: (state: BocpdState<S>) => {
      const next = bocpdUpdate(model, state, data[state.t], options)
      return { ...next, terminated: next.t >= data.length }
    },
  }
}

/**
 * The run-length posterior as a dense row $p(r_t = 0, \dots, \text{length} - 1 \mid x_{1:t})$ (untracked run
 * lengths are 0, and longer ones are left out).
 *
 * @param state The recursion's state.
 * @param length The number of run lengths in the row, from 0.
 * @returns A new array of `length` probabilities.
 *
 * @example The posterior after three observations
 * const model = normalKnownVariance({ priorSd: 10 })
 * let s = bocpdInit(model)
 * for (const x of [0, 0.1, 6]) s = bocpdUpdate(model, s, x, { hazard: 0.1 })
 * print('p(r = 0 ... 4) =', runLengthRow(s, 5))
 */
export function runLengthRow(state: BocpdState, length: number): Float64Array {
  const out = new Float64Array(length)
  const r = toFlat(state.runLengths)
  const lp = toFlat(state.logPosterior)
  for (let i = 0; i < r.length; i++) if (r[i] < length) out[r[i]] = Math.exp(lp[i])
  return out
}

/**
 * The one-step forecast $p(x_{t+1} \mid x_{1:t}) = \sum_r p(x_{t+1} \mid r_t = r, x^{(r)})\, p(r_t = r \mid x_{1:t})$:
 * its mean and variance (by the law of total variance over the run-length mixture). If the model's predictive throws
 * (an autoregression given no regressors), the forecast is mean 0 and variance 1 rather than an error.
 *
 * @param model The segment model.
 * @param state The recursion's state.
 * @param next What the next value is conditioned on, for models that need it (its own value is not used).
 * @returns The forecast's mean and variance.
 *
 * @example The forecast mixes the old level with the prior
 * const model = normalKnownVariance({ priorSd: 10 })
 * let s = bocpdInit(model)
 * for (const x of [2, 2.1, 1.9, 2]) s = bocpdUpdate(model, s, x, { hazard: 0.1 })
 * print('forecast', bocpdForecast(model, s))
 */
export function bocpdForecast<O, S extends RunStats>(
  model: ConjugatePredictive<O, S>,
  state: BocpdState<S>,
  next?: O,
): { mean: number; variance: number } {
  try {
    const d = model.predictive(state.stats, next)
    const m = toFlat(d.mean())
    const v = toFlat(d.variance())
    const w = toFlat(state.logPosterior).map(Math.exp)
    let mean = 0
    let second = 0
    for (let i = 0; i < w.length; i++) {
      mean += w[i] * m[i]
      second += w[i] * (v[i] + m[i] * m[i])
    }
    return { mean, variance: second - mean * mean }
  } catch {
    return { mean: 0, variance: 1 }
  }
}

/**
 * The one-step predictive density
 * $p(x_{t+1} = v \mid x_{1:t}) = \sum_r p(x_{t+1} = v \mid r_t = r, x^{(r)})\, p(r_t = r \mid x_{1:t})$ at each value
 * $v$ of `values`: the run-length mixture that `bocpdForecast` summarises by its mean and variance, for drawing. For a
 * count model the values should be integers (the result is then a probability mass). If the model's predictive throws,
 * the result is all zeros rather than an error.
 *
 * @param model The segment model.
 * @param state The recursion's state.
 * @param values The values $v$ at which to evaluate the density.
 * @param next What the next value is conditioned on, for models that need it (its own value is not used).
 * @returns A new array of the density at each value.
 *
 * @example A mixture of a narrow and a wide bump
 * const model = normalKnownVariance({ priorSd: 10 })
 * let s = bocpdInit(model)
 * for (const x of [2, 2.1, 1.9, 2]) s = bocpdUpdate(model, s, x, { hazard: 0.1 })
 * print('p(x = -10, 0, 2, 10) =', bocpdPredictiveDensity(model, s, [-10, 0, 2, 10]))
 */
export function bocpdPredictiveDensity<O, S extends RunStats>(
  model: ConjugatePredictive<O, S>,
  state: BocpdState<S>,
  values: ArrayLike<number>,
  next?: O,
): Float64Array {
  try {
    const d = model.predictive(state.stats, next)
    const lw = toFlat(state.logPosterior)
    const out = new Float64Array(values.length)
    const terms = new Float64Array(lw.length)
    for (let j = 0; j < values.length; j++) {
      const lp = toFlat(d.logProb(values[j]) as Tensor)
      for (let i = 0; i < lw.length; i++) terms[i] = lw[i] + (lp.length === 1 ? lp[0] : lp[i])
      out[j] = Math.exp(logSumExp(terms))
    }
    return out
  } catch {
    return new Float64Array(values.length)
  }
}

/**
 * The posterior probability $\pr(\text{lo} \le r_t < \text{hi} \mid x_{1:t})$ of a range of run lengths, e.g. of a
 * change in the last $w$ steps (`runLengthMass(state, 0, w)`).
 *
 * @param state The recursion's state.
 * @param lo The smallest run length counted.
 * @param hi The first run length not counted (default all above `lo`).
 * @returns The probability, at most 1.
 *
 * @example How likely is a change in the last two steps?
 * const model = normalKnownVariance({ priorSd: 10 })
 * let s = bocpdInit(model)
 * for (const x of [0, 0.1, -0.1, 0]) s = bocpdUpdate(model, s, x, { hazard: 0.1 })
 * print('before the jump:', runLengthMass(s, 0, 2))
 * const after = bocpdUpdate(model, s, 6, { hazard: 0.1 })
 * print('after the jump:', runLengthMass(after, 0, 2))
 */
export function runLengthMass(state: BocpdState, lo: number, hi = Infinity): number {
  const r = toFlat(state.runLengths)
  const lp = toFlat(state.logPosterior)
  let mass = 0
  for (let i = 0; i < r.length; i++) if (r[i] >= lo && r[i] < hi) mass += Math.exp(lp[i])
  return Math.min(1, mass)
}

/** The result of `detectChangepoints`. */
export interface ChangepointDetection {
  /**
   * $p(r_t = r \mid x_{1:t})$ for $t = 1, \dots, n$ (rows) and $r = 0, \dots, \text{maxRunLength}$ (columns), float64
   * `[n, maxRunLength + 1]`.
   */
  posterior: Tensor
  /** The most probable run length after each observation, int32 `[n]`. */
  map: Tensor
  /** One-step forecast means of each observation from the ones before it, float64 `[n]`. */
  forecastMean: Tensor
  /** One-step forecast variances, likewise. */
  forecastVariance: Tensor
  /** $\log p(x_{1:n})$. */
  logEvidence: number
  /**
   * Estimated changepoints: indices t (0-based) where a segment begins, by backtracking the MAP run lengths
   * (`mapChangepoints`).
   */
  changepoints: number[]
}

/**
 * Run the recursion over a whole series and collect the dense run-length posterior, the MAP run lengths, the one-step
 * forecasts and the changepoints they imply.
 *
 * @param model The segment model.
 * @param data The series.
 * @param options The hazard and pruning, as `bocpdUpdate` takes them, and `maxRunLength`, the largest run length kept
 *   in the dense posterior (default $n$; it only caps the columns, not the recursion).
 * @returns The dense posterior, the MAP run lengths, the forecasts, the log evidence and the changepoints.
 *
 * @example A shift in the mean is found where it happens
 * const model = normalKnownVariance({ priorSd: 10 })
 * const data = [0, 0.1, -0.2, 0.1, 5, 5.2, 4.9, 5.1]
 * const d = detectChangepoints(model, data, { hazard: 0.1 })
 * print('changepoints', d.changepoints)
 * print('MAP run lengths', d.map)
 * print('forecast means', d.forecastMean)
 */
export function detectChangepoints<O, S extends RunStats>(
  model: ConjugatePredictive<O, S>,
  data: readonly O[],
  options: BocpdOptions & { maxRunLength?: number } = {},
): ChangepointDetection {
  const n = data.length
  const width = (options.maxRunLength ?? n) + 1
  const posterior = new Float64Array(n * width)
  const map = new Int32Array(n)
  const fm = new Float64Array(n)
  const fv = new Float64Array(n)
  let state = bocpdInit(model)
  for (let t = 0; t < n; t++) {
    const f = bocpdForecast(model, state, data[t])
    fm[t] = f.mean
    fv[t] = f.variance
    state = bocpdUpdate(model, state, data[t], options)
    posterior.set(runLengthRow(state, width), t * width)
    map[t] = state.map
  }
  return {
    posterior: fromData(posterior, [n, width]),
    map: fromData(map),
    forecastMean: fromData(fm),
    forecastVariance: fromData(fv),
    logEvidence: state.logEvidence,
    changepoints: mapChangepoints(map),
  }
}

/**
 * Changepoints read from the MAP run lengths after observations $0, \dots, n - 1$ ($r_t$ counts $x_t$ itself), by
 * backtracking: the last segment begins at $s = n - \text{map}[n - 1]$, the one before it ends at $s - 1$ and begins
 * at $s - \text{map}[s - 1]$, and so on. Transient dips of the MAP run length that later runs overrule leave no
 * changepoint. Returns the segment starts after 0, ascending.
 *
 * @param map The MAP run length after each observation, as `detectChangepoints` or the states' `map` give it.
 * @returns The 0-based indices where a segment begins, ascending, without 0.
 *
 * @example Backtrack the segments
 * print('starts', mapChangepoints([1, 2, 3, 1, 2, 3]))
 * // A dip at index 3 that the final run length of 6 overrules:
 * print('starts', mapChangepoints([1, 2, 3, 1, 5, 6]))
 */
export function mapChangepoints(map: ArrayLike<number>): number[] {
  const starts: number[] = []
  let t = map.length - 1
  while (t >= 0) {
    const start = t + 1 - Math.min(map[t], t + 1)
    if (start <= 0) break
    starts.push(start)
    t = start - 1
  }
  return starts.reverse()
}

// ── Conjugate segment models ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * Numbers as a float64 vector.
 *
 * @param v The numbers, copied.
 * @returns A new rank-1 tensor.
 */
const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v))
/**
 * One number as a length-1 vector: a statistic of the single prior run.
 *
 * @param v The number.
 * @returns A rank-1 tensor of length 1.
 */
const one = (v: number): Tensor => vec([v])

/**
 * Throws `DomainError` unless `v` is positive and finite.
 *
 * @param what The parameter's name with its model's, for the error message.
 * @param v The value to check.
 */
function positive(what: string, v: number): void {
  if (!(v > 0 && Number.isFinite(v))) throw new DomainError('positive', `${what} must be positive and finite, got ${v}`)
}

/**
 * A normal segment with known standard deviation $\sigma$ and a normal prior $\mu \sim \Gauss(\mu_0, \sigma_0^2)$ on
 * its mean. A run's posterior is $\Gauss(\text{mean}, 1/\text{precision})$; the predictive is
 * $\Gauss(\text{mean}, 1/\text{precision} + \sigma^2)$. A non-positive standard deviation throws `DomainError`.
 *
 * @param options The prior and the noise.
 * @param options.mean The prior mean $\mu_0$ of a segment's level (default 0).
 * @param options.priorSd The prior standard deviation $\sigma_0$ of a segment's level (default 1).
 * @param options.sd The known noise standard deviation $\sigma$ within a segment (default 1).
 * @returns The segment model; its statistics are each run's posterior `mean` and `precision`.
 *
 * @example One observation halves the uncertainty
 * const m = normalKnownVariance({ mean: 0, priorSd: 1, sd: 1 })
 * const s = m.update(m.prior(), 2)
 * print('posterior mean', s.mean, 'precision', s.precision)
 * const next = m.predictive(s)
 * print('predictive mean', next.mean(), 'variance', next.variance())
 */
export function normalKnownVariance({
  mean = 0,
  priorSd = 1,
  sd = 1,
}: { mean?: number; priorSd?: number; sd?: number } = {}): ConjugatePredictive<
  number,
  { mean: Tensor; precision: Tensor }
> {
  positive('normalKnownVariance: priorSd', priorSd)
  positive('normalKnownVariance: sd', sd)
  const noise = 1 / (sd * sd)
  return {
    name: 'normal, known variance',
    prior: () => ({ mean: one(mean), precision: one(1 / (priorSd * priorSd)) }),
    predictive: (s) => {
      const m = toFlat(s.mean)
      const p = toFlat(s.precision)
      return Normal(vec(m), vec(Array.from(p, (v) => Math.sqrt(1 / v + sd * sd))))
    },
    update: (s, x) => {
      const m = toFlat(s.mean)
      const p = toFlat(s.precision)
      const p2 = Float64Array.from(p, (v) => v + noise)
      return { mean: vec(Array.from(m, (v, i) => (v * p[i] + x * noise) / p2[i])), precision: vec(p2) }
    },
    value: (x) => x,
  }
}

/**
 * A normal segment with unknown mean and precision under a normal–gamma prior:
 * $\lambda \sim \operatorname{Gamma}(\alpha_0, \beta_0)$ (rate $\beta_0$),
 * $\mu \mid \lambda \sim \Gauss(\mu_0, 1/(\kappa_0\lambda))$. The predictive is Student t with $2\alpha$ degrees of
 * freedom, location $\mu$ and scale $\sqrt{\beta(\kappa + 1)/(\alpha\kappa)}$ (Murphy, 2007, "Conjugate Bayesian
 * analysis of the Gaussian distribution", §3.6). A non-positive `kappa`, `alpha` or `beta` throws `DomainError`.
 *
 * @param options The prior.
 * @param options.mean The prior mean $\mu_0$ (default 0).
 * @param options.kappa The prior's pseudo-count for the mean, $\kappa_0$ (default 1).
 * @param options.alpha The shape $\alpha_0$ of the gamma prior on the precision (default 1).
 * @param options.beta The rate $\beta_0$ of the gamma prior on the precision (default 1).
 * @returns The segment model; its statistics are each run's `mean`, `kappa`, `alpha` and `beta`.
 *
 * @example One observation updates all four statistics
 * const m = normalGamma()
 * const s = m.update(m.prior(), 2)
 * print('mean', s.mean, 'kappa', s.kappa, 'alpha', s.alpha, 'beta', s.beta)
 * print('predictive mean', m.predictive(s).mean())
 */
export function normalGamma({
  mean = 0,
  kappa = 1,
  alpha = 1,
  beta = 1,
}: { mean?: number; kappa?: number; alpha?: number; beta?: number } = {}): ConjugatePredictive<
  number,
  { mean: Tensor; kappa: Tensor; alpha: Tensor; beta: Tensor }
> {
  positive('normalGamma: kappa', kappa)
  positive('normalGamma: alpha', alpha)
  positive('normalGamma: beta', beta)
  return {
    name: 'normal–gamma',
    prior: () => ({ mean: one(mean), kappa: one(kappa), alpha: one(alpha), beta: one(beta) }),
    predictive: (s) => {
      const k = toFlat(s.kappa)
      const a = toFlat(s.alpha)
      const b = toFlat(s.beta)
      return StudentT(
        vec(Array.from(a, (v) => 2 * v)),
        vec(toFlat(s.mean)),
        vec(Array.from(a, (v, i) => Math.sqrt((b[i] * (k[i] + 1)) / (v * k[i])))),
      )
    },
    update: (s, x) => {
      const m = toFlat(s.mean)
      const k = toFlat(s.kappa)
      const a = toFlat(s.alpha)
      const b = toFlat(s.beta)
      return {
        mean: vec(Array.from(m, (v, i) => (k[i] * v + x) / (k[i] + 1))),
        kappa: vec(Array.from(k, (v) => v + 1)),
        alpha: vec(Array.from(a, (v) => v + 0.5)),
        beta: vec(Array.from(b, (v, i) => v + (k[i] * (x - m[i]) ** 2) / (2 * (k[i] + 1)))),
      }
    },
    value: (x) => x,
  }
}

/**
 * Poisson counts with a gamma prior on the rate, $\lambda \sim \operatorname{Gamma}(a, b)$ ($a$ = `shape`, $b$ =
 * `rate`). A run with count sum $s$ over $n$ observations has posterior $\operatorname{Gamma}(a + s, b + n)$; the
 * predictive is negative binomial with $r$ the posterior shape and success probability $p$ = rate/(rate + 1) of the
 * posterior, counting failures (mean shape/rate). A non-positive `shape` or `rate` throws `DomainError`.
 *
 * @param options The prior.
 * @param options.shape The shape $a$ of the gamma prior on the rate (default 1).
 * @param options.rate The rate $b$ of the gamma prior on the rate (default 1).
 * @returns The segment model; its statistics are each run's posterior `shape` and `rate`.
 *
 * @example A count of 3 moves the rate's mean from 1 to 2
 * const m = poissonGamma({ shape: 1, rate: 1 })
 * const s = m.update(m.prior(), 3)
 * print('shape', s.shape, 'rate', s.rate)
 * print('predictive mean', m.predictive(s).mean())
 */
export function poissonGamma({ shape = 1, rate = 1 }: { shape?: number; rate?: number } = {}): ConjugatePredictive<
  number,
  { shape: Tensor; rate: Tensor }
> {
  positive('poissonGamma: shape', shape)
  positive('poissonGamma: rate', rate)
  return {
    name: 'Poisson–gamma',
    prior: () => ({ shape: one(shape), rate: one(rate) }),
    predictive: (s) => {
      const b = toFlat(s.rate)
      return NegativeBinomial(vec(toFlat(s.shape)), vec(Array.from(b, (v) => v / (v + 1))))
    },
    update: (s, x) => ({
      shape: vec(Array.from(toFlat(s.shape), (v) => v + x)),
      rate: vec(Array.from(toFlat(s.rate), (v) => v + 1)),
    }),
    value: (x) => x,
  }
}

/**
 * Bernoulli trials (0 or 1) with a beta prior $p \sim \operatorname{Beta}(\alpha, \beta)$; the predictive is
 * $\Bern(\alpha/(\alpha + \beta))$. A non-positive `alpha` or `beta` throws `DomainError`.
 *
 * @param options The prior.
 * @param options.alpha The prior count of ones, $\alpha$ (default 1).
 * @param options.beta The prior count of zeros, $\beta$ (default 1).
 * @returns The segment model; its statistics are each run's posterior `alpha` and `beta`.
 *
 * @example Two ones and a zero
 * const m = betaBernoulli()
 * let s = m.prior()
 * for (const x of [1, 1, 0]) s = m.update(s, x)
 * print('alpha', s.alpha, 'beta', s.beta, 'p(next = 1)', m.predictive(s).mean())
 */
export function betaBernoulli({ alpha = 1, beta = 1 }: { alpha?: number; beta?: number } = {}): ConjugatePredictive<
  number,
  { alpha: Tensor; beta: Tensor }
> {
  positive('betaBernoulli: alpha', alpha)
  positive('betaBernoulli: beta', beta)
  return {
    name: 'beta–Bernoulli',
    prior: () => ({ alpha: one(alpha), beta: one(beta) }),
    predictive: (s) => {
      const a = toFlat(s.alpha)
      const b = toFlat(s.beta)
      return Bernoulli(vec(Array.from(a, (v, i) => v / (v + b[i]))))
    },
    update: (s, x) => ({
      alpha: vec(Array.from(toFlat(s.alpha), (v) => v + x)),
      beta: vec(Array.from(toFlat(s.beta), (v) => v + 1 - x)),
    }),
    value: (x) => x,
  }
}

/** One observation of an autoregression: the value `y` and its regressors `z` (lags, and 1 for an intercept). */
export interface Regressed {
  /** The observed value. */
  readonly y: number
  /** Its regressors: the lags, newest first, then 1 for an intercept. */
  readonly z: readonly number[]
}

/**
 * A series as autoregression observations: for $t \ge p$, $y = x[t]$ and $\zvec = (x[t - 1], \dots, x[t - p])$, with a
 * trailing 1 when `intercept` is true (default). The first $p$ values have no full lag vector and are dropped, so
 * observation $i$ is $x[i + p]$.
 *
 * @param x The series.
 * @param p The autoregressive order: the number of lags.
 * @param options Whether to add an intercept.
 * @param options.intercept Append a 1 to every regressor vector (default true).
 * @returns The $\max(0, T - p)$ observations, each with $p$ (or $p + 1$) regressors.
 *
 * @example Two lags and an intercept
 * print(laggedObservations([1, 2, 3, 4], 2))
 */
export function laggedObservations(x: VectorLike, p: number, { intercept = true } = {}): Regressed[] {
  const v = isTensor(x) ? toFlat(x) : Array.from(x)
  const out: Regressed[] = []
  for (let t = p; t < v.length; t++) {
    const z = Array.from({ length: p }, (_, k) => v[t - 1 - k])
    if (intercept) z.push(1)
    out.push({ y: v[t], z })
  }
  return out
}

/**
 * A linear-Gaussian regression segment, $y = \zvec^\top\wvec + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$,
 * with the normal–inverse-gamma prior $\wvec \mid \sigma^2 \sim \Gauss(\zeros, \sigma^2\Vmat_0)$,
 * $\Vmat_0 = s^2\Imat$ ($s$ = `priorScale`), and $\sigma^2 \sim \operatorname{InvGamma}(\alpha, \beta)$. With lagged
 * values as regressors (`laggedObservations`) it detects switches between autoregressive regimes. Each run keeps the
 * posterior mean $\wvec$ and scaled covariance $\Vmat$ by recursive least squares; the predictive is Student t with
 * $2\alpha$ degrees of freedom, location $\zvec^\top\wvec$ and scale $\sqrt{\beta(1 + \zvec^\top\Vmat\zvec)/\alpha}$
 * (Bishop, 2006, "Pattern Recognition and Machine Learning", §3.3 and exercise 3.12). The predictive needs the next
 * observation's regressors (it throws `TypeError` without them). Invalid options throw `DomainError`.
 *
 * @param options The size of the regression and its prior.
 * @param options.dimension The number of regressors per observation (lags, plus one for an intercept); a positive
 *   integer.
 * @param options.priorScale The prior standard deviation $s$ of each weight, in units of the noise (default 1).
 * @param options.alpha The shape $\alpha$ of the inverse-gamma prior on the noise variance (default 1).
 * @param options.beta The scale $\beta$ of the inverse-gamma prior on the noise variance (default 1).
 * @returns The segment model; its statistics are each run's `w` (`[R, p]`), `V` (`[R, p, p]`), `alpha` and `beta`.
 *
 * @example Learn a doubling series' coefficient
 * // x[t] = 2 x[t - 1]: one lag and an intercept, so the weights should approach (2, 0).
 * const m = regressionNormalGamma({ dimension: 2, priorScale: 10 })
 * let s = m.prior()
 * for (const o of laggedObservations([1, 2, 4, 8, 16, 32], 1)) s = m.update(s, o)
 * print('w =', s.w)
 * print('predictive mean after 32:', m.predictive(s, { y: NaN, z: [32, 1] }).mean())
 */
export function regressionNormalGamma({
  dimension,
  priorScale = 1,
  alpha = 1,
  beta = 1,
}: {
  /** Regressors per observation (lags, plus one for an intercept). */
  dimension: number
  priorScale?: number
  alpha?: number
  beta?: number
}): ConjugatePredictive<Regressed, { w: Tensor; V: Tensor; alpha: Tensor; beta: Tensor }> {
  const p = dimension
  if (!(Number.isInteger(p) && p >= 1))
    throw new DomainError('regressionNormalGamma', `regressionNormalGamma: dimension must be ≥ 1, got ${p}`)
  positive('regressionNormalGamma: priorScale', priorScale)
  positive('regressionNormalGamma: alpha', alpha)
  positive('regressionNormalGamma: beta', beta)
  const moments = (s: { w: Tensor; V: Tensor }, z: readonly number[]) => {
    const w = toFlat(s.w)
    const V = toFlat(s.V)
    const R = w.length / p
    const loc = new Float64Array(R)
    const quad = new Float64Array(R)
    const Vz = new Float64Array(R * p)
    for (let r = 0; r < R; r++) {
      let l = 0
      for (let i = 0; i < p; i++) {
        l += z[i] * w[r * p + i]
        let acc = 0
        for (let j = 0; j < p; j++) acc += V[r * p * p + i * p + j] * z[j]
        Vz[r * p + i] = acc
      }
      let q = 0
      for (let i = 0; i < p; i++) q += z[i] * Vz[r * p + i]
      loc[r] = l
      quad[r] = q
    }
    return { w, V, R, loc, quad, Vz }
  }
  return {
    name: 'regression normal–inverse-gamma',
    prior: () => {
      const V = new Float64Array(p * p)
      for (let i = 0; i < p; i++) V[i * p + i] = priorScale * priorScale
      return { w: fromData(new Float64Array(p), [1, p]), V: fromData(V, [1, p, p]), alpha: one(alpha), beta: one(beta) }
    },
    predictive: (s, x) => {
      if (!x) throw new TypeError('regressionNormalGamma: the predictive needs the regressors of the next observation')
      const { loc, quad } = moments(s, x.z)
      const a = toFlat(s.alpha)
      const b = toFlat(s.beta)
      return StudentT(
        vec(Array.from(a, (v) => 2 * v)),
        vec(loc),
        vec(Array.from(a, (v, r) => Math.sqrt((b[r] * (1 + quad[r])) / v))),
      )
    },
    update: (s, x) => {
      const { w, V, R, loc, quad, Vz } = moments(s, x.z)
      const w2 = new Float64Array(R * p)
      const V2 = new Float64Array(R * p * p)
      const b = toFlat(s.beta)
      const b2 = new Float64Array(R)
      for (let r = 0; r < R; r++) {
        const c = 1 + quad[r]
        const e = x.y - loc[r]
        for (let i = 0; i < p; i++) {
          w2[r * p + i] = w[r * p + i] + (Vz[r * p + i] * e) / c
          for (let j = 0; j < p; j++)
            V2[r * p * p + i * p + j] = V[r * p * p + i * p + j] - (Vz[r * p + i] * Vz[r * p + j]) / c
        }
        b2[r] = b[r] + (e * e) / (2 * c)
      }
      return {
        w: fromData(w2, [R, p]),
        V: fromData(V2, [R, p, p]),
        alpha: vec(Array.from(toFlat(s.alpha), (v) => v + 0.5)),
        beta: vec(b2),
      }
    },
    value: (x) => x.y,
  }
}
