/**
 * Bayesian online changepoint detection (Adams and MacKay, 2007, "Bayesian Online Changepoint Detection",
 * arXiv:0710.3742): the exact forward recursion over the run length rₜ, the number of observations since the last
 * changepoint, for any segment model with a conjugate posterior.
 *
 * With a hazard H(τ) and each run's posterior predictive πₜ⁽ʳ⁾ = p(xₜ | rₜ₋₁ = r, x⁽ʳ⁾), every run either grows,
 *
 *   p(rₜ = r + 1, x₁:ₜ) = p(rₜ₋₁ = r, x₁:ₜ₋₁) πₜ⁽ʳ⁾ (1 − H(r + 1)),
 *
 * or all runs feed one changepoint,
 *
 *   p(rₜ = 0, x₁:ₜ) = Σᵣ p(rₜ₋₁ = r, x₁:ₜ₋₁) πₜ⁽ʳ⁾ H(r + 1),
 *
 * and dividing by the evidence p(xₜ | x₁:ₜ₋₁) gives the run-length posterior. Everything is kept in log space. Each run
 * carries the statistics of its conjugate posterior (a count and running sums), updated by one term per observation;
 * the run at r = 0 starts from the prior. Run lengths whose posterior mass falls below a threshold, or beyond the
 * `maxRuns` most probable, are discarded and the rest renormalised, which bounds the cost per step (Adams and MacKay,
 * §2.4).
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
  readonly name: string
  /** The prior's statistics, as a single run (every field has a leading axis of length 1). */
  prior(): S
  /**
   * The posterior predictive of the next observed value under each run: a batch of R. `x` supplies what the value is
   * conditioned on, when the model needs it (lags); its own value is not used.
   */
  predictive(stats: S, x?: O): Univariate<Tensor>
  /** The statistics of every run after absorbing `x`. */
  update(stats: S, x: O): S
  /** The observed value of an observation (the number itself for scalar models). */
  value(x: O): number
}

/**
 * The hazard: a changepoint probability per step, constant (memoryless, geometric gaps with mean 1/H), or a function
 * of the run length τ = rₜ₋₁ + 1 giving H(τ) for each run.
 */
export type Hazard = number | ((tau: Tensor) => Tensor)

/** The constant hazard of geometric gaps with mean `meanGap` λ: H = 1/λ (Adams and MacKay, §2.1). */
export function constantHazard(meanGap: number): number {
  if (!(meanGap >= 1))
    throw new DomainError('constantHazard', `constantHazard: the mean gap must be at least 1, got ${meanGap}`)
  return 1 / meanGap
}

// ── State ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The recursion's state after t observations. */
export interface BocpdState<S extends RunStats = RunStats> extends Status {
  /** Observations absorbed. */
  t: number
  /** The run lengths still tracked, int32 [R] (ascending; not contiguous once pruning has removed some). */
  runLengths: Tensor
  /** log p(rₜ = runLengths[i] | x₁:ₜ), float64 [R], normalised. */
  logPosterior: Tensor
  /** The conjugate statistics of each tracked run, aligned with `runLengths`. */
  stats: S
  /** The most probable run length. */
  map: number
  /** log p(xₜ | x₁:ₜ₋₁), the one-step predictive log density of the last observation (0 at t = 0). */
  logPredictive: number
  /** log p(x₁:ₜ), summed over steps (the evidence, up to the mass discarded by pruning). */
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
  /** Keep at most this many run lengths, the most probable (default ∞). */
  maxRuns?: number
}

/** log Σ exp(vᵢ) of plain numbers, by the tensor reduction (one definition of the stable form). */
const logSumExp = (v: ArrayLike<number>): number => logsumexp(fromData(Float64Array.from(v))) as number

function hazards(hazard: Hazard, tau: Float64Array): Float64Array {
  if (typeof hazard === 'number') {
    if (!(hazard >= 0 && hazard <= 1))
      throw new DomainError('bocpd', `bocpd: the hazard must be in [0, 1], got ${hazard}`)
    return new Float64Array(tau.length).fill(hazard)
  }
  return Float64Array.from(toFlat(hazard(fromData(tau))))
}

function takeRuns<S extends RunStats>(stats: S, keep: readonly number[]): S {
  const idx = fromData(Int32Array.from(keep))
  return Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, take(v, idx) as Tensor])) as S
}

function prepend<S extends RunStats>(prior: S, stats: S): S {
  return Object.fromEntries(Object.entries(prior).map(([k, v]) => [k, concat([v, stats[k]], 0)])) as S
}

/** The state before any observation: all mass at r₀ = 0, a changepoint just before x₁. */
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
 * at a time (the algorithm `bocpd` steps through a fixed series with it).
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
 * Bayesian online changepoint detection over a fixed series, as an algorithm: step t absorbs `data[t]`, and the run
 * terminates after the last observation. `init` takes nothing.
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

/** The run-length posterior as a dense row p(rₜ = 0 … length − 1 | x₁:ₜ) (untracked run lengths are 0). */
export function runLengthRow(state: BocpdState, length: number): Float64Array {
  const out = new Float64Array(length)
  const r = toFlat(state.runLengths)
  const lp = toFlat(state.logPosterior)
  for (let i = 0; i < r.length; i++) if (r[i] < length) out[r[i]] = Math.exp(lp[i])
  return out
}

/**
 * The one-step forecast p(xₜ₊₁ | x₁:ₜ) = Σᵣ p(xₜ₊₁ | rₜ = r, x⁽ʳ⁾) p(rₜ = r | x₁:ₜ): its mean and variance (by the
 * law of total variance over the run-length mixture). `next` supplies what the next value is conditioned on, for
 * models that need it.
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
 * The one-step predictive density p(xₜ₊₁ = v | x₁:ₜ) = Σᵣ p(xₜ₊₁ = v | rₜ = r, x⁽ʳ⁾) p(rₜ = r | x₁:ₜ) at each value
 * v of `values`: the run-length mixture that `bocpdForecast` summarises by its mean and variance, for drawing. For a
 * count model the values should be integers (the result is then a probability mass). `next` supplies what the next
 * value is conditioned on, for models that need it.
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

/** The posterior probability P(lo ≤ rₜ < hi | x₁:ₜ) of a range of run lengths, e.g. of a change in the last w steps. */
export function runLengthMass(state: BocpdState, lo: number, hi = Infinity): number {
  const r = toFlat(state.runLengths)
  const lp = toFlat(state.logPosterior)
  let mass = 0
  for (let i = 0; i < r.length; i++) if (r[i] >= lo && r[i] < hi) mass += Math.exp(lp[i])
  return Math.min(1, mass)
}

/** The result of `detectChangepoints`. */
export interface ChangepointDetection {
  /** p(rₜ = r | x₁:ₜ) for t = 1 … n (rows) and r = 0 … maxRunLength (columns), float64 [n, maxRunLength + 1]. */
  posterior: Tensor
  /** The most probable run length after each observation, int32 [n]. */
  map: Tensor
  /** One-step forecasts of each observation from the ones before it: means and variances, float64 [n]. */
  forecastMean: Tensor
  forecastVariance: Tensor
  /** log p(x₁:ₙ). */
  logEvidence: number
  /**
   * Estimated changepoints: indices t (0-based) where a segment begins, by backtracking the MAP run lengths
   * (`mapChangepoints`).
   */
  changepoints: number[]
}

/**
 * Run the recursion over a whole series and collect the dense run-length posterior, the MAP run lengths, the one-step
 * forecasts and the changepoints they imply. `maxRunLength` (default n) caps the posterior's columns.
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
 * Changepoints read from the MAP run lengths after observations 0 … n − 1 (rₜ counts xₜ itself), by backtracking:
 * the last segment begins at s = n − map[n − 1], the one before it ends at s − 1 and begins at s − map[s − 1], and so
 * on. Transient dips of the MAP run length that later runs overrule leave no changepoint. Returns the segment starts
 * after 0, ascending.
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

const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v))
const one = (v: number): Tensor => vec([v])

function positive(what: string, v: number): void {
  if (!(v > 0 && Number.isFinite(v))) throw new DomainError('positive', `${what} must be positive and finite, got ${v}`)
}

/**
 * A normal segment with known standard deviation σ and a normal prior μ ~ N(μ₀, σ₀²) on its mean. A run's posterior
 * is N(mean, 1/precision); the predictive is N(mean, 1/precision + σ²).
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
 * A normal segment with unknown mean and precision under a normal–gamma prior: λ ~ Gamma(α₀, β₀) (rate β₀), μ | λ ~
 * N(μ₀, 1/(κ₀λ)). The predictive is Student t with 2α degrees of freedom, location μ and scale √(β(κ + 1)/(ακ))
 * (Murphy, 2007, "Conjugate Bayesian analysis of the Gaussian distribution", §3.6).
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
 * Poisson counts with a gamma prior on the rate, λ ~ Gamma(shape, rate). A run with count sum s over r observations has
 * posterior Gamma(shape + s, rate + r); the predictive is negative binomial with r = shape and p = rate/(rate + 1),
 * counting failures.
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

/** Bernoulli trials (0 or 1) with a beta prior p ~ Beta(α, β); the predictive is Bernoulli(α/(α + β)). */
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
  readonly y: number
  readonly z: readonly number[]
}

/**
 * A series as autoregression observations: for t ≥ p, y = x[t] and z = (x[t − 1], …, x[t − p]), with a trailing 1
 * when `intercept` is true (default). The first p values have no full lag vector and are dropped, so observation i is
 * x[i + p].
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
 * A linear-Gaussian regression segment, y = zᵀw + ε, ε ~ N(0, σ²), with the normal–inverse-gamma prior w | σ² ~
 * N(0, σ² V₀), V₀ = priorScale² I, and σ² ~ InvGamma(α, β). With lagged values as regressors (`laggedObservations`)
 * it detects switches between autoregressive regimes. Each run keeps the posterior mean w and scaled covariance V by
 * recursive least squares; the predictive is Student t with 2α degrees of freedom, location zᵀw and scale
 * √(β(1 + zᵀVz)/α) (Bishop, 2006, "Pattern Recognition and Machine Learning", §3.3 and exercise 3.12).
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
