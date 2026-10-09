/**
 * Sequential Monte Carlo: resampling schemes (Douc, Cappé & Moulines, 2005), the bootstrap particle filter (Gordon,
 * Salmond & Smith, 1993) and an adaptive tempered SMC sampler for static targets (Del Moral, Doucet & Jasra, 2006).
 *
 * Particles are stored row-major, $N \times d$; weights are kept as normalised log-weights, and both samplers carry a
 * running estimate of the log evidence.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { importanceEffectiveSampleSize } from 'aifn-compute/probability/stats'
import { fromData, logsumexp, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { VectorLike } from './types'
import { data, mat, standardNormals, toF64, toNumber, vec, type F64 } from './util'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * $\log \sum_i \exp(a_i)$ of a working array ($-\infty$ when empty or all $-\infty$).
 *
 * @param a The values $a_i$.
 * @returns Their log-sum-exp, computed stably.
 */
const logSumExp = (a: F64): number => (a.length === 0 ? -Infinity : toNumber(logsumexp(vec(a))))

/**
 * Kish's effective sample size of log-weights (the one definition is in `aifn-compute/probability/stats`).
 *
 * @param logWeights The log-weights, normalised or not.
 * @returns $(\sum_i w_i)^2 / \sum_i w_i^2$, between 1 and the number of weights.
 */
const essOfLogWeights = (logWeights: ArrayLike<number>): number =>
  importanceEffectiveSampleSize(logWeights, { log: true })

/** A resampling scheme. All are unbiased: each particle $i$ is copied $Nw_i$ times in expectation. */
export type ResamplingScheme = 'multinomial' | 'stratified' | 'systematic' | 'residual'

/** The four resampling schemes, from most to least resampling noise (residual and systematic are close). */
export const resamplingSchemes: readonly ResamplingScheme[] = ['multinomial', 'stratified', 'residual', 'systematic']

/**
 * Ancestor indices by inverting the cdf of `w` at sorted uniforms `u` (ascending, in $[0, 1)$).
 *
 * @param w The normalised weights.
 * @param u The uniforms, ascending.
 * @returns For each uniform, the index $i$ with $\sum_{j < i} w_j \le u < \sum_{j \le i} w_j$ (the last index when
 *   rounding leaves $u$ above the total), so ascending.
 */
function invertSorted(w: F64, u: F64): Int32Array {
  const out = new Int32Array(u.length)
  let c = w[0]
  let i = 0
  for (let k = 0; k < u.length; k++) {
    while (u[k] >= c && i < w.length - 1) c += w[++i]
    out[k] = i
  }
  return out
}

/**
 * The weights divided by their sum. Throws `DomainError` for a negative or NaN weight, or a sum that is not positive
 * and finite.
 *
 * @param weights The weights, not necessarily normalised; copied.
 * @returns A new working array of weights summing to 1.
 */
function normalised(weights: VectorLike): F64 {
  const w = toF64(weights, 'resample')
  let total = 0
  for (let i = 0; i < w.length; i++) {
    if (!(w[i] >= 0)) throw new DomainError('resample', 'resample: weights must be non-negative numbers')
    total += w[i]
  }
  if (!(total > 0) || !Number.isFinite(total))
    throw new DomainError('resample', 'resample: weights must have a positive finite sum')
  return w.map((v) => v / total)
}

/**
 * Draw `n` ancestor indices (int32, ascending) in proportion to `weights` (need not be normalised) by one of four
 * schemes (Douc, Cappé & Moulines, 2005): `multinomial` inverts the cdf at $n$ independent sorted uniforms;
 * `stratified` at one uniform in each stratum $[k/n, (k + 1)/n)$; `systematic` at $(k + u)/n$ for a single $u$;
 * `residual` copies $\lfloor nw_i \rfloor$ of each particle and fills the rest multinomially from the remainders.
 * Throws `DomainError` for a negative or NaN weight, or weights whose sum is not positive and finite.
 *
 * @param s The stream the uniforms are drawn from.
 * @param weights The weights $w_i$ of the particles, one per particle; normalised here.
 * @param scheme The resampling scheme.
 * @param n The number of indices to draw; default the number of weights.
 * @returns $n$ ancestor indices into the weights, ascending.
 *
 * @example Every scheme on the same weights
 * // Ten draws from weights 0.1, 0.2, 0.7: about 1, 2 and 7 copies.
 * for (const scheme of resamplingSchemes) print(`${scheme}:`, resample(stream(1), [0.1, 0.2, 0.7], scheme, 10))
 *
 * @example Unnormalised weights, and how often a scheme misses its expectation
 * // Systematic resampling copies each particle floor(n w) or ceil(n w) times; multinomial can stray further.
 * const w = [1, 1, 1, 5]
 * for (const scheme of ['multinomial', 'systematic']) {
 *   const copies = [0, 1, 2, 3].map((seed) => toFlat(resample(stream(seed), w, scheme)).filter((i) => i === 3).length)
 *   print(`${scheme}: copies of particle 3 (expected 2.5) over four draws =`, copies)
 * }
 */
export function resample(s: Stream, weights: VectorLike, scheme: ResamplingScheme = 'systematic', n?: number): Tensor {
  const w = normalised(weights)
  const N = n ?? w.length
  let out: Int32Array
  if (scheme === 'multinomial') {
    // Sorted uniforms from normalised exponential spacings, O(n) (Devroye, 1986, §V.3).
    const e = new Float64Array(N + 1)
    let total = 0
    for (let k = 0; k <= N; k++) total += e[k] = -Math.log(1 - uniform(s))
    const u = new Float64Array(N)
    let acc = 0
    for (let k = 0; k < N; k++) u[k] = (acc += e[k]) / total
    out = invertSorted(w, u)
  } else if (scheme === 'stratified' || scheme === 'systematic') {
    const u0 = uniform(s)
    const u = Float64Array.from(
      { length: N },
      (_, k) => (k + (scheme === 'systematic' ? u0 : k === 0 ? u0 : uniform(s))) / N,
    )
    out = invertSorted(w, u)
  } else {
    const copies = new Int32Array(w.length)
    let kept = 0
    const residual = new Float64Array(w.length)
    for (let i = 0; i < w.length; i++) {
      copies[i] = Math.floor(N * w[i])
      kept += copies[i]
      residual[i] = N * w[i] - copies[i]
    }
    const rest = N - kept
    out = new Int32Array(N)
    let k = 0
    for (let i = 0; i < w.length; i++) for (let c = 0; c < copies[i]; c++) out[k++] = i
    if (rest > 0) {
      const extra = data(resample(s, residual, 'multinomial', rest))
      for (let j = 0; j < rest; j++) out[k++] = extra[j]
      out.sort()
    }
  }
  return fromData(out, [N])
}

// ---------------------------------------------------------------------------------------------------------------------
// The bootstrap particle filter.

/**
 * A state-space model for the particle filter: $\xvec_0 \sim p(\xvec_0)$,
 * $\xvec_t \sim p(\xvec_t \mid \xvec_{t-1})$, $\yvec_t \sim p(\yvec_t \mid \xvec_t)$, with $d$-dimensional
 * states. Observations may be numbers or vectors (whatever `logObservation` reads).
 */
export type StateSpaceModel<Y = number> = {
  /** The dimension $d$ of the state. */
  dim: number
  /** A draw of $\xvec_0$ ($d$ values, or a number when $d = 1$) from the stream it is given. */
  sampleInitial: (s: Stream) => VectorLike | number
  /** A draw of $\xvec_t$ given $\xvec_{t-1}$ = `x`, for time $t \ge 1$. */
  sampleTransition: (x: Vector, t: number, s: Stream) => VectorLike | number
  /** $\log p(\yvec_t \mid \xvec_t)$ at $\xvec_t$ = `x`. */
  logObservation: (y: Y, x: Vector, t: number) => number
}

/** The start of a particle filter: the observations $\yvec_0, \yvec_1, \dots, \yvec_{T-1}$. */
export type FilterStart<Y = number> = { observations: readonly Y[] }

/** The state of `particleFilter` after assimilating `t` observations. */
export interface ParticleFilterState<Y = number> extends Status {
  /** Observations assimilated so far (0 in the initial state); the last one was $\yvec_{t-1}$. */
  t: number
  /** The particles after propagation and before resampling ($N \times d$): the support of the weights below. */
  particles: Matrix
  /** Normalised log-weights of `particles`. */
  logWeights: Vector
  /** Normalised weights of `particles`. */
  weights: Vector
  /** Kish's ESS of the weights, $1/\sum_i w_i^2$. */
  ess: number
  /** Whether the particles were resampled after weighting (ESS below `threshold` $\times N$). */
  resampled: boolean
  /** Ancestor of each particle carried to the next step (int32; identity when not resampled). */
  ancestors: Tensor
  /** The particles carried to the next step (resampled or not), $N \times d$. */
  carried: Matrix
  /** The normalised log-weights of `carried` (uniform after resampling). */
  carriedLogWeights: Vector
  /** The filtering mean $\expect[\xvec_t \mid \yvec_{0:t}]$ ($d$ values), from the weighted particles. */
  mean: Vector
  /** The filtering variance $\var[\xvec_t \mid \yvec_{0:t}]$ of each coordinate ($d$ values). */
  variance: Vector
  /**
   * The log of the unbiased estimate of $p(\yvec_{0:t})$ (Del Moral, 2004): the sum of the log of each step's
   * weighted mean likelihood.
   */
  logEvidence: number
  /** The observations, as given at the start. */
  observations: readonly Y[]
  /** Set when every particle had zero likelihood for an observation; stops the runners. */
  diverged: boolean
}

/** Options for `particleFilter`. */
export type ParticleFilterOptions = {
  /** Number of particles $N$. Default 500. */
  particles?: number
  /** The resampling scheme. Default `'systematic'`. */
  resampling?: ResamplingScheme
  /**
   * Resample when ESS $<$ threshold $\times N$ (1 resamples whenever the weights are not all equal, 0 never).
   * Default 0.5.
   */
  threshold?: number
}

/**
 * The bootstrap particle filter (Gordon, Salmond & Smith, 1993): propagate each particle through the transition,
 * weight it by the likelihood of the new observation, and resample when the effective sample size falls below
 * threshold $\times N$ (adaptive resampling, Liu & Chen, 1995). Step $t$ assimilates $\yvec_t$ (drawing from its step
 * stream; step 0 weights the initial particles, drawn by `init`) and the run is done after the last observation.
 *
 * @param model The state-space model: initial draw, transition draw and observation log-likelihood.
 * @param options The number of `particles` $N$, the `resampling` scheme and the resampling `threshold`.
 * @returns The filter as an algorithm: start it from `{ observations }` and run it for as many steps as observations.
 *
 * @example A Gaussian random walk, against the Kalman filter
 * // x0 ~ N(0, 1), x_t = x_{t-1} + N(0, 1), y_t = x_t + N(0, 1): the Kalman filter is exact.
 * const ys = [0.5, 1.2, 0.8, 2.0, 2.5]
 * const model = {
 *   dim: 1,
 *   sampleInitial: (s) => normal(s),
 *   sampleTransition: (x, t, s) => x.data[0] + normal(s),
 *   logObservation: (y, x) => -0.5 * (y - x.data[0]) ** 2 - 0.5 * Math.log(2 * Math.PI),
 * }
 * const pf = run(particleFilter(model, { particles: 1000 }), { observations: ys }, ys.length)
 * let [m, P, logZ] = [0, 1, 0]
 * ys.forEach((y, t) => {
 *   if (t > 0) P += 1
 *   logZ += -0.5 * Math.log(2 * Math.PI * (P + 1)) - (y - m) ** 2 / (2 * (P + 1))
 *   const K = P / (P + 1)
 *   ;[m, P] = [m + K * (y - m), (1 - K) * P]
 * })
 * print('particle filter: mean', pf.mean, 'variance', pf.variance, 'log evidence', pf.logEvidence)
 * print('Kalman filter:   mean', m, 'variance', P, 'log evidence', logZ)
 *
 * @example The ESS through a run
 * const model = {
 *   dim: 1,
 *   sampleInitial: (s) => normal(s, 0, 3),
 *   sampleTransition: (x, t, s) => x.data[0] + normal(s, 0, 0.5),
 *   logObservation: (y, x) => -2 * (y - x.data[0]) ** 2,
 * }
 * // A wide prior and a sharp likelihood: the first observation collapses the weights.
 * const pf = particleFilter(model, { particles: 200 })
 * for (let t = 1; t <= 4; t++) {
 *   const s = run(pf, { observations: [0, 0.2, 0.1, 0.4] }, t)
 *   print(`after y${t - 1}: ESS =`, s.ess, 'resampled', s.resampled, 'mean', s.mean)
 * }
 */
export function particleFilter<Y = number>(
  model: StateSpaceModel<Y>,
  options: ParticleFilterOptions = {},
): Algorithm<FilterStart<Y>, ParticleFilterState<Y>> {
  const name = 'particle-filter'
  const { particles: N = 500, resampling = 'systematic', threshold = 0.5 } = options
  const d = model.dim
  const draw = (v: VectorLike | number) => toF64(v, name)
  return {
    name,
    init: ({ observations }, s) => {
      const init = child(s, 'initial')
      const x = new Float64Array(N * d)
      for (let i = 0; i < N; i++) x.set(draw(model.sampleInitial(child(init, i))), i * d)
      const logW = new Float64Array(N).fill(-Math.log(N))
      const moments = weightedMoments(x, logW, N, d)
      return {
        t: 0,
        particles: mat(x, N, d),
        logWeights: vec(logW),
        weights: vec(logW.map(Math.exp)),
        ess: N,
        resampled: false,
        ancestors: fromData(
          Int32Array.from({ length: N }, (_, i) => i),
          [N],
        ),
        carried: mat(x, N, d),
        carriedLogWeights: vec(logW),
        mean: vec(moments.mean),
        variance: vec(moments.variance),
        logEvidence: 0,
        observations,
        diverged: false,
      }
    },
    step: (s, ctx) => {
      const t = s.t
      const draws = ctx.stream
      const prev = data(s.carried)
      const prevLogW = data(s.carriedLogWeights)
      // The initial particles are already draws from p(x₀); later ones move through the transition.
      const x = new Float64Array(N * d)
      if (t === 0) x.set(prev)
      else {
        const move = child(draws, 'transition')
        for (let i = 0; i < N; i++)
          x.set(draw(model.sampleTransition(vec(prev.slice(i * d, (i + 1) * d)), t, child(move, i))), i * d)
      }
      const y = s.observations[t]
      const logLik = new Float64Array(N)
      const unnorm = new Float64Array(N)
      for (let i = 0; i < N; i++) {
        logLik[i] = model.logObservation(y, vec(x.slice(i * d, (i + 1) * d)), t)
        unnorm[i] = prevLogW[i] + logLik[i]
      }
      // p(yₜ | y₀:ₜ₋₁) ≈ Σ wᵢ p(yₜ | xᵢ) with the carried (normalised) weights.
      const increment = logSumExp(unnorm)
      const logW = unnorm.map((v) => v - increment)
      const ess = essOfLogWeights(logW)
      const moments = weightedMoments(x, logW, N, d)
      const resampled = Number.isFinite(increment) && ess < threshold * N
      let ancestors = Int32Array.from({ length: N }, (_, i) => i)
      let carried = x
      let carriedLogW = logW
      if (resampled) {
        ancestors = Int32Array.from(data(resample(child(draws, 'resample'), logW.map(Math.exp), resampling)))
        carried = new Float64Array(N * d)
        for (let i = 0; i < N; i++) carried.set(x.subarray(ancestors[i] * d, (ancestors[i] + 1) * d), i * d)
        carriedLogW = new Float64Array(N).fill(-Math.log(N))
      }
      return {
        ...s,
        t: t + 1,
        particles: mat(x, N, d),
        logWeights: vec(logW),
        weights: vec(logW.map(Math.exp)),
        ess,
        resampled,
        ancestors: fromData(ancestors, [N]),
        carried: mat(carried, N, d),
        carriedLogWeights: vec(carriedLogW),
        mean: vec(moments.mean),
        variance: vec(moments.variance),
        logEvidence: s.logEvidence + increment,
        diverged: !Number.isFinite(increment),
      }
    },
    done: (s) => s.t >= s.observations.length,
  }
}

/**
 * The weighted mean and variance of each coordinate of the particles.
 *
 * @param x The particles, row-major $N \times d$.
 * @param logW Their normalised log-weights ($N$ values).
 * @param N The number of particles.
 * @param d The dimension of each particle.
 * @returns `mean` and `variance` ($d$ values each), under the weights.
 */
function weightedMoments(x: F64, logW: F64, N: number, d: number) {
  const mean = new Float64Array(d)
  const variance = new Float64Array(d)
  for (let i = 0; i < N; i++) {
    const w = Math.exp(logW[i])
    for (let k = 0; k < d; k++) mean[k] += w * x[i * d + k]
  }
  for (let i = 0; i < N; i++) {
    const w = Math.exp(logW[i])
    for (let k = 0; k < d; k++) variance[k] += w * (x[i * d + k] - mean[k]) ** 2
  }
  return { mean, variance }
}

// ---------------------------------------------------------------------------------------------------------------------
// Tempered SMC.

/** A Bayesian model for tempered SMC: a prior to draw from, and the log prior and log-likelihood. */
export type TemperedModel = {
  /** The dimension $d$ of $\thetavec$. */
  dim: number
  /** A draw of $\thetavec$ from the prior ($d$ values, or a number when $d = 1$) from the stream it is given. */
  samplePrior: (s: Stream) => VectorLike | number
  /** $\log p(\thetavec)$, up to a constant (used only by the moves). */
  logPrior: (theta: Vector) => number
  /** $\log L(\thetavec)$, the log-likelihood; its constants count in the evidence. */
  logLikelihood: (theta: Vector) => number
}

/** The state of `temperedSmc`. */
export interface TemperedSmcState extends Status {
  /** Tempering steps taken (0 in the initial state). */
  t: number
  /**
   * The inverse temperature $\beta \in [0, 1]$ of the current target
   * $\pi_\beta \propto p(\thetavec) L(\thetavec)^\beta$.
   */
  beta: number
  /** Particles ($N \times d$) targeting $\pi_\beta$ after reweighting, resampling and moves. */
  particles: Matrix
  /** Their normalised log-weights: uniform, $-\log N$, since every step resamples. */
  logWeights: Vector
  /** ESS of the incremental weights $L^{\beta' - \beta}$ before resampling. */
  ess: number
  /** $\log Z_\beta / Z_0$, the running estimate of the log evidence ($\log Z_1$ at $\beta = 1$). */
  logEvidence: number
  /** Acceptance rate of the random-walk moves on the last step (NaN at $t = 0$). */
  acceptanceRate: number
  /** Set when every particle had zero incremental weight; stops the runners. */
  diverged: boolean
}

/** Options for `temperedSmc`. */
export type TemperedSmcOptions = {
  /** Number of particles $N$. Default 500. */
  particles?: number
  /** Choose each next $\beta$ so that the incremental ESS is this fraction of $N$ (bisection). Default 0.5. */
  targetEss?: number
  /** Random-walk Metropolis moves per particle per step. Default 5. */
  moves?: number
  /**
   * Proposal scale as a multiple of the resampled particles' standard deviation per coordinate. Default
   * $2.38/\sqrt{d}$.
   */
  scale?: number
  /** The resampling scheme. Default `'systematic'`. */
  resampling?: ResamplingScheme
}

/**
 * Adaptive tempered SMC (Del Moral, Doucet & Jasra, 2006; Jasra et al., 2011): start from prior draws, raise $\beta$
 * by bisection so that the incremental weights $L(\thetavec)^{\beta' - \beta}$ keep an ESS of `targetEss`
 * $\times N$, resample, and rejuvenate with random-walk Metropolis moves that leave $\pi_{\beta'}$ invariant. The
 * product of the mean incremental weights estimates the evidence $Z = \int p(\thetavec)L(\thetavec)\,d\thetavec$.
 * Step $t$ draws from its step stream; done at $\beta = 1$.
 *
 * @param model The model: prior draw, log prior and log-likelihood.
 * @param options The number of `particles` $N$, the `targetEss`, the `moves`, their `scale` and the `resampling`
 *   scheme.
 * @returns The sampler as an algorithm: start it with `undefined` and run it until `beta` reaches 1.
 *
 * @example A conjugate Gaussian mean, against the exact posterior and evidence
 * // theta ~ N(0, 1), y_i ~ N(theta, 1): the posterior is N(sum y / (n + 1), 1 / (n + 1)).
 * const ys = [1.2, 0.8, 1.5, 0.9, 1.1]
 * const n = ys.length
 * const model = {
 *   dim: 1,
 *   samplePrior: (s) => normal(s),
 *   logPrior: (t) => -0.5 * t.data[0] ** 2,
 *   logLikelihood: (t) => ys.reduce((a, y) => a - 0.5 * (y - t.data[0]) ** 2 - 0.5 * Math.log(2 * Math.PI), 0),
 * }
 * const s = run(temperedSmc(model, { particles: 300 }), undefined, 50)
 * const [sy, syy] = [ys.reduce((a, b) => a + b), ys.reduce((a, b) => a + b * b)]
 * print('tempering steps =', s.t, 'final beta =', s.beta)
 * print('posterior mean =', mean(s.particles), 'exact', sy / (n + 1))
 * print('posterior variance =', variance(s.particles), 'exact', 1 / (n + 1))
 * const exact = -0.5 * n * Math.log(2 * Math.PI) - 0.5 * Math.log(1 + n) - 0.5 * (syy - sy ** 2 / (n + 1))
 * print('log evidence =', s.logEvidence, 'exact', exact)
 */
export function temperedSmc(model: TemperedModel, options: TemperedSmcOptions = {}): Algorithm<void, TemperedSmcState> {
  const name = 'tempered-smc'
  const d = model.dim
  const { particles: N = 500, targetEss = 0.5, moves = 5, resampling = 'systematic' } = options
  const scale = options.scale ?? 2.38 / Math.sqrt(d)
  const row = (x: F64, i: number) => vec(x.slice(i * d, (i + 1) * d))
  return {
    name,
    init: (_start, s) => {
      const x = new Float64Array(N * d)
      for (let i = 0; i < N; i++) x.set(toF64(model.samplePrior(child(s, 'prior', i)), name), i * d)
      return {
        t: 0,
        beta: 0,
        particles: mat(x, N, d),
        logWeights: vec(new Float64Array(N).fill(-Math.log(N))),
        ess: N,
        logEvidence: 0,
        acceptanceRate: NaN,
        diverged: false,
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const x = Float64Array.from(data(s.particles))
      const logL = Float64Array.from({ length: N }, (_, i) => model.logLikelihood(row(x, i)))
      const essAt = (delta: number) => essOfLogWeights(logL.map((l) => delta * l))
      // Bisection for the largest Δβ whose incremental ESS stays above the target.
      let delta = 1 - s.beta
      if (essAt(delta) < targetEss * N) {
        let lo = 0
        let hi = delta
        for (let k = 0; k < 50; k++) {
          const mid = 0.5 * (lo + hi)
          if (essAt(mid) >= targetEss * N) lo = mid
          else hi = mid
        }
        delta = Math.max(lo, 1e-12)
      }
      const beta = Math.min(1, s.beta + delta)
      const inc = logL.map((l) => delta * l)
      const increment = logSumExp(inc) - Math.log(N)
      const logW = inc.map((v) => v - logSumExp(inc))
      const ess = essOfLogWeights(logW)
      const anc = data(resample(child(draws, 'resample'), logW.map(Math.exp), resampling))
      const y = new Float64Array(N * d)
      for (let i = 0; i < N; i++) y.set(x.subarray(anc[i] * d, (anc[i] + 1) * d), i * d)
      // Proposal scale from the particles' spread.
      const sd = new Float64Array(d)
      for (let k = 0; k < d; k++) {
        let m = 0
        let m2 = 0
        for (let i = 0; i < N; i++) {
          m += y[i * d + k]
          m2 += y[i * d + k] ** 2
        }
        m /= N
        sd[k] = Math.sqrt(Math.max(m2 / N - m * m, 1e-12)) * scale
      }
      const logTarget = (v: Vector) => model.logPrior(v) + beta * model.logLikelihood(v)
      let accepted = 0
      for (let i = 0; i < N; i++) {
        let xi = y.slice(i * d, (i + 1) * d)
        let li = logTarget(vec(xi))
        const u = child(draws, 'move', i)
        for (let m = 0; m < moves; m++) {
          const eps = standardNormals(child(u, m), d)
          const prop = xi.map((v, k) => v + sd[k] * eps[k])
          const lp = logTarget(vec(prop))
          if (Math.log(uniform(u)) < lp - li) {
            xi = prop
            li = lp
            accepted++
          }
        }
        y.set(xi, i * d)
      }
      return {
        ...s,
        t: s.t + 1,
        beta,
        particles: mat(y, N, d),
        logWeights: vec(new Float64Array(N).fill(-Math.log(N))),
        ess,
        logEvidence: s.logEvidence + increment,
        acceptanceRate: accepted / (N * moves),
        diverged: !Number.isFinite(increment),
      }
    },
    done: (s) => s.beta >= 1,
  }
}
