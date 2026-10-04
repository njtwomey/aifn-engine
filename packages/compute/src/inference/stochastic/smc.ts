/**
 * Sequential Monte Carlo: resampling schemes (Douc, Cappé & Moulines, 2005), the bootstrap particle filter (Gordon,
 * Salmond & Smith, 1993) and an adaptive tempered SMC sampler for static targets (Del Moral, Doucet & Jasra, 2006).
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { importanceEffectiveSampleSize } from 'aifn-compute/probability/stats'
import { fromData, logsumexp, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { VectorLike } from './types'
import { data, mat, standardNormals, toF64, toNumber, vec, type F64 } from './util'
import { DomainError } from 'aifn-compute/foundation/errors'

/** log Σ exp(aᵢ) of a working array (−Infinity when empty or all −Infinity). */
const logSumExp = (a: F64): number => (a.length === 0 ? -Infinity : toNumber(logsumexp(vec(a))))

/** Kish's effective sample size of log-weights (the one definition is in `aifn-compute/probability/stats`). */
const essOfLogWeights = (logWeights: ArrayLike<number>): number =>
  importanceEffectiveSampleSize(logWeights, { log: true })

/** A resampling scheme. All are unbiased: each particle i is copied Nwᵢ times in expectation. */
export type ResamplingScheme = 'multinomial' | 'stratified' | 'systematic' | 'residual'

/** The four resampling schemes, from most to least resampling noise (residual and systematic are close). */
export const resamplingSchemes: readonly ResamplingScheme[] = ['multinomial', 'stratified', 'residual', 'systematic']

/** Ancestor indices by inverting the cdf of `w` at sorted uniforms `u` (ascending, in [0, 1)). */
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
 * schemes (Douc, Cappé & Moulines, 2005): `multinomial` inverts the cdf at n independent sorted uniforms;
 * `stratified` at one uniform in each stratum [k/n, (k + 1)/n); `systematic` at (k + u)/n for a single u; `residual`
 * copies ⌊nwᵢ⌋ of each particle and fills the rest multinomially from the remainders. Default n = weights.length.
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
 * A state-space model for the particle filter: x₀ ~ p(x₀), xₜ ~ p(xₜ | xₜ₋₁), yₜ ~ p(yₜ | xₜ), with d-dimensional
 * states. Observations may be numbers or vectors (whatever `logObservation` reads).
 */
export type StateSpaceModel<Y = number> = {
  dim: number
  sampleInitial: (s: Stream) => VectorLike | number
  /** A draw of xₜ given xₜ₋₁ = `x`, for time t ≥ 1. */
  sampleTransition: (x: Vector, t: number, s: Stream) => VectorLike | number
  /** log p(yₜ | xₜ = x). */
  logObservation: (y: Y, x: Vector, t: number) => number
}

/** The start of a particle filter: the observations y₀, y₁, …, y_{T−1}. */
export type FilterStart<Y = number> = { observations: readonly Y[] }

/** The state of `particleFilter` after assimilating `t` observations. */
export interface ParticleFilterState<Y = number> extends Status {
  /** Observations assimilated so far (0 in the initial state); the last one was y_{t−1}. */
  t: number
  /** The particles after propagation and before resampling (N×d): the support of the weights below. */
  particles: Matrix
  /** Normalised log-weights of `particles`. */
  logWeights: Vector
  /** Normalised weights of `particles`. */
  weights: Vector
  /** Kish's ESS of the weights, 1/Σwᵢ². */
  ess: number
  /** Whether the particles were resampled after weighting (ESS below threshold × N). */
  resampled: boolean
  /** Ancestor of each particle carried to the next step (int32; identity when not resampled). */
  ancestors: Tensor
  /** The particles carried to the next step (resampled or not) and their normalised log-weights. */
  carried: Matrix
  carriedLogWeights: Vector
  /** Filtering mean and variance E[xₜ | y₀:ₜ], Var[xₜ | y₀:ₜ] (per coordinate). */
  mean: Vector
  variance: Vector
  /** The unbiased estimate of log p(y₀:ₜ) (Del Moral, 2004). */
  logEvidence: number
  observations: readonly Y[]
  diverged: boolean
}

/** Options for `particleFilter`. */
export type ParticleFilterOptions = {
  /** Number of particles N. Default 500. */
  particles?: number
  resampling?: ResamplingScheme
  /** Resample when ESS < threshold × N (1 resamples every step, 0 never). Default 0.5. */
  threshold?: number
}

/**
 * The bootstrap particle filter (Gordon, Salmond & Smith, 1993): propagate each particle through the transition,
 * weight it by the likelihood of the new observation, and resample when the effective sample size falls below
 * threshold × N (adaptive resampling, Liu & Chen, 1995). Step t assimilates y_t (drawing from its step stream) and
 * the run is done after the last observation.
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
  dim: number
  samplePrior: (s: Stream) => VectorLike | number
  logPrior: (theta: Vector) => number
  logLikelihood: (theta: Vector) => number
}

/** The state of `temperedSmc`. */
export interface TemperedSmcState extends Status {
  t: number
  /** The inverse temperature β ∈ [0, 1] of the current target π_β ∝ p(θ) L(θ)^β. */
  beta: number
  /** Particles (N×d) targeting π_β after reweighting, resampling and moves. */
  particles: Matrix
  logWeights: Vector
  /** ESS of the incremental weights L^(β′−β) before resampling. */
  ess: number
  /** log Z_β / Z₀, the running estimate of the log evidence (log Z₁ at β = 1). */
  logEvidence: number
  /** Acceptance rate of the random-walk moves on the last step. */
  acceptanceRate: number
  diverged: boolean
}

/** Options for `temperedSmc`. */
export type TemperedSmcOptions = {
  /** Number of particles N. Default 500. */
  particles?: number
  /** Choose each next β so that the incremental ESS is this fraction of N (bisection). Default 0.5. */
  targetEss?: number
  /** Random-walk Metropolis moves per step. Default 5. */
  moves?: number
  /** Proposal scale as a multiple of the particles' standard deviation per coordinate. Default 2.38/√d. */
  scale?: number
  resampling?: ResamplingScheme
}

/**
 * Adaptive tempered SMC (Del Moral, Doucet & Jasra, 2006; Jasra et al., 2011): start from prior draws, raise β by
 * bisection so that the incremental weights L(θ)^(β′−β) keep an ESS of `targetEss` × N, resample, and rejuvenate
 * with random-walk Metropolis moves that leave π_β′ invariant. The product of the mean incremental weights estimates
 * the evidence Z = ∫ p(θ)L(θ) dθ. Step t draws from its step stream; done at β = 1.
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
