/**
 * Langevin samplers: the unadjusted Langevin algorithm (ULA), the Metropolis-adjusted Langevin algorithm (MALA;
 * Roberts & Tweedie, 1996) and stochastic-gradient Langevin dynamics (SGLD; Welling & Teh, 2011).
 *
 * All discretise the Langevin diffusion dθ = ∇ log π(θ) dt + √2 dW, whose stationary law is π, with the Euler step
 * θ′ = θ + h ∇ log π(θ) + √(2h) ξ, ξ ~ N(0, I). ULA keeps every step and so samples a biased law π_h ≠ π whose bias
 * grows with h; MALA treats the step as a proposal and corrects it with a Metropolis–Hastings test, so π is exact.
 */

import { child, integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { badLogDensity } from './metropolis'
import type { AcceptRejectState, ChainStart, ChainState, LogDensity, VectorLike } from './types'
import { allFinite, data, logDensityAndGrad, standardNormals, toF64, vec, type F64 } from './util'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The state of `unadjustedLangevin` and `mala`. */
export type LangevinState = AcceptRejectState & {
  /** ∇ log π(x). */
  grad: Vector
  /** The mean of the last proposal, x + h∇ log π(x) (the drift step before noise). */
  drift: Vector
  /** The step size h. */
  stepSize: number
}

/** Options for the Langevin samplers. */
export type LangevinOptions = {
  /** Time step h of the Euler step θ′ = θ + h∇ log π(θ) + √(2h)ξ. Default 0.1. */
  stepSize?: number
}

function langevin(
  target: LogDensity,
  name: string,
  adjusted: boolean,
  h: number,
): Algorithm<ChainStart, LangevinState> {
  const d = target.dim
  const sd = Math.sqrt(2 * h)
  // log q(to | from) up to a constant, for q(· | x) = N(x + h∇ log π(x), 2hI).
  const logQ = (to: F64, from: F64, gradFrom: F64) => {
    let s = 0
    for (let i = 0; i < d; i++) s += (to[i] - from[i] - h * gradFrom[i]) ** 2
    return -s / (4 * h)
  }
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new ShapeError(name, `${name}: x0 has ${x.length} values for dimension ${d}`)
      const { value, grad } = logDensityAndGrad(target, x)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: value,
        grad: vec(grad),
        drift: X,
        stepSize: h,
        proposal: X,
        proposalLogDensity: value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        diverged: badLogDensity(value) || !allFinite(x) || !allFinite(grad),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const xi = standardNormals(child(draws, 'noise'), d)
      const x = data(s.x)
      const g = data(s.grad)
      const drift = new Float64Array(d)
      const y = new Float64Array(d)
      for (let i = 0; i < d; i++) {
        drift[i] = x[i] + h * g[i]
        y[i] = drift[i] + sd * xi[i]
      }
      const next = logDensityAndGrad(target, y)
      let logRatio = 0
      let acceptance = 1
      let accepted = true
      if (adjusted) {
        logRatio = next.value - s.logDensity + logQ(x, y, next.grad) - logQ(y, x, g)
        acceptance = Number.isNaN(logRatio) ? 0 : Math.min(1, Math.exp(logRatio))
        accepted = uniform(draws) < acceptance
      }
      const acceptedCount = s.acceptedCount + (accepted ? 1 : 0)
      return {
        ...s,
        t: s.t + 1,
        x: accepted ? vec(y) : s.x,
        logDensity: accepted ? next.value : s.logDensity,
        grad: accepted ? vec(next.grad) : s.grad,
        drift: vec(drift),
        proposal: vec(y),
        proposalLogDensity: next.value,
        logAcceptanceRatio: logRatio,
        acceptance,
        accepted,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        diverged: accepted && (badLogDensity(next.value) || !allFinite(y) || !allFinite(next.grad)),
      }
    },
  }
}

/**
 * The unadjusted Langevin algorithm: θ′ = θ + h∇ log π(θ) + √(2h)ξ, every step kept (Roberts & Tweedie, 1996, §1.4
 * call it ULA and show it can be transient for large h). Its stationary law is biased by O(h): for a Gaussian
 * N(0, σ²) it is N(0, σ²/(1 − h/(2σ²))). `acceptance` is 1 on every step.
 */
export function unadjustedLangevin(
  target: LogDensity,
  options: LangevinOptions = {},
): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'unadjusted-langevin', false, options.stepSize ?? 0.1)
}

/**
 * The Metropolis-adjusted Langevin algorithm (Roberts & Tweedie, 1996): propose the ULA step and accept with the
 * Metropolis–Hastings ratio, whose proposal density q(θ′ | θ) = N(θ + h∇ log π(θ), 2hI) is not symmetric. Its
 * optimal acceptance rate is about 0.574 in high dimension (Roberts & Rosenthal, 1998).
 */
export function mala(target: LogDensity, options: LangevinOptions = {}): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'mala', true, options.stepSize ?? 0.1)
}

// ---------------------------------------------------------------------------------------------------------------------
// SGLD.

/**
 * A model for SGLD: N data points with per-datum log-likelihood gradients and a log-prior gradient, for the posterior
 * log π(θ) = log p(θ) + Σᵢ log p(yᵢ | θ) + const.
 */
export type MinibatchModel = {
  dim: number
  /** N, the number of data points. */
  size: number
  gradLogPrior: (theta: Vector) => VectorLike
  /** ∇θ log p(yᵢ | θ) for data point i. */
  gradLogLikelihood: (theta: Vector, i: number) => VectorLike
}

/** The state of `sgld`. */
export type SgldState = ChainState & {
  /** The last minibatch (int32 indices into the data). */
  batch: Tensor
  /** The minibatch estimate of ∇ log π at the previous point: ∇ log p(θ) + (N/n) Σ_{i∈batch} ∇ log p(yᵢ | θ). */
  gradEstimate: Vector
  /** The step size ε_t used on the last step. */
  stepSize: number
}

/** Options for `sgld`. */
export type SgldOptions = {
  /** Minibatch size n (drawn without replacement each step). Default 10. */
  batchSize?: number
  /**
   * Step size ε_t, or a schedule t ↦ ε_t. Welling & Teh use ε_t = a(b + t)^(−γ) with γ ∈ (0.5, 1] so that Σε = ∞ and
   * Σε² < ∞. Default 1e-3.
   */
  stepSize?: number | ((t: number) => number)
}

/**
 * Stochastic-gradient Langevin dynamics (Welling & Teh, 2011, eq. 4): θ′ = θ + (ε_t/2)(∇ log p(θ) +
 * (N/n) Σ_{i∈batch} ∇ log p(yᵢ | θ)) + η, η ~ N(0, ε_t I). There is no accept/reject step: as ε_t → 0 the injected
 * noise dominates the minibatch noise and the iterates sample the posterior. `logDensity` is NaN (never evaluated).
 */
export function sgld(model: MinibatchModel, options: SgldOptions = {}): Algorithm<ChainStart, SgldState> {
  const name = 'sgld'
  const { batchSize = 10, stepSize = 1e-3 } = options
  const d = model.dim
  const N = model.size
  const n = Math.min(batchSize, N)
  const rate = (t: number) => (typeof stepSize === 'function' ? stepSize(t) : stepSize)
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new ShapeError(name, `${name}: x0 has ${x.length} values for dimension ${d}`)
      return {
        t: 0,
        x: vec(x),
        logDensity: NaN,
        batch: fromData(new Int32Array(0), [0]),
        gradEstimate: vec(new Float64Array(d)),
        stepSize: NaN,
        diverged: !allFinite(x),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      // A partial Fisher–Yates shuffle picks n indices without replacement.
      const order = Int32Array.from({ length: N }, (_, i) => i)
      const pick = child(draws, 'batch')
      for (let i = 0; i < n; i++) {
        const j = i + integers(pick, N - i)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const batch = order.slice(0, n)
      const x = data(s.x)
      const g = toF64(model.gradLogPrior(s.x), name)
      for (const i of batch) {
        const gi = toF64(model.gradLogLikelihood(s.x, i), name)
        for (let k = 0; k < d; k++) g[k] += (N / n) * gi[k]
      }
      const eps = rate(s.t)
      const eta = standardNormals(child(draws, 'noise'), d)
      const y = new Float64Array(d)
      for (let k = 0; k < d; k++) y[k] = x[k] + 0.5 * eps * g[k] + Math.sqrt(eps) * eta[k]
      return {
        ...s,
        t: s.t + 1,
        x: vec(y),
        batch: fromData(batch, [n]),
        gradEstimate: vec(g),
        stepSize: eps,
        diverged: !allFinite(y),
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Langevin on a batch of particles, and persistent chains with a replay buffer (energy-based models).

/** A batch score: ∇ₓ log π at every row of x ([n, d] → [n, d]), e.g. −∇ₓ E(x) for an energy E. */
export type BatchScore = (x: Tensor) => Tensor

/** Options of `langevinParticles` and `persistentLangevin`. */
export type ParticleLangevinOptions = {
  /** The gradient step α of x′ = x + α∇ log π(x) + σξ. Default 0.01. */
  stepSize?: number
  /**
   * The noise standard deviation σ. Default √(2α), the Euler step of the Langevin diffusion, whose law tends to π as
   * α → 0. Smaller noise (JEM takes α = 1, σ = 0.01; Grathwohl et al., 2019, §4) samples a sharpened law: short-run
   * chains that serve as negatives in training, not as exact draws.
   */
  noise?: number
  /** Clip every coordinate to [−bound, bound] after each step (the box the data live in). Default: no clipping. */
  bound?: number
}

/** The state of `langevinParticles`. */
export type ParticleLangevinState = {
  t: number
  /** The particles, [n, d]. */
  x: Tensor
  /** The score at the particles before the last step, [n, d]. */
  grad: Tensor
  /** The step size α. */
  stepSize: number
  diverged: boolean
}

function langevinMove(score: BatchScore, x: Tensor, s: Stream, alpha: number, sigma: number, bound: number) {
  const [n, d] = x.shape
  const g = toFlat(score(x))
  const xs = toFlat(x)
  const xi = sigma > 0 ? standardNormals(s, n * d) : new Float64Array(n * d)
  const y = new Float64Array(n * d)
  for (let k = 0; k < n * d; k++) {
    const v = xs[k] + alpha * g[k] + sigma * xi[k]
    y[k] = v > bound ? bound : v < -bound ? -bound : v
  }
  return { x: fromData(y, [n, d]), grad: fromData(Float64Array.from(g), [n, d]) }
}

/**
 * Unadjusted Langevin steps on a batch of particles at once: x′ = x + α∇ log π(x) + σξ, ξ ~ N(0, I) per row, with a
 * batch score (one network evaluation moves every particle). With σ = √(2α) this is `unadjustedLangevin` run on n
 * independent chains; with smaller σ it is the short-run sampler of energy-based-model training (Du & Mordatch, 2019;
 * Nijkamp et al., 2019). `init` takes the particles `{ x }` ([n, d]).
 */
export function langevinParticles(
  score: BatchScore,
  options: ParticleLangevinOptions = {},
): Algorithm<{ x: Tensor }, ParticleLangevinState> {
  const alpha = options.stepSize ?? 0.01
  const sigma = options.noise ?? Math.sqrt(2 * alpha)
  const bound = options.bound ?? Infinity
  return {
    name: 'langevin-particles',
    init: ({ x }) => {
      if (x.shape.length !== 2)
        throw new ShapeError('langevinParticles', 'langevinParticles: the particles must be an [n, d] tensor')
      return {
        t: 0,
        x,
        grad: fromData(new Float64Array(x.shape[0] * x.shape[1]), x.shape),
        stepSize: alpha,
        diverged: false,
      }
    },
    step: (s, ctx) => {
      const next = langevinMove(score, s.x, child(ctx.stream, 'noise'), alpha, sigma, bound)
      return { ...s, t: s.t + 1, x: next.x, grad: next.grad, diverged: !allFinite(toFlat(next.x)) }
    },
  }
}

/** A replay buffer of persistent chains: the last positions of past samples, [m, d]. Plain data. */
export type ChainBuffer = {
  /** The stored particles, [m, d]. */
  readonly samples: Tensor
}

/** Options of `persistentLangevin`. */
export type PersistentLangevinOptions = ParticleLangevinOptions & {
  /** Langevin steps per draw (K). Default 20 (JEM). */
  steps?: number
  /** The probability ρ that a drawn chain restarts from `fresh` instead of its stored position. Default 0.05 (JEM). */
  reinitialise?: number
  /** Fresh starting points: n rows [n, d] from the stream (JEM: uniform on the data's box). */
  fresh: (s: Stream, n: number) => Tensor
}

/** What `persistentLangevin` returns. */
export type PersistentDraw = {
  /** The n samples after K steps, [n, d]. */
  x: Tensor
  /** Where the chains started (stored or fresh), [n, d]. */
  start: Tensor
  /** The buffer with the drawn slots overwritten by the new samples. */
  buffer: ChainBuffer
  /** The buffer slots drawn (int32 [n]). */
  slots: Tensor
  /** 1 where a chain was restarted from `fresh` (uint8 [n]). */
  restarted: Uint8Array
}

/** A replay buffer of m fresh points. */
export function chainBuffer(s: Stream, m: number, fresh: (s: Stream, n: number) => Tensor): ChainBuffer {
  return { samples: fresh(s, m) }
}

/**
 * Draw n samples by persistent short-run Langevin with a replay buffer (Du & Mordatch, 2019; Grathwohl et al., 2019,
 * Algorithm 1): pick n slots of the buffer uniformly, restart each with probability ρ from `fresh`, run K steps of
 * `langevinParticles` from there, and write the results back to their slots. Chains thus persist across training
 * steps (persistent contrastive divergence; Tieleman, 2008) while ρ keeps new mass entering. Pure: the stream decides
 * every draw (`slots`, `restart`, `fresh`, and step k's noise from `child(s, 'step', k)`).
 */
export function persistentLangevin(
  score: BatchScore,
  buffer: ChainBuffer,
  s: Stream,
  n: number,
  options: PersistentLangevinOptions,
): PersistentDraw {
  const { steps = 20, reinitialise = 0.05, fresh } = options
  const alpha = options.stepSize ?? 0.01
  const sigma = options.noise ?? Math.sqrt(2 * alpha)
  const bound = options.bound ?? Infinity
  const [m, d] = buffer.samples.shape
  const stored = toFlat(buffer.samples)
  const pick = child(s, 'slots')
  const coin = child(s, 'restart')
  const slots = Int32Array.from({ length: n }, () => integers(pick, m))
  const restarted = Uint8Array.from({ length: n }, () => (uniform(coin) < reinitialise ? 1 : 0))
  const replacements = toFlat(fresh(child(s, 'fresh'), n))
  const start = new Float64Array(n * d)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) start[i * d + j] = restarted[i] ? replacements[i * d + j] : stored[slots[i] * d + j]
  let x = fromData(start, [n, d])
  for (let k = 0; k < steps; k++) x = langevinMove(score, x, child(s, 'step', k), alpha, sigma, bound).x
  const next = Float64Array.from(stored)
  const xs = toFlat(x)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) next[slots[i] * d + j] = xs[i * d + j]
  return {
    x,
    start: fromData(start, [n, d]),
    buffer: { samples: fromData(next, [m, d]) },
    slots: fromData(slots, [n]),
    restarted,
  }
}
