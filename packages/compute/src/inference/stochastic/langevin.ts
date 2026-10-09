/**
 * Langevin samplers: the unadjusted Langevin algorithm (ULA), the Metropolis-adjusted Langevin algorithm (MALA;
 * Roberts & Tweedie, 1996) and stochastic-gradient Langevin dynamics (SGLD; Welling & Teh, 2011), and short-run
 * Langevin on a batch of particles, with persistent chains in a replay buffer, for energy-based models.
 *
 * All discretise the Langevin diffusion $d\thetavec = \nabla \log \pi(\thetavec)\,dt + \sqrt{2}\,d\mathbf{W}_t$
 * ($\mathbf{W}_t$ a Brownian motion), whose stationary law is $\pi$, with the Euler step
 * $\thetavec' = \thetavec + h \nabla \log \pi(\thetavec) + \sqrt{2h}\,\xivec$,
 * $\xivec \sim \Gauss(\zeros, \Imat)$. ULA keeps every step and so samples a biased law $\pi_h \ne \pi$ whose bias
 * grows with $h$; MALA treats the step as a proposal and corrects it with a Metropolis–Hastings test, so $\pi$ is
 * exact.
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
  /** $\nabla \log \pi(\xvec)$. */
  grad: Vector
  /**
   * The mean of the last proposal, $\xvec + h\nabla \log \pi(\xvec)$ (the drift step before noise; $\xvec_0$ at
   * $t = 0$).
   */
  drift: Vector
  /** The step size $h$. */
  stepSize: number
}

/** Options for the Langevin samplers. */
export type LangevinOptions = {
  /**
   * Time step $h$ of the Euler step $\thetavec' = \thetavec + h\nabla \log \pi(\thetavec) + \sqrt{2h}\,\xivec$.
   * Default 0.1.
   */
  stepSize?: number
}

/**
 * The Langevin step shared by ULA and MALA. Step $t$ draws its noise from `child(ctx.stream, 'noise')` and, when
 * adjusted, its uniform from `ctx.stream` itself. Throws `ShapeError` when $\xvec_0$ does not have `target.dim`
 * values.
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff of `logDensity`) and `dim`.
 * @param name The sampler's name, for the algorithm and error messages.
 * @param adjusted True for MALA (a Metropolis–Hastings test on each step), false for ULA (every step kept).
 * @param h The time step $h$ (not checked).
 * @returns The sampler as an algorithm.
 */
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
 * The unadjusted Langevin algorithm: $\thetavec' = \thetavec + h\nabla \log \pi(\thetavec) + \sqrt{2h}\,\xivec$,
 * every step kept (Roberts & Tweedie, 1996, §1.4 call it ULA and show it can be transient for large $h$). Its
 * stationary law is biased by $O(h)$: for a Gaussian $\Gauss(0, \sigma^2)$ it is
 * $\Gauss(0, \sigma^2/(1 - h/(2\sigma^2)))$. `acceptance` is 1 on every step.
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff of `logDensity`) and `dim`.
 * @param options The time step `stepSize` $h$.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example The $O(h)$ bias on a standard normal
 * // At h = 0.5 the stationary variance is 1 / (1 - 0.25) = 4/3; MALA's is 1.
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * for (const alg of [unadjustedLangevin(target, { stepSize: 0.5 }), mala(target, { stepSize: 0.5 })]) {
 *   const { draws } = sampleChains(alg, { x0: [0] }, { chains: 4, steps: 300, warmup: 20, stream: stream(1) })
 *   print(`${alg.name}: mean =`, mean(draws), 'variance =', variance(draws))
 * }
 */
export function unadjustedLangevin(
  target: LogDensity,
  options: LangevinOptions = {},
): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'unadjusted-langevin', false, options.stepSize ?? 0.1)
}

/**
 * The Metropolis-adjusted Langevin algorithm (Roberts & Tweedie, 1996): propose the ULA step and accept with the
 * Metropolis–Hastings ratio, whose proposal density
 * $q(\thetavec' \mid \thetavec) = \Gauss(\thetavec + h\nabla \log \pi(\thetavec), 2h\Imat)$ is not symmetric. Its
 * optimal acceptance rate is about 0.574 in high dimension (Roberts & Rosenthal, 1998).
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff of `logDensity`) and `dim`.
 * @param options The time step `stepSize` $h$.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example Moments of a standard normal in two dimensions
 * const target = { kind: 'log-density', dim: 2, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const { draws, traces } = sampleChains(mala(target, { stepSize: 0.8 }), { x0: [0, 0] }, {
 *   chains: 2, steps: 300, warmup: 20, stream: stream(2),
 * })
 * const x = reshape(draws, [-1, 2])
 * print('mean =', mean(x, 0))
 * print('second moments =', div(matmul(transpose(x), x), x.shape[0]))
 * print('acceptance rate =', traces[0].final.acceptanceRate)
 *
 * @example Larger steps are rejected more often
 * const target = { kind: 'log-density', dim: 2, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * for (const stepSize of [0.1, 0.8, 2, 4]) {
 *   const s = run(mala(target, { stepSize }), { x0: [0, 0] }, 300)
 *   print(`h = ${stepSize}: acceptance rate =`, s.acceptanceRate)
 * }
 */
export function mala(target: LogDensity, options: LangevinOptions = {}): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'mala', true, options.stepSize ?? 0.1)
}

// ---------------------------------------------------------------------------------------------------------------------
// SGLD.

/**
 * A model for SGLD: $N$ data points with per-datum log-likelihood gradients and a log-prior gradient, for the posterior
 * $\log \pi(\thetavec) = \log p(\thetavec) + \sum_i \log p(y_i \mid \thetavec) + \text{const}$.
 */
export type MinibatchModel = {
  /** The length $d$ of $\thetavec$. */
  dim: number
  /** $N$, the number of data points. */
  size: number
  /** $\nabla_\thetavec \log p(\thetavec)$, $d$ values. */
  gradLogPrior: (theta: Vector) => VectorLike
  /** $\nabla_\thetavec \log p(y_i \mid \thetavec)$ for data point $i$ ($0 \le i < N$), $d$ values. */
  gradLogLikelihood: (theta: Vector, i: number) => VectorLike
}

/** The state of `sgld`. */
export type SgldState = ChainState & {
  /** The last minibatch (int32 indices into the data). */
  batch: Tensor
  /**
   * The minibatch estimate of $\nabla \log \pi$ at the previous point:
   * $\nabla \log p(\thetavec) + (N/n) \sum_{i \in \text{batch}} \nabla \log p(y_i \mid \thetavec)$.
   */
  gradEstimate: Vector
  /** The step size $\varepsilon_t$ used on the last step (NaN at $t = 0$). */
  stepSize: number
}

/** Options for `sgld`. */
export type SgldOptions = {
  /** Minibatch size $n$ (drawn without replacement each step; at most $N$). Default 10. */
  batchSize?: number
  /**
   * Step size $\varepsilon_t$, or a schedule $t \mapsto \varepsilon_t$, called with the number of steps already taken
   * (0 on the first). Welling & Teh use $\varepsilon_t = a(b + t)^{-\gamma}$ with $\gamma \in (0.5, 1]$ so that
   * $\sum \varepsilon_t = \infty$ and $\sum \varepsilon_t^2 < \infty$. Default 1e-3.
   */
  stepSize?: number | ((t: number) => number)
}

/**
 * Stochastic-gradient Langevin dynamics (Welling & Teh, 2011, eq. 4):
 * $\thetavec' = \thetavec + \frac{\varepsilon_t}{2}\hat\gvec(\thetavec) + \etavec$,
 * $\etavec \sim \Gauss(\zeros, \varepsilon_t \Imat)$, with the estimate of $\nabla \log \pi(\thetavec)$ from a
 * minibatch $\Bcal$ of $n$ data points,
 * $\hat\gvec(\thetavec) = \nabla \log p(\thetavec) + \frac{N}{n} \sum_{i \in \Bcal} \nabla \log p(y_i \mid \thetavec)$.
 * There is no accept/reject step: as $\varepsilon_t \to 0$ the injected noise dominates the minibatch noise and the
 * iterates sample the posterior. `logDensity` is NaN (never evaluated). Step $t$ draws its minibatch from
 * `child(ctx.stream, 'batch')` and its noise from `child(ctx.stream, 'noise')`.
 *
 * @param model The model: dimension, data size and the gradients of the log-prior and of each datum's log-likelihood.
 * @param options The `batchSize` $n$ and the `stepSize` $\varepsilon_t$ or its schedule.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example The posterior of a Gaussian mean
 * // y_i ~ N(theta, 1), prior N(0, 10^2): the posterior is N(sum y / (N + 0.01), 1 / (N + 0.01)).
 * const y = toFlat(normal(stream('data'), 1, 1, { shape: [100] }))
 * const model = {
 *   dim: 1, size: 100,
 *   gradLogPrior: (t) => [-t.data[0] / 100],
 *   gradLogLikelihood: (t, i) => [y[i] - t.data[0]],
 * }
 * const { draws } = sampleChains(sgld(model, { stepSize: 2e-3 }), { x0: [0] }, {
 *   chains: 2, steps: 300, warmup: 50, stream: stream(3),
 * })
 * print('posterior mean =', y.reduce((a, b) => a + b) / 100.01, 'sd =', 1 / Math.sqrt(100.01))
 * print('SGLD mean =', mean(draws), 'sd =', Math.sqrt(variance(draws)))
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

/**
 * A batch score: $\nabla_\xvec \log \pi$ at every row of $\Xmat$ ($n \times d$ to $n \times d$), e.g.
 * $-\nabla_\xvec E(\xvec)$ for an energy $E$.
 */
export type BatchScore = (x: Tensor) => Tensor

/** Options of `langevinParticles` and `persistentLangevin`. */
export type ParticleLangevinOptions = {
  /** The gradient step $\alpha$ of $\xvec' = \xvec + \alpha\nabla \log \pi(\xvec) + \sigma\xivec$. Default 0.01. */
  stepSize?: number
  /**
   * The noise standard deviation $\sigma$. Default $\sqrt{2\alpha}$, the Euler step of the Langevin diffusion, whose
   * law tends to $\pi$ as $\alpha \to 0$. Smaller noise (JEM takes $\alpha = 1$, $\sigma = 0.01$; Grathwohl et al.,
   * 2019, §4) samples a sharpened law: short-run chains that serve as negatives in training, not as exact draws.
   */
  noise?: number
  /**
   * Clip every coordinate to $[-b, b]$, $b$ the bound, after each step (the box the data live in). Default: no
   * clipping.
   */
  bound?: number
}

/** The state of `langevinParticles`. */
export type ParticleLangevinState = {
  /** Steps taken (0 in the initial state). */
  t: number
  /** The particles, $n \times d$. */
  x: Tensor
  /** The score at the particles before the last step, $n \times d$ (zeros at $t = 0$). */
  grad: Tensor
  /** The step size $\alpha$. */
  stepSize: number
  /** Set when a particle is no longer finite; stops the runners. */
  diverged: boolean
}

/**
 * One Langevin step of every particle: $\xvec' = \xvec + \alpha \nabla \log \pi(\xvec) + \sigma\xivec$, clipped
 * to $[-b, b]$.
 *
 * @param score The batch score, evaluated once at `x`.
 * @param x The particles, $n \times d$; not modified.
 * @param s The stream of the noise (no draws when $\sigma = 0$).
 * @param alpha The gradient step $\alpha$.
 * @param sigma The noise standard deviation $\sigma$.
 * @param bound The clipping bound $b$ (`Infinity` for none).
 * @returns The moved particles `x` and the score `grad` at the old ones, both new $n \times d$ tensors.
 */
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
 * Unadjusted Langevin steps on a batch of particles at once:
 * $\xvec' = \xvec + \alpha\nabla \log \pi(\xvec) + \sigma\xivec$, $\xivec \sim \Gauss(\zeros, \Imat)$ per row,
 * with a batch score (one network evaluation moves every particle). With $\sigma = \sqrt{2\alpha}$ this is
 * `unadjustedLangevin` run on $n$ independent chains; with smaller $\sigma$ it is the short-run sampler of
 * energy-based-model training (Du & Mordatch, 2019; Nijkamp et al., 2019). `init` takes the particles `{ x }`
 * ($n \times d$; `ShapeError` for another rank). Step $t$ draws its noise from `child(ctx.stream, 'noise')`.
 *
 * @param score The batch score $\nabla \log \pi$, mapping $n \times d$ particles to $n \times d$ gradients.
 * @param options The step $\alpha$, the noise $\sigma$ and the clipping `bound`.
 * @returns The sampler as an algorithm over the whole batch.
 *
 * @example 1000 particles relax to a standard normal
 * // The stationary variance of the Euler step is 1 / (1 - alpha / 2).
 * const s = run(langevinParticles((x) => neg(x), { stepSize: 0.05 }), { x: full([1000, 1], 3) }, 150, {
 *   stream: stream(1),
 * })
 * print('mean =', mean(s.x), 'variance =', variance(s.x))
 * print('1 / (1 - alpha / 2) =', 1 / (1 - 0.025))
 *
 * @example Less noise, a sharper law
 * // Noise below sqrt(2 alpha) concentrates the particles near the mode.
 * for (const noise of [Math.sqrt(0.1), 0.1, 0]) {
 *   const s = run(langevinParticles((x) => neg(x), { stepSize: 0.05, noise }), { x: full([1000, 1], 3) }, 150)
 *   print(`noise ${noise.toFixed(3)}: variance =`, variance(s.x))
 * }
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

/** A replay buffer of persistent chains: the last positions of past samples, $m \times d$. Plain data. */
export type ChainBuffer = {
  /** The stored particles, $m \times d$. */
  readonly samples: Tensor
}

/** Options of `persistentLangevin`. */
export type PersistentLangevinOptions = ParticleLangevinOptions & {
  /** Langevin steps per draw ($K$). Default 20 (JEM). */
  steps?: number
  /**
   * The probability $\rho$ that a drawn chain restarts from `fresh` instead of its stored position. Default 0.05 (JEM).
   */
  reinitialise?: number
  /** Fresh starting points: $n$ rows ($n \times d$) from the stream (JEM: uniform on the data's box). */
  fresh: (s: Stream, n: number) => Tensor
}

/** What `persistentLangevin` returns. */
export type PersistentDraw = {
  /** The $n$ samples after $K$ steps, $n \times d$. */
  x: Tensor
  /** Where the chains started (stored or fresh), $n \times d$. */
  start: Tensor
  /** The buffer with the drawn slots overwritten by the new samples. */
  buffer: ChainBuffer
  /** The buffer slots drawn (int32, $n$ values). */
  slots: Tensor
  /** 1 where a chain was restarted from `fresh` (uint8, $n$ values). */
  restarted: Uint8Array
}

/**
 * A replay buffer of $m$ fresh points.
 *
 * @param s The stream passed to `fresh`.
 * @param m The number of slots of the buffer.
 * @param fresh Draws $m$ starting points ($m \times d$) from the stream it is given.
 * @returns The buffer, holding `fresh(s, m)`.
 *
 * @example Five points uniform on a box
 * const buffer = chainBuffer(stream(1), 5, (s, n) => uniform(s, -1, 1, { shape: [n, 2] }))
 * print('samples =', buffer.samples)
 */
export function chainBuffer(s: Stream, m: number, fresh: (s: Stream, n: number) => Tensor): ChainBuffer {
  return { samples: fresh(s, m) }
}

/**
 * Draw $n$ samples by persistent short-run Langevin with a replay buffer (Du & Mordatch, 2019; Grathwohl et al., 2019,
 * Algorithm 1): pick $n$ slots of the buffer uniformly (independently, so a slot can be drawn twice, and then the
 * later chain's result is the one stored), restart each with probability $\rho$ from `fresh`, run $K$ steps of
 * `langevinParticles` from there, and write the results back to their slots. Chains thus persist across training
 * steps (persistent contrastive divergence; Tieleman, 2008) while $\rho$ keeps new mass entering. Pure: the stream
 * decides every draw (`slots`, `restart`, `fresh`, and step $k$'s noise from `child(s, 'step', k)`), and the buffer
 * passed in is not modified.
 *
 * @param score The batch score $\nabla \log \pi$, mapping $n \times d$ particles to $n \times d$ gradients.
 * @param buffer The replay buffer, $m \times d$.
 * @param s The stream of this draw; use a new one for each call.
 * @param n The number of chains to run (and samples returned).
 * @param options The Langevin step, noise and bound, the steps $K$, the restart probability $\rho$ and `fresh`.
 * @returns The samples, their starts, the updated buffer, and which slots were drawn and restarted.
 *
 * @example The buffer drifts to a standard normal over calls
 * const fresh = (s, n) => uniform(s, -4, 4, { shape: [n, 1] })
 * let buffer = chainBuffer(stream(0), 100, fresh)
 * print('buffer at start: mean', mean(buffer.samples), 'variance', variance(buffer.samples))
 * for (let call = 1; call <= 30; call++)
 *   buffer = persistentLangevin((x) => neg(x), buffer, stream(call), 20, { stepSize: 0.05, steps: 10, fresh }).buffer
 * print('after 30 calls: mean', mean(buffer.samples), 'variance', variance(buffer.samples))
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
