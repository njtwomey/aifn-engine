/**
 * Metropolis–Hastings samplers: the general kernel with a user proposal, the Gaussian random walk (Metropolis et al.,
 * 1953) and the independence sampler (Hastings, 1970; Tierney, 1994).
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { AcceptRejectState, ChainStart, LogDensity, VectorLike } from './types'
import { allFinite, data, logDensityAt, perCoordinate, standardNormals, toF64, vec, type F64 } from './util'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The state of a Metropolis–Hastings chain. */
export type MetropolisState = AcceptRejectState

/**
 * A proposal: `proposal`, the proposed point $\xvec'$ ($d$ values), and `logProposalRatio`,
 * $\log q(\xvec \mid \xvec') - \log q(\xvec' \mid \xvec)$ (0 for a symmetric proposal).
 */
export type Proposal = { proposal: VectorLike; logProposalRatio: number }

/**
 * Is a log-density value a failure (NaN or $+\infty$) rather than a point outside the support ($-\infty$)?
 *
 * @param v A value of $\log \pi$.
 * @returns True for NaN or $+\infty$; false for every finite value and for $-\infty$.
 */
export const badLogDensity = (v: number) => Number.isNaN(v) || v === Infinity

/**
 * The initial state shared by accept/reject samplers: the chain at $\xvec_0$, with no step taken. Throws `ShapeError`
 * when $\xvec_0$ does not have `target.dim` values.
 *
 * @param target The target, evaluated once at $\xvec_0$.
 * @param x0 The start point $\xvec_0$ ($d$ values); copied.
 * @param name The sampler's name, for error messages.
 * @returns The state at $t = 0$: the proposal is $\xvec_0$ itself, the ratios and rates NaN, and `diverged` set when
 *   $\log \pi(\xvec_0)$ is NaN or $+\infty$ or $\xvec_0$ is not finite.
 */
export function startState(target: LogDensity, x0: VectorLike, name: string): MetropolisState {
  const x = toF64(x0, name)
  if (x.length !== target.dim)
    throw new ShapeError(name, `${name}: x0 has ${x.length} values for dimension ${target.dim}`)
  const logDensity = logDensityAt(target, x)
  return {
    t: 0,
    x: vec(x),
    logDensity,
    proposal: vec(x),
    proposalLogDensity: logDensity,
    logAcceptanceRatio: NaN,
    acceptance: NaN,
    accepted: false,
    acceptedCount: 0,
    acceptanceRate: NaN,
    diverged: badLogDensity(logDensity) || !allFinite(x),
  }
}

/**
 * The Metropolis–Hastings accept/reject step: accept with probability $\min(1, \exp(\log r))$ using one uniform from
 * `draws`. A NaN ratio (e.g. a proposal outside the support of both terms) is rejected.
 *
 * @param s The state before the step; not modified.
 * @param proposal The proposed point $\xvec'$ as a working array of $d$ values.
 * @param proposalLogDensity $\log \pi(\xvec')$.
 * @param logRatio The log Metropolis–Hastings ratio $\log r$ of the proposal.
 * @param draws The stream the one uniform is drawn from.
 * @returns The state after step $t + 1$: at $\xvec'$ when accepted, at the old point otherwise, with the counts and
 *   rate updated. `diverged` is set only when an accepted point has a failed log-density or is not finite.
 */
export function acceptReject<S extends MetropolisState>(
  s: S,
  proposal: F64,
  proposalLogDensity: number,
  logRatio: number,
  draws: Stream,
): MetropolisState {
  const acceptance = Number.isNaN(logRatio) ? 0 : Math.min(1, Math.exp(logRatio))
  const accepted = uniform(draws) < acceptance
  const acceptedCount = s.acceptedCount + (accepted ? 1 : 0)
  return {
    t: s.t + 1,
    x: accepted ? vec(proposal) : s.x,
    logDensity: accepted ? proposalLogDensity : s.logDensity,
    proposal: vec(proposal),
    proposalLogDensity,
    logAcceptanceRatio: logRatio,
    acceptance,
    accepted,
    acceptedCount,
    acceptanceRate: acceptedCount / (s.t + 1),
    diverged: accepted && (badLogDensity(proposalLogDensity) || !allFinite(proposal)),
  }
}

/**
 * Metropolis–Hastings with a user proposal (Hastings, 1970): from $\xvec$, draw $\xvec'$ with `propose(x, s)`, which
 * also returns $\log q(\xvec \mid \xvec') - \log q(\xvec' \mid \xvec)$, and accept with probability
 * $\min\big(1, \pi(\xvec')q(\xvec \mid \xvec') / (\pi(\xvec)q(\xvec' \mid \xvec))\big)$. Step $t$ draws from
 * its step stream `ctx.stream`: the proposal from its child `'propose'`, the uniform from the stream itself.
 *
 * @param target The target $\pi$, through its (possibly unnormalised) `logDensity` and `dim`.
 * @param propose Draws a proposal from the current point $\xvec$ using the stream it is given, and returns it with its
 *   log proposal ratio (0 for a symmetric proposal).
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example A uniform random walk on a standard normal
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const propose = (x, s) => ({ proposal: [x.data[0] + uniform(s, -2, 2)], logProposalRatio: 0 })
 * const { draws } = sampleChains(metropolisHastings(target, propose), { x0: [0] }, { steps: 300, stream: stream(1) })
 * print('mean =', mean(draws))
 * print('variance =', variance(draws))
 *
 * @example A proposal that is not symmetric needs its ratio
 * // Propose x' = x + 0.5 + N(0, 1): q(x | x') / q(x' | x) = exp(-((x - x' - 0.5)^2 - (x' - x - 0.5)^2) / 2).
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const propose = (x, s) => {
 *   const y = x.data[0] + 0.5 + normal(s)
 *   return { proposal: [y], logProposalRatio: -((x.data[0] - y - 0.5) ** 2 - (y - x.data[0] - 0.5) ** 2) / 2 }
 * }
 * const s = run(metropolisHastings(target, propose), { x0: [0] }, 200)
 * print('acceptance rate =', s.acceptanceRate)
 * print('last point =', s.x)
 */
export function metropolisHastings(
  target: LogDensity,
  propose: (x: Tensor, s: Stream) => Proposal,
): Algorithm<ChainStart, MetropolisState> {
  const name = 'metropolis-hastings'
  return {
    name,
    init: ({ x0 }) => startState(target, x0, name),
    step: (s, ctx) => {
      const draws = ctx.stream
      const { proposal, logProposalRatio } = propose(s.x, child(draws, 'propose'))
      const y = toF64(proposal, name)
      const logY = logDensityAt(target, y)
      return acceptReject(s, y, logY, logY - s.logDensity + logProposalRatio, draws)
    },
  }
}

/** Options for `randomWalkMetropolis`. */
export type RandomWalkOptions = {
  /**
   * Standard deviation $\sigmavec$ of the Gaussian proposal: one number, or one per coordinate ($d$ values). Default 1.
   */
  scale?: number | ArrayLike<number>
}

/**
 * Random-walk Metropolis (Metropolis et al., 1953): propose $\xvec' = \xvec + \sigmavec \odot \epsilonvec$ with
 * $\epsilonvec \sim \Gauss(\zeros, \Imat)$ and accept with probability $\min(1, \pi(\xvec')/\pi(\xvec))$; the
 * proposal is symmetric, so $q$ cancels. An acceptance rate near 0.234 is optimal for many dimensions (Roberts, Gelman
 * & Gilks, 1997), near 0.44 in one. Throws `ShapeError` when `scale` is an array not of length $d$.
 *
 * @param target The target $\pi$, through its (possibly unnormalised) `logDensity` and `dim`.
 * @param options The proposal's `scale` $\sigmavec$.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example Moments of a correlated Gaussian
 * // N(0, [[1, 0.8], [0.8, 1]]): log pi(x) = -(x1^2 - 1.6 x1 x2 + x2^2) / (2 (1 - 0.64)).
 * const logDensity = (x) => {
 *   const [a, b] = [get(x, 0), get(x, 1)]
 *   return div(sub(add(mul(a, a), mul(b, b)), mul(1.6, mul(a, b))), -0.72)
 * }
 * const target = { kind: 'log-density', dim: 2, normalised: false, logDensity }
 * const { draws } = sampleChains(randomWalkMetropolis(target, { scale: 1 }), { x0: [0, 0] }, {
 *   chains: 2, steps: 300, warmup: 50, stream: stream(3),
 * })
 * const x = reshape(draws, [-1, 2])
 * print('mean =', mean(x, 0))
 * print('second moments =', div(matmul(transpose(x), x), x.shape[0]))
 *
 * @example The scale trades acceptance for step length
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * for (const scale of [0.1, 2.4, 20]) {
 *   const s = run(randomWalkMetropolis(target, { scale }), { x0: [0] }, 300)
 *   print(`scale ${scale}: acceptance rate =`, s.acceptanceRate)
 * }
 */
export function randomWalkMetropolis(
  target: LogDensity,
  options: RandomWalkOptions = {},
): Algorithm<ChainStart, MetropolisState> {
  const name = 'random-walk-metropolis'
  const scale = perCoordinate(options.scale ?? 1, target.dim, name)
  return {
    name,
    init: ({ x0 }) => startState(target, x0, name),
    step: (s, ctx) => {
      const draws = ctx.stream
      const eps = standardNormals(child(draws, 'propose'), target.dim)
      const x = data(s.x)
      const y = new Float64Array(target.dim)
      for (let i = 0; i < y.length; i++) y[i] = x[i] + scale[i] * eps[i]
      const logY = logDensityAt(target, y)
      return acceptReject(s, y, logY, logY - s.logDensity, draws)
    },
  }
}

/** A proposal distribution for the independence sampler: a sampler and its log-density (up to a constant). */
export type IndependentProposal = {
  /** Draws a point $\xvec' \sim q$ ($d$ values) from the stream it is given. */
  sample: (s: Stream) => VectorLike
  /** $\log q(\xvec)$, up to a constant. */
  logDensity: (x: Tensor) => number
}

/**
 * The independence sampler (Tierney, 1994): propose $\xvec' \sim q$ independently of $\xvec$ and accept with
 * probability $\min(1, w(\xvec')/w(\xvec))$ where $w = \pi/q$ is the importance weight. It mixes well only when $q$
 * covers $\pi$'s tails.
 *
 * @param target The target $\pi$, through its (possibly unnormalised) `logDensity` and `dim`.
 * @param proposal The proposal $q$: `sample` draws from it, `logDensity` evaluates it at the proposed and the current
 *   point.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example A wide Gaussian proposal for $\Gauss(1, 0.25)$
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-2, sum(square(sub(x, 1)))) }
 * const proposal = { sample: (s) => [normal(s, 0, 2)], logDensity: (x) => -0.5 * (x.data[0] / 2) ** 2 }
 * const { draws } = sampleChains(independenceMetropolis(target, proposal), { x0: [0] }, {
 *   chains: 2, steps: 300, stream: stream(2),
 * })
 * print('mean =', mean(draws))
 * print('variance =', variance(draws))
 */
export function independenceMetropolis(
  target: LogDensity,
  proposal: IndependentProposal,
): Algorithm<ChainStart, MetropolisState> {
  const name = 'independence-metropolis'
  return {
    name,
    init: ({ x0 }) => startState(target, x0, name),
    step: (s, ctx) => {
      const draws = ctx.stream
      const y = toF64(proposal.sample(child(draws, 'propose')), name)
      const logY = logDensityAt(target, y)
      const logRatio = logY - proposal.logDensity(vec(y)) - (s.logDensity - proposal.logDensity(s.x))
      return acceptReject(s, y, logY, logRatio, draws)
    },
  }
}
