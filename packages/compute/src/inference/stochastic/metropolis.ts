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

/** A proposal: the proposed point and log q(x | x′) − log q(x′ | x) (0 for a symmetric proposal). */
export type Proposal = { proposal: VectorLike; logProposalRatio: number }

/** Is a log-density value a failure (NaN or +Infinity) rather than a point outside the support (−Infinity)? */
export const badLogDensity = (v: number) => Number.isNaN(v) || v === Infinity

/** The initial state shared by accept/reject samplers. */
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
 * The Metropolis–Hastings accept/reject step: accept with probability min(1, exp(logRatio)) using one uniform from
 * `draws`. A NaN ratio (e.g. a proposal outside the support of both terms) is rejected.
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
 * Metropolis–Hastings with a user proposal (Hastings, 1970): from x, draw x′ with `propose(x, s)`, which also returns
 * log q(x | x′) − log q(x′ | x), and accept with probability min(1, π(x′)q(x | x′) / (π(x)q(x′ | x))). Step t draws
 * from its step stream `ctx.stream`: the proposal from its child `'propose'`, the uniform from the stream itself.
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
  /** Standard deviation of the Gaussian proposal: one number, or one per coordinate. Default 1. */
  scale?: number | ArrayLike<number>
}

/**
 * Random-walk Metropolis (Metropolis et al., 1953): propose x′ = x + σ ⊙ ε with ε ~ N(0, I) and accept with
 * probability min(1, π(x′)/π(x)); the proposal is symmetric, so q cancels. An acceptance rate near 0.234 is optimal
 * for many dimensions (Roberts, Gelman & Gilks, 1997), near 0.44 in one.
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
  sample: (s: Stream) => VectorLike
  logDensity: (x: Tensor) => number
}

/**
 * The independence sampler (Tierney, 1994): propose x′ ~ q independently of x and accept with probability
 * min(1, w(x′)/w(x)) where w = π/q is the importance weight. It mixes well only when q covers π's tails.
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
