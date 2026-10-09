/**
 * The state shapes shared by the samplers of `aifn-compute/inference/stochastic`. States are plain data: step $t$ draws
 * only from the runner's step stream (`ctx.stream`), so no state holds a stream.
 */

import type { Status, Vector, VectorLike } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`: the target of a sampler, and the vector input alias.
export type { LogDensity, VectorLike } from 'aifn-compute/foundation/contracts'

/** The start of a chain: `x0`, its first point (a vector of $d$ values). */
export type ChainStart = { x0: VectorLike }

/**
 * Fields every Metropolis-type sampler state carries. `x` is the current point and `logDensity` $\log \pi(\xvec)$.
 * Step $t$ draws from the runner's `ctx.stream`, so a step is a pure function of the state and $t$.
 */
export interface ChainState extends Status {
  /** Steps taken (0 in the initial state). */
  t: number
  /** The current point $\xvec$ of the chain ($d$ values). */
  x: Vector
  /** The target's log-density $\log \pi(\xvec)$ at the current point. */
  logDensity: number
  /** Set when $\log \pi(\xvec)$ is NaN or $+\infty$, or $\xvec$ is not finite; stops the runners. */
  diverged: boolean
}

/** Fields of a sampler with an accept/reject step. */
export interface AcceptRejectState extends ChainState {
  /** The point $\xvec'$ proposed on the last step (the start point at $t = 0$). */
  proposal: Vector
  /** $\log \pi(\xvec')$ at the proposed point. */
  proposalLogDensity: number
  /**
   * The log Metropolis–Hastings ratio $\log r$ of the last proposal; the acceptance probability is $\min(1, r)$.
   */
  logAcceptanceRatio: number
  /** $\min(1, \exp(\log r))$: the acceptance probability of the last proposal (NaN at $t = 0$). */
  acceptance: number
  /** Whether the last proposal was accepted. */
  accepted: boolean
  /** Accepted proposals so far. */
  acceptedCount: number
  /** The fraction of steps whose proposal was accepted (NaN at $t = 0$). */
  acceptanceRate: number
}
