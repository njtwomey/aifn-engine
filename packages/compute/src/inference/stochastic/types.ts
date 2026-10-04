/**
 * The state shapes shared by the samplers of `aifn-compute/inference/stochastic`. States are plain data: step t draws only
 * from the runner's step stream (`ctx.stream`), so no state holds a stream.
 */

import type { Status, Vector, VectorLike } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`: the target of a sampler, and the vector input alias.
export type { LogDensity, VectorLike } from 'aifn-compute/foundation/contracts'

/** The start of a chain: its first point. */
export type ChainStart = { x0: VectorLike }

/**
 * Fields every Metropolis-type sampler state carries. `x` is the current point and `logDensity` log π(x). Step t draws
 * from the runner's `ctx.stream`, so a step is a pure function of the state and t.
 */
export interface ChainState extends Status {
  /** Steps taken (0 in the initial state). */
  t: number
  x: Vector
  logDensity: number
  /** Set when log π(x) is NaN or +Infinity, or x is not finite; stops the runners. */
  diverged: boolean
}

/** Fields of a sampler with an accept/reject step. */
export interface AcceptRejectState extends ChainState {
  /** The point proposed on the last step and log π there (the start point at t = 0). */
  proposal: Vector
  proposalLogDensity: number
  /** The log Metropolis–Hastings ratio of the last proposal; the acceptance probability is min(1, exp of it). */
  logAcceptanceRatio: number
  /** min(1, exp(logAcceptanceRatio)): the acceptance probability of the last proposal (NaN at t = 0). */
  acceptance: number
  accepted: boolean
  /** Accepted proposals so far and their fraction of steps (NaN at t = 0). */
  acceptedCount: number
  acceptanceRate: number
}
