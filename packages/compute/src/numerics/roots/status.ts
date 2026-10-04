/**
 * The `Status` flags of the root finders, derived from their `failure` field in one place: a non-finite value or a
 * diverging iteration sets `diverged`; any other failure (no sign change, a zero derivative, a singular Jacobian)
 * sets `terminated`. The runner stops on either, and on `converged`.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/** A root-finder state: a `Status` with a failure reason. */
export type FailureState = Status & {
  /** Whether the method converged within specified tolerances. */
  converged: boolean
  /** Failure diagnosis string, or `null` if the method is healthy. */
  failure: string | null
}

/** Failures that mean the iterates left every bound, as opposed to a method that cannot take its next step. */
const DIVERGENT = new Set(['not finite', 'diverging'])

/**
 * Set `diverged` and `terminated` status flags on a state from its `failure` reason.
 *
 * @param s Algorithm state with failure information.
 * @returns Updated state with normalised status flags.
 */
export function flag<S extends FailureState>(s: S): S {
  const failed = s.failure !== null
  const diverged = failed && DIVERGENT.has(s.failure as string)
  return { ...s, converged: s.converged && !failed, diverged, terminated: failed && !diverged }
}

/**
 * Wrap an algorithm to apply `flag` to every produced state.
 *
 * @param alg Root-finding algorithm.
 * @returns Wrapped algorithm with automated status flag updates.
 */
export function flagged<Start, S extends FailureState>(alg: Algorithm<Start, S>): Algorithm<Start, S> {
  return {
    name: alg.name,
    init: (start, stream) => flag(alg.init(start, stream)),
    step: (state, ctx) => flag(alg.step(state, ctx)),
  }
}
