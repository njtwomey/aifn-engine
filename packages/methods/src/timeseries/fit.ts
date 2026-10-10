/**
 * Private: maximum-likelihood and least-squares fits as traceable algorithms. Each fit maps unconstrained coordinates
 * $\uvec$ to valid parameters (a logistic map to $(0, 1)$, partial autocorrelations to a stationary AR polynomial, and
 * so on) and runs Nelder–Mead (Nelder and Mead, 1965) from `aifn-compute/optim` on the objective in $\uvec$, so every
 * step of a fit can be traced.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { nelderMead, type NelderMeadState } from 'aifn-compute/optim/derivative-free'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/** The state of a fit: the optimiser's state and the parameters its best vertex decodes to. */
export type FitState<P> = Status & {
  /** Steps taken. */
  t: number
  /** The parameters at the best vertex. */
  params: P
  /** The objective at the best vertex (a negative log-likelihood or a sum of squares). */
  objective: number
  /** Objective evaluations so far. */
  evaluations: number
  /** True when the simplex size and the value spread are both within the tolerance. */
  converged: boolean
  /** True when the best vertex is not finite (invalid parameters score `INVALID`, which never diverges). */
  diverged: boolean
  /** The Nelder–Mead state in the unconstrained coordinates. */
  optimiser: NelderMeadState
}

/** Objective values for invalid parameters: finite, so the simplex simply moves away from them. */
export const INVALID = 1e300

/**
 * A fit as a traceable algorithm: Nelder–Mead on `objective` over the unconstrained coordinates, with each state's
 * parameters decoded from the best vertex. A non-finite objective value is replaced by `INVALID`. `init` takes no
 * start; the simplex starts around `u0`.
 *
 * @param name The algorithm's name, as traces show it.
 * @param objective The value to minimise at a point $\uvec$ of the unconstrained coordinates.
 * @param decode Maps a point $\uvec$ to the parameters it stands for.
 * @param u0 The starting point, one entry per coordinate.
 * @param tolerance The Nelder–Mead tolerance on both the simplex size and the value spread.
 * @returns The algorithm, whose states are `FitState`s.
 */
export function simplexFit<P>(
  name: string,
  objective: (u: number[]) => number,
  decode: (u: number[]) => P,
  u0: number[],
  tolerance = 1e-9,
): Algorithm<void, FitState<P>> {
  const f = (u: number[]) => {
    const v = objective(u)
    return Number.isFinite(v) ? v : INVALID
  }
  const nm = nelderMead((x) => f(toFlat(x)), { xTolerance: tolerance, fTolerance: tolerance, divergeAbove: Infinity })
  const wrap = (s: NelderMeadState): FitState<P> => ({
    t: s.t,
    params: decode(toFlat(s.x)),
    objective: s.value,
    evaluations: s.evaluations,
    converged: s.converged,
    diverged: s.diverged,
    optimiser: s,
  })
  return {
    name,
    init: (_start, s) => wrap(nm.init({ x0: u0 }, s)),
    step: (s, ctx) => wrap(nm.step(s.optimiser, ctx)),
  }
}
