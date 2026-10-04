/**
 * The density of a scalar SDE, evolved by its Fokker–Planck equation (moved from `aifn-compute/dynamics/sde` so that compute does not
 * import the named PDEs).
 */

import type { Algorithm } from 'aifn-compute/foundation/trace'
import { fokkerPlanck, type Grid1, type PdeState, type Profile, type TimeScheme } from './solvers'

/**
 * The density of a scalar SDE dX = a(X) dt + σ(X) dW evolved by its Fokker–Planck equation p_t = −(ap)_x +
 * ½(σ²p)_xx on a grid (`fokkerPlanck`'s conservative solver with reflecting ends, default implicit steps). The solver's
 * state reports the mass and the stability number. `init` takes `{ u0 }`, the initial density.
 */
export function densityEvolution({
  drift,
  sigma,
  grid,
  dt,
  scheme: timeScheme = 'implicit',
  tEnd,
}: {
  drift: (x: number) => number
  sigma: (x: number) => number
  grid: Grid1
  dt: number
  scheme?: TimeScheme
  tEnd?: number
}): Algorithm<{ u0: Profile }, PdeState> {
  return fokkerPlanck({ drift, diffusion: (x) => 0.5 * sigma(x) ** 2, grid, dt, scheme: timeScheme, tEnd })
}
