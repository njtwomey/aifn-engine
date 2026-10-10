/**
 * The density of a scalar SDE, evolved by its Fokker–Planck equation (moved from `aifn-compute/dynamics/sde` so that
 * compute does not import the named PDEs). A thin wrapper of `fokkerPlanck` that takes the noise amplitude $\sigma$
 * instead of the diffusion coefficient $D = \sigma^2/2$.
 */

import type { Algorithm } from 'aifn-compute/foundation/trace'
import { fokkerPlanck, type Grid1, type PdeState, type Profile, type TimeScheme } from './solvers'

/**
 * The density of a scalar SDE $dX = a(X) \, dt + \sigma(X) \, dW$ evolved by its Fokker–Planck equation
 * $p_t = -(a p)_x + \tfrac{1}{2}(\sigma^2 p)_{xx}$ on a grid (`fokkerPlanck`'s conservative solver with reflecting
 * ends, default implicit steps). The solver's state reports the mass and the stability number. `init` takes
 * `{ u0 }`, the initial density.
 *
 * @param options The SDE, the grid and the time stepping.
 * @param options.drift The drift $a(x)$.
 * @param options.sigma The noise amplitude $\sigma(x)$.
 * @param options.grid The grid; its ends are reflecting walls.
 * @param options.dt The time step $\Delta t$.
 * @param options.scheme The time-stepping scheme (default `'implicit'`).
 * @param options.tEnd The time at which the run is done (default never).
 * @returns The solver as an `Algorithm`, one time step per step.
 *
 * @example Brownian motion spreads: the variance grows by t
 * // dX = dW from a narrow bump at 0 (variance 0.04), far from the walls at -5 and 5.
 * const grid = { a: -5, b: 5, n: 101 }
 * const bm = densityEvolution({ drift: () => 0, sigma: () => 1, grid, dt: 0.02 })
 * const u0 = (x) => Math.exp(-(x ** 2) / 0.08) / Math.sqrt(0.08 * Math.PI)
 * const xs = gridPoints(grid).data
 * for (const steps of [0, 25, 50]) {
 *   const s = run(bm, { u0 }, steps)
 *   const variance = s.u.data.reduce((acc, p, i) => acc + p * xs[i] ** 2 * 0.1, 0)
 *   print('t =', s.time, ' mass =', s.mass, ' variance =', variance)
 * }
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
