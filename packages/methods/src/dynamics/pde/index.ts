/**
 * `aifn-methods/dynamics/pde`: method-of-lines finite differences on a uniform 1-D grid, each solver a traceable
 * `Algorithm` in time whose state reports the solution, its mass and the scheme's stability number against its limit
 * (CFL).
 *
 * - Diffusion: `heatEquation` (explicit, implicit or Crank–Nicolson, the `TimeScheme`; Dirichlet, Neumann or periodic
 *   ends, the `Boundary`), with `HeatOptions`.
 * - Advection: `transportEquation` (upwind, Lax–Friedrichs or Lax–Wendroff), with `TransportOptions`.
 * - Waves: `waveEquation` (leapfrog), with `WaveOptions`; its `WaveState` adds the previous level and the discrete
 *   energy.
 * - Densities: `fokkerPlanck` (conservative, exponentially fitted flux with reflecting ends), with
 *   `FokkerPlanckOptions`, and `densityEvolution`, the density of a scalar SDE
 *   $dX = a(X) \, dt + \sigma(X) \, dW$ by its Fokker–Planck equation.
 * - Grids and profiles: a `Grid1` of $n$ points on $[a, b]$, `gridPoints` for its coordinates, and a `Profile` (values
 *   or a function of $x$) for the initial condition.
 * - `pdeAlgorithms`: the registry entries of the five solvers.
 *
 * Every state is a `PdeState`, whose `Stability` reports the stability number (rather than refusing an unstable step)
 * and whose `diverged` flags a non-finite solution. Invalid settings throw `DomainError`.
 */

export { densityEvolution } from './density'
export {
  fokkerPlanck,
  gridPoints,
  heatEquation,
  transportEquation,
  waveEquation,
  type Boundary,
  type FokkerPlanckOptions,
  type Grid1,
  type HeatOptions,
  type PdeState,
  type Profile,
  type Stability,
  type TimeScheme,
  type TransportOptions,
  type WaveOptions,
  type WaveState,
} from './solvers'
export { pdeAlgorithms } from './registry'
