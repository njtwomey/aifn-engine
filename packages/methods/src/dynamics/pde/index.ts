/**
 * `aifn-methods/dynamics/pde`: method-of-lines finite differences on a uniform 1-D grid, each solver a traceable
 * `Algorithm` in time whose state reports the solution, its mass and the scheme's stability number against its limit
 * (CFL).
 *
 * - `heatEquation` (explicit, implicit, Crank–Nicolson; Dirichlet, Neumann or periodic ends).
 * - `transportEquation` (upwind, Lax–Friedrichs, Lax–Wendroff).
 * - `waveEquation` (leapfrog, with the discrete energy).
 * - `fokkerPlanck` (conservative, upwinded flux form with reflecting ends).
 * - `gridPoints` for the grid's coordinates.
 * - `densityEvolution`: the density of a scalar SDE dX = a(X) dt + σ(X) dW by its Fokker–Planck equation.
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
