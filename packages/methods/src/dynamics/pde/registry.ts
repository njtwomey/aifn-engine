/**
 * The registry of `aifn-methods/dynamics/pde`: the method-of-lines solvers as traceable algorithms.
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as density from './density'
import * as solvers from './solvers'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/pde')
const state = { iterate: 'u', objective: 'mass', flags: ['diverged'] } as const
const PDE = ['partial-differential-equations']

algorithm(
  {
    key: 'heatEquation',
    name: 'Heat equation',
    summary: 'u_t = D u_xx by the method of lines: FTCS, implicit Euler or Crank–Nicolson.',
    problem: 'pde',
    state,
    notes: ['heat-equation', ...PDE],
    cite: ['crank1947', 'leveque2007'],
  },
  solvers.heatEquation,
)
algorithm(
  {
    key: 'transportEquation',
    name: 'Transport equation',
    summary: 'u_t + c u_x = 0 by upwind, Lax–Friedrichs or Lax–Wendroff differences.',
    problem: 'pde',
    state,
    notes: ['transport-and-continuity-equations', ...PDE],
    cite: ['courant1928', 'leveque2007'],
  },
  solvers.transportEquation,
)
algorithm(
  {
    key: 'waveEquation',
    name: 'Wave equation',
    summary: 'u_tt = c² u_xx by the leapfrog scheme.',
    problem: 'pde',
    state,
    notes: PDE,
    cite: ['leveque2007'],
  },
  solvers.waveEquation,
)
algorithm(
  {
    key: 'fokkerPlanck',
    name: 'Fokker–Planck equation',
    summary: 'p_t = −(ap)_x + (Dp)_xx in conservative form with reflecting ends.',
    problem: 'pde',
    state,
    notes: ['fokker-planck-equation', ...PDE],
    cite: ['risken1996'],
  },
  solvers.fokkerPlanck,
)
algorithm(
  {
    key: 'densityEvolution',
    name: 'Density of a scalar SDE',
    summary: 'The density of dX = a(X) dt + σ(X) dW evolved by its Fokker–Planck equation.',
    problem: 'sde',
    state,
    notes: ['fokker-planck-equation', 'stochastic-differential-equations', 'ornstein-uhlenbeck-process'],
  },
  density.densityEvolution,
)

/** The algorithms of the module, keyed by factory name. */
export const pdeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', solvers, density) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
