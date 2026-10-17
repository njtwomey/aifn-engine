/**
 * The registry of `aifn-methods/dynamics/pde`: the method-of-lines solvers as traceable algorithms, each with the
 * solution as its iterate, the mass as its objective and the `diverged` flag; and the periodic spectral solutions as
 * functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as density from './density'
import * as solvers from './solvers'
import * as spectral from './spectral'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/pde')
const fn = definer<FunctionInfo>('function', 'dynamics/pde')
const OPERATORS = ['operator-learning', 'physics-informed-neural-operator']
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

fn({ key: 'periodicGrid', name: 'Periodic grid', role: 'construction', notes: OPERATORS }, spectral.periodicGrid)
fn(
  {
    key: 'evaluateSeries',
    name: 'Fourier series on a grid',
    tex: 'a_0 + \\sum_k (a_k \\cos 2\\pi kx + b_k \\sin 2\\pi kx)',
    role: 'transform',
    notes: OPERATORS,
  },
  spectral.evaluateSeries,
)
fn(
  {
    key: 'heatSolution',
    name: 'Periodic heat equation, exact',
    tex: 'a_k \\mapsto a_k e^{-D(2\\pi k)^2 t}',
    role: 'solver',
    summary: 'The exact solution of u_t = D u_xx on the periodic unit interval, mode by mode.',
    notes: [...OPERATORS, 'heat-equation'],
  },
  spectral.heatSolution,
)
fn(
  {
    key: 'solveBurgersPeriodic',
    name: 'Periodic Burgers equation, pseudo-spectral',
    summary: 'u_t + (u²/2)_x = ν u_xx on [0, 1) by Fourier pseudo-spectral differences and integrating-factor RK4.',
    role: 'solver',
    notes: ['fourier-neural-operator', 'operator-learning'],
    cite: ['li2021fno'],
  },
  spectral.solveBurgersPeriodic,
)

/** The algorithms of the module, keyed by factory name. */
export const pdeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', solvers, density) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module (the periodic spectral solutions), keyed by name. */
export const pdeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', spectral) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
