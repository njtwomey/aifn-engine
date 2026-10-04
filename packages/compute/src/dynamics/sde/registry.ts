/**
 * The algorithms of `aifn-compute/dynamics/sde`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as processes from './processes'
import * as integrators from './integrators'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/sde')

algorithm(
  {
    key: 'eulerMaruyama',
    name: 'Euler–Maruyama',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['euler-maruyama-method', 'stochastic-differential-equations'],
    cite: ['maruyama1955', 'kloeden1992'],
  },
  integrators.eulerMaruyama,
)
algorithm(
  {
    key: 'milstein',
    name: 'Milstein',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-differential-equations'],
    cite: ['kloeden1992'],
  },
  integrators.milstein,
)
algorithm(
  {
    key: 'stochasticRungeKutta',
    name: 'Stochastic Runge–Kutta',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-differential-equations'],
    cite: ['kloeden1992'],
  },
  integrators.stochasticRungeKutta,
)

/** Every algorithm of the module, keyed by factory name. */
export const sdeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', integrators) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'dynamics/sde')

fn(
  { key: 'brownianMotion', name: 'Brownian motion', role: 'construction', notes: ['brownian-motion', 'random-walk'] },
  processes.brownianMotion,
)
fn(
  {
    key: 'ornsteinUhlenbeck',
    name: 'Ornstein–Uhlenbeck process',
    summary: 'Moments, transition law and exact sampler of the OU process.',
    role: 'construction',
    notes: ['ornstein-uhlenbeck-process', 'stochastic-differential-equations'],
  },
  processes.ornsteinUhlenbeck,
)
fn(
  {
    key: 'geometricBrownianMotion',
    name: 'Geometric Brownian motion',
    role: 'construction',
    notes: ['itos-lemma', 'stochastic-differential-equations'],
  },
  processes.geometricBrownianMotion,
)
fn(
  {
    key: 'increments',
    name: 'Brownian increments',
    role: 'simulation',
    random: true,
    notes: ['brownian-motion', 'ito-integral'],
  },
  integrators.increments,
)
fn(
  {
    key: 'paths',
    name: 'Sample paths of an SDE',
    role: 'simulation',
    random: true,
    notes: ['stochastic-differential-equations', 'euler-maruyama-method'],
  },
  integrators.paths,
)

/** The functions of the module, keyed by name. */
export const sdeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', processes, integrators) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
