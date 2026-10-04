/**
 * The algorithms of `aifn-compute/transport`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as oneD from './oneD'
import * as discrete from './discrete'
import * as gromov from './gromov'

const algorithm = definer<AlgorithmInfo>('algorithm', 'transport')

algorithm(
  {
    key: 'sinkhornSteps',
    name: 'Sinkhorn',
    problem: 'transport',
    state: { iterate: 'plan', objective: 'value', flags: ['converged', 'diverged'] },
    notes: ['entropic-regularisation-and-sinkhorn', 'optimal-transport'],
    cite: ['cuturi2013'],
  },
  discrete.sinkhornSteps,
)
algorithm(
  {
    key: 'gromovWassersteinSteps',
    name: 'Gromov–Wasserstein',
    problem: 'transport',
    state: { iterate: 'plan', objective: 'loss', flags: ['converged', 'diverged'] },
    notes: ['optimal-transport'],
  },
  gromov.gromovWassersteinSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const transportAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', discrete, gromov) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'transport')
const OT = ['optimal-transport', 'wasserstein-distances']

fn({ key: 'costMatrix', name: 'Cost matrix', role: 'construction', notes: OT }, discrete.costMatrix)
fn({ key: 'uniformWeights', name: 'Uniform weights', role: 'construction', notes: OT }, discrete.uniformWeights)
fn(
  {
    key: 'exactTransport',
    name: 'Exact optimal transport (LP)',
    role: 'solver',
    notes: ['earth-movers-distance', 'kantorovich-duality', ...OT],
  },
  discrete.exactTransport,
)
fn(
  {
    key: 'sinkhorn',
    name: 'Entropic optimal transport',
    role: 'solver',
    notes: ['entropic-regularisation-and-sinkhorn', 'sinkhorn-divergences'],
  },
  discrete.sinkhorn,
)
fn(
  {
    key: 'wasserstein1d',
    name: 'One-dimensional Wasserstein distance',
    summary: 'From the quantile functions: the monotone coupling is optimal on the line.',
    role: 'estimator',
    notes: OT,
  },
  oneD.wasserstein1d,
)
fn({ key: 'monotonePlan', name: 'Monotone transport plan', role: 'solver', notes: OT }, oneD.monotonePlan)
fn(
  {
    key: 'barycenter1d',
    name: 'One-dimensional Wasserstein barycentre',
    role: 'solver',
    notes: ['wasserstein-barycentres'],
  },
  oneD.barycenter1d,
)
fn(
  {
    key: 'slicedWasserstein',
    name: 'Sliced Wasserstein distance',
    role: 'estimator',
    random: true,
    notes: ['sliced-wasserstein-distance'],
  },
  oneD.slicedWasserstein,
)
fn({ key: 'gromovWasserstein', name: 'Gromov–Wasserstein', role: 'solver', notes: OT }, gromov.gromovWasserstein)

/** The functions of the module, keyed by name. */
export const transportFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', discrete, oneD, gromov) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
