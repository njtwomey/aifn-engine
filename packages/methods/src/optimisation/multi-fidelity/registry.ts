/**
 * Registry entries of `aifn-methods/optimisation/multi-fidelity`: the run of a schedule over learning curves as a
 * step-through algorithm, and the schedules and the power-law extrapolation as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as extrapolate from './extrapolate'
import * as run from './run'
import * as schedules from './schedules'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optimisation/multi-fidelity')
const fn = definer<FunctionInfo>('function', 'optimisation/multi-fidelity')

algorithm(
  {
    key: 'multiFidelitySteps',
    name: 'Successive halving, Hyperband and ASHA over learning curves',
    summary:
      'Run a schedule over given learning curves, a rung per step or (ASHA) one finished evaluation per step, tracking budget, incumbent and the true best.',
    problem: 'search-space',
    state: { flags: ['terminated'] },
    cite: ['jamieson2016', 'li2018hyperband', 'li2020asha'],
  },
  run.multiFidelitySteps,
)

fn(
  {
    key: 'successiveHalvingSchedule',
    name: 'Successive-halving schedule',
    summary: 'The rungs of successive halving: configurations and budget per rung, the best 1/η kept each time.',
    role: 'construction',
    cite: ['jamieson2016', 'li2020asha'],
  },
  schedules.successiveHalvingSchedule,
)

fn(
  {
    key: 'hyperbandSchedule',
    name: 'Hyperband schedule',
    summary: 'The brackets of Hyperband, from the most aggressive halving to plain full-budget evaluation.',
    role: 'construction',
    cite: ['li2018hyperband'],
  },
  schedules.hyperbandSchedule,
)

fn(
  {
    key: 'fitPowerLaw',
    name: 'Power-law learning-curve fit',
    summary: 'Least-squares fit of a + b t^(−c) to a partial learning curve, for extrapolating it.',
    role: 'fit',
    cite: ['domhan2015'],
  },
  extrapolate.fitPowerLaw,
)

/** The algorithms of the module, keyed by factory name. */
export const multiFidelityAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', run) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module (the schedules and the extrapolation), keyed by name. */
export const multiFidelityFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', schedules, extrapolate) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
