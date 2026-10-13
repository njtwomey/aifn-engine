/**
 * Registry entries of `aifn-methods/optimisation/population`: population-based training as a step-through
 * algorithm, and its toy problem.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as pbt from './pbt'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optimisation/population')
const fn = definer<FunctionInfo>('function', 'optimisation/population')

algorithm(
  {
    key: 'populationBasedTrainingSteps',
    name: 'Population-based training',
    summary:
      'Train a population in parallel; every few steps the bottom members copy the top ones (exploit) and perturb or resample their hyperparameters (explore).',
    problem: 'objective',
    state: { iterate: 'theta', objective: 'best', flags: ['diverged'] },
    random: true,
    cite: ['jaderberg2017'],
  },
  pbt.populationBasedTrainingSteps,
)

fn(
  {
    key: 'pbtToyProblem',
    name: 'PBT toy problem',
    summary: 'Score 1.2 − θ0² − θ1² while gradient steps climb 1.2 − h0θ0² − h1θ1²: the toy of Jaderberg et al.',
    role: 'construction',
    cite: ['jaderberg2017'],
  },
  pbt.pbtToyProblem,
)

/** The algorithms of the module, keyed by factory name. */
export const populationAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', pbt) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const populationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', pbt) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
