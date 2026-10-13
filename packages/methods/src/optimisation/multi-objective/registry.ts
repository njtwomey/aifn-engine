/**
 * Registry entries of `aifn-methods/optimisation/multi-objective`: the Pareto front, the hypervolume and its
 * improvements, and the scalarisation of ParEGO.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as hypervolume from './hypervolume'
import * as pareto from './pareto'
import * as scalarisation from './scalarisation'

const fn = definer<FunctionInfo>('function', 'optimisation/multi-objective')

fn(
  {
    key: 'paretoFront',
    name: 'Pareto front',
    summary: 'The points no other point dominates, ordered along the front, for objectives minimised or maximised.',
    role: 'transform',
    cite: ['zitzler1999'],
  },
  pareto.paretoFront,
)

fn(
  {
    key: 'hypervolume',
    name: 'Hypervolume indicator',
    summary: 'The exact volume a set of points dominates up to a reference point: a sweep in 2-D, slicing beyond.',
    role: 'property',
    cite: ['zitzler1999', 'while2006'],
  },
  hypervolume.hypervolume,
)

fn(
  {
    key: 'hypervolumeImprovement',
    name: 'Hypervolume improvement',
    summary: 'How much the hypervolume of a front grows when each candidate point is added to it, exactly.',
    role: 'property',
    cite: ['zitzler1999', 'while2012'],
  },
  hypervolume.hypervolumeImprovement,
)

fn(
  {
    key: 'expectedHypervolumeImprovement',
    name: 'Expected hypervolume improvement',
    summary:
      'The posterior expectation of the hypervolume improvement: exact for two objectives, Monte Carlo for more.',
    role: 'estimator',
    random: true,
    cite: ['emmerich2006', 'emmerich2011', 'yang2019'],
  },
  hypervolume.expectedHypervolumeImprovement,
)

fn(
  {
    key: 'chebyshevScalarisation',
    name: 'Augmented Chebyshev scalarisation',
    summary: 'ParEGO’s max of weighted objectives plus ρ times their sum, which reaches non-convex parts of a front.',
    role: 'transform',
    cite: ['knowles2006parego'],
  },
  scalarisation.chebyshevScalarisation,
)

fn(
  {
    key: 'simplexLattice',
    name: 'Simplex lattice of weights',
    summary: 'Every weight vector with entries in steps of 1/s summing to 1: the weights ParEGO draws from.',
    role: 'construction',
    cite: ['knowles2006parego'],
  },
  scalarisation.simplexLattice,
)

/** The functions of the module, keyed by name. */
export const multiObjectiveFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', pareto, hypervolume, scalarisation) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
