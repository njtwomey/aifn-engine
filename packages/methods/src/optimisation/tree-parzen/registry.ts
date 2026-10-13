/**
 * Registry entries of `aifn-methods/optimisation/tree-parzen`: the tree-structured Parzen estimator as a step-through
 * algorithm.
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as tpe from './tpe'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optimisation/tree-parzen')

algorithm(
  {
    key: 'treeParzenSteps',
    name: 'Tree-structured Parzen estimator',
    summary:
      'Split the observations at the γ-quantile, fit Parzen densities l and g to the good and bad groups, and evaluate the candidate from l with the largest l/g.',
    problem: 'search-space',
    state: { iterate: 'nextU', objective: 'best', flags: [] },
    random: true,
    cite: ['bergstra2011', 'watanabe2023'],
  },
  tpe.treeParzenSteps,
)

/** The algorithms of the module, keyed by factory name. */
export const treeParzenAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', tpe) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
