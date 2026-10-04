/** The registry of `aifn-methods/learning/trees-and-ensembles/bagging`. */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as forest from './forest'

definer<AlgorithmInfo>('algorithm', 'learning/trees-and-ensembles/bagging')(
  {
    key: 'forestGrowth',
    name: 'Random forest growth',
    summary: 'One tree per step on a bootstrap sample with random feature subsets.',
    problem: 'objective',
    state: { iterate: 'trees', flags: [] },
    random: true,
    notes: ['random-forest', 'bagging', 'extremely-randomised-trees'],
    cite: ['breiman2001', 'breiman1996'],
  },
  forest.forestGrowth,
)

/** The algorithms of the module, keyed by factory name. */
export const baggingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', forest) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
