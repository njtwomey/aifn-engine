/** The registry of `aifn-methods/learning/trees-and-ensembles/boosting`. */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as ensembles from './ensembles'

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/trees-and-ensembles/boosting')

algorithm(
  {
    key: 'adaBoostSteps',
    name: 'AdaBoost',
    summary: 'One weak learner per step on reweighted examples; the vote weight grows with its accuracy.',
    problem: 'objective',
    state: { iterate: 'alphas', objective: 'trainingError', flags: [] },
    notes: ['adaboost'],
    cite: ['freund1997'],
  },
  ensembles.adaBoostSteps,
)
algorithm(
  {
    key: 'gradientBoostingSteps',
    name: 'Gradient boosting',
    summary: 'One regression tree per step fitted to the negative gradient of the loss, added with shrinkage.',
    problem: 'objective',
    state: { iterate: 'raw', objective: 'loss', flags: [] },
    notes: ['gradient-boosting', 'xgboost-lightgbm-catboost'],
    cite: ['friedman2001'],
  },
  ensembles.gradientBoostingSteps,
)

/** The algorithms of the module, keyed by factory name. */
export const boostingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', ensembles) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
