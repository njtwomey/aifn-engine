/** The registry of `aifn-methods/learning/linear`. */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as perceptron from './perceptron'

definer<AlgorithmInfo>('algorithm', 'learning/linear')(
  {
    key: 'perceptronSteps',
    name: 'Perceptron',
    summary: 'Visit one example per step and add yx to the weights when it is misclassified; the averaged weights too.',
    problem: 'objective',
    state: { iterate: 'weights', objective: 'mistakes', flags: [] },
    random: true,
    notes: ['perceptron'],
    cite: ['rosenblatt1958'],
  },
  perceptron.perceptronSteps,
)

/** The algorithms of the module, keyed by factory name. */
export const linearAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', perceptron) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
