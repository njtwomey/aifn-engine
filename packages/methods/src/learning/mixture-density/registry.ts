/** The algorithms of `aifn-methods/learning/mixture-density`: Adam training of a mixture density network. */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as training from './training'

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/mixture-density')

algorithm(
  {
    key: 'mdnTraining',
    name: 'Mixture density network training by Adam',
    summary:
      'Minibatch Adam on the mixture negative log-likelihood of an MLP with a Gaussian-mixture head (or the squared error of the same body).',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: ['gaussian-mixture-model', 'multilayer-perceptron'],
    cite: ['bishop2006'],
  },
  training.mdnTraining,
)

/** The algorithms of the module, keyed by factory name. */
export const mixtureDensityAlgorithms = entries<AlgorithmInfo>('algorithm', training) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>
