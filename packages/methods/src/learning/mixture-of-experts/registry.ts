/**
 * The algorithms of `aifn-methods/learning/mixture-of-experts`: EM and Adam training as step-through algorithms, and
 * the streaming run.
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as em from './em'
import * as training from './training'

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/mixture-of-experts')
const NOTES = ['mixture-of-experts']

algorithm(
  {
    key: 'moeEm',
    name: 'EM for a mixture of linear experts',
    summary:
      'E-step responsibilities from the gate and each expert’s likelihood; M-step weighted least squares (or IRLS) per expert and a soft-target logistic refit of the gate, flat or hierarchical.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    notes: NOTES,
    cite: ['jacobs1991'],
  },
  em.moeEm,
)
algorithm(
  {
    key: 'moeTraining',
    name: 'Mixture-of-experts training by Adam',
    summary: 'Minibatch Adam on the data loss plus the load-balancing, importance and router z auxiliary losses.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: NOTES,
    cite: ['shazeer2017', 'fedus2022'],
  },
  training.moeTraining,
)

/** The algorithms of the module, keyed by factory name. */
export const mixtureOfExpertsAlgorithms = entries<AlgorithmInfo>('algorithm', em, training) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>
