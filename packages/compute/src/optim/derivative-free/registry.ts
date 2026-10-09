/**
 * The algorithms of `aifn-compute/optim/derivative-free`, registered with what each factory takes (`problem`) and the
 * roles of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a
 * generic trace view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as nelderMead from './nelderMead'
import * as stochastic from './stochastic'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/derivative-free')

algorithm(
  {
    key: 'nelderMead',
    name: 'Nelder–Mead',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', flags: ['converged', 'diverged'] },
    notes: ['nelder-mead'],
    cite: ['nelder1965'],
  },
  nelderMead.nelderMead,
)
algorithm(
  {
    key: 'simulatedAnnealing',
    name: 'Simulated annealing',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['simulated-annealing'],
    cite: ['kirkpatrick1983'],
  },
  stochastic.simulatedAnnealing,
)
algorithm(
  {
    key: 'cmaEs',
    name: 'CMA-ES',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', stepSize: 'sigma', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['evolution-strategies-and-covariance-matrix-adaptation'],
    cite: ['hansen2001'],
  },
  stochastic.cmaEs,
)

/** Every algorithm of the module, keyed by factory name. */
export const derivativeFreeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', nelderMead, stochastic) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
