/**
 * The algorithms of `aifn-compute/optim/second-order`, registered with what each factory takes (`problem`) and the
 * roles of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a
 * generic trace view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as leastSquares from './leastSquares'
import * as newton from './newton'
import * as quasiNewton from './quasiNewton'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/second-order')

algorithm(
  {
    key: 'newton',
    name: 'Newton',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['newtons-method'],
    cite: ['nocedal2006'],
  },
  newton.newton,
)
algorithm(
  {
    key: 'trustRegion',
    name: 'Trust region',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'radius',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['trust-region-methods'],
    cite: ['nocedal2006'],
  },
  newton.trustRegion,
)
algorithm(
  {
    key: 'bfgs',
    name: 'BFGS',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'bfgs',
    notes: ['quasi-newton-methods'],
    cite: ['nocedal2006'],
  },
  quasiNewton.bfgs,
)
algorithm(
  {
    key: 'lbfgs',
    name: 'L-BFGS',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'l-bfgs',
    notes: ['quasi-newton-methods'],
    cite: ['liu1989'],
  },
  quasiNewton.lbfgs,
)
algorithm(
  {
    key: 'owlqn',
    name: 'OWL-QN',
    summary:
      'L-BFGS for f(x) + C‖x‖₁: pseudo-gradient, orthant-projected direction and backtracking along the projected path.',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['quasi-newton-methods', 'lasso', 'elastic-net'],
    cite: ['liu1989'],
  },
  quasiNewton.owlqn,
)
algorithm(
  {
    key: 'gaussNewton',
    name: 'Gauss–Newton',
    problem: 'least-squares',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['gauss-newton-and-levenberg-marquardt'],
    cite: ['nocedal2006'],
  },
  leastSquares.gaussNewton,
)
algorithm(
  {
    key: 'levenbergMarquardt',
    name: 'Levenberg–Marquardt',
    problem: 'least-squares',
    state: { iterate: 'x', objective: 'value', grad: 'grad', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['gauss-newton-and-levenberg-marquardt'],
    cite: ['levenberg1944', 'marquardt1963'],
  },
  leastSquares.levenbergMarquardt,
)

/** Every algorithm of the module, keyed by factory name. */
export const secondOrderAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', leastSquares, newton, quasiNewton) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
