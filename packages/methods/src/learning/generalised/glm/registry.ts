/** The registry of `aifn-methods/learning/generalised/glm`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as logistic from './logistic'
import * as model from './model'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/generalised/glm')
const fn = definer<FunctionInfo>('function', 'learning/generalised/glm')

algorithm(
  {
    key: 'softmaxNewton',
    name: 'Multinomial logistic regression by Newton',
    problem: 'objective',
    state: { iterate: 'weights', objective: 'loss', grad: 'grad', flags: ['converged', 'diverged'] },
    notes: ['multinomial-logistic-regression', 'newtons-method'],
  },
  logistic.softmaxNewton,
)
algorithm(
  {
    key: 'negativeBinomialAlternation',
    name: 'Negative binomial GLM by alternation',
    summary: 'Alternate IRLS for the coefficients and a one-dimensional MLE for the dispersion θ.',
    problem: 'least-squares',
    state: { iterate: 'coefficients', objective: 'deviance', flags: ['converged', 'diverged'] },
    notes: ['negative-binomial-and-overdispersion'],
  },
  model.negativeBinomialAlternation,
)
fn(
  {
    key: 'thetaMaximumLikelihood',
    name: 'Negative binomial θ by maximum likelihood',
    role: 'fit',
    notes: ['negative-binomial-and-overdispersion', 'negative-binomial-distribution'],
  },
  model.thetaMaximumLikelihood,
)

/** The algorithms of the module, keyed by factory name. */
export const glmAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  logistic,
  model,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const glmFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', model) as Table<FunctionInfo>
