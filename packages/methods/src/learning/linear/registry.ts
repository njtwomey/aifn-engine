/**
 * Registry entries of `aifn-methods/learning/linear`: the perceptron and stochastic gradient descent as step-through
 * algorithms, and the regularisation paths (elastic net, ridge, least-angle regression) as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as elasticNet from './elasticNet'
import * as lars from './lars'
import * as leastSquares from './leastSquares'
import * as perceptron from './perceptron'
import * as sgd from './sgd'

const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/linear')
const fn = definer<FunctionInfo>('function', 'learning/linear')

algorithm(
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

algorithm(
  {
    key: 'linearSgdSteps',
    name: 'SGD for penalised linear regression',
    summary:
      'One shuffled pass per step of per-example gradient steps on penalised least squares, as SGDRegressor (truncated L1).',
    problem: 'objective',
    state: { iterate: 'weights', objective: 'objective', stepSize: 'eta', flags: ['diverged'] },
    random: true,
    cite: ['bottou2010', 'tsuruoka2009'],
  },
  sgd.linearSgdSteps,
)

fn(
  {
    key: 'elasticNetPath',
    name: 'Elastic-net path',
    summary: 'Elastic-net (or lasso) fits over a decreasing grid of penalties, each warm-started from the last.',
    role: 'fit',
    cite: ['friedman2010', 'zou2005'],
  },
  elasticNet.elasticNetPath,
)

fn(
  {
    key: 'ridgePath',
    name: 'Ridge path',
    summary: 'Ridge fits and effective degrees of freedom at many penalties from one SVD of the centred design.',
    role: 'fit',
    cite: ['hoerl1970', 'hastie2009'],
  },
  leastSquares.ridgePath,
)

fn(
  {
    key: 'leastAngleRegression',
    name: 'Least-angle regression',
    summary: 'The knots of the piecewise-linear LARS path, or of the lasso path with the lasso modification.',
    role: 'fit',
    cite: ['efron2004'],
  },
  lars.leastAngleRegression,
)

/** The algorithms of the module, keyed by factory name. */
export const linearAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', perceptron, sgd) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module (the regularisation paths), keyed by name. */
export const linearFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', elasticNet, leastSquares, lars) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
