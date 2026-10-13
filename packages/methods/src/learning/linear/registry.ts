/**
 * Registry entries of `aifn-methods/learning/linear`: the perceptron, stochastic gradient descent, MacKay's evidence
 * updates and hypergradient descent on the ridge penalty as step-through algorithms; the regularisation paths
 * (elastic net, group lasso, ridge, least-angle regression, forward stagewise), subset selection, the evidence and the
 * ridge hypergradient as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as elasticNet from './elasticNet'
import * as evidence from './evidence'
import * as groupLasso from './groupLasso'
import * as hypergradient from './hypergradient'
import * as lars from './lars'
import * as leastSquares from './leastSquares'
import * as perceptron from './perceptron'
import * as sgd from './sgd'
import * as subsets from './subsets'

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

algorithm(
  {
    key: 'evidenceMaximisationSteps',
    name: 'Evidence maximisation (MacKay)',
    summary:
      'Fixed-point updates of the prior and noise precisions of Bayesian linear regression that climb the evidence.',
    problem: 'objective',
    state: { iterate: 'logAlpha', objective: 'logEvidence', flags: ['converged', 'diverged'] },
    cite: ['mackay1992', 'bishop2006'],
  },
  evidence.evidenceMaximisationSteps,
)

algorithm(
  {
    key: 'ridgeHyperDescentSteps',
    name: 'Hypergradient descent on the ridge penalty',
    summary: 'Gradient descent on log lambda for the validation loss of ridge, by the implicit function theorem.',
    problem: 'objective',
    state: { iterate: 'logLambda', objective: 'loss', grad: 'gradient', flags: ['converged', 'diverged'] },
    cite: ['bengio2000', 'pedregosa2016'],
  },
  hypergradient.ridgeHyperDescentSteps,
)

fn(
  {
    key: 'bayesianLinearEvidence',
    name: 'Evidence of Bayesian linear regression',
    summary:
      'The log marginal likelihood at given precisions, split into the best-fit likelihood and the Occam factor.',
    role: 'inference',
    cite: ['bishop2006', 'mackay1992'],
  },
  evidence.bayesianLinearEvidence,
)

fn(
  {
    key: 'evidenceMaximisation',
    name: 'Empirical Bayes for linear regression',
    summary: "MacKay's updates of the precisions run to convergence, as BayesianRidge, with the evidence at each step.",
    role: 'fit',
    cite: ['mackay1992', 'bishop2006'],
  },
  evidence.evidenceMaximisation,
)

fn(
  {
    key: 'groupLassoPath',
    name: 'Group-lasso path',
    summary: 'Group-lasso fits over a decreasing grid of penalties, each warm-started from the last.',
    role: 'fit',
    cite: ['yuan2006'],
  },
  groupLasso.groupLassoPath,
)

fn(
  {
    key: 'bestSubset',
    name: 'Best-subset selection',
    summary:
      'The least-squares fit of smallest RSS for each subset size, by exhaustive search, and the RSS of every subset.',
    role: 'fit',
    cite: ['hastie2009'],
  },
  subsets.bestSubset,
)

fn(
  {
    key: 'stepwise',
    name: 'Stepwise selection',
    summary: 'Forward, backward or bidirectional greedy selection of columns by RSS, AIC or BIC.',
    role: 'fit',
    cite: ['hastie2009'],
  },
  subsets.stepwise,
)

fn(
  {
    key: 'forwardStagewise',
    name: 'Incremental forward stagewise regression',
    summary: 'Tiny steps on the weight most correlated with the residual; tends to the monotone lasso path.',
    role: 'fit',
    cite: ['hastie2009', 'efron2004'],
  },
  subsets.forwardStagewise,
)

fn(
  {
    key: 'ridgeHypergradient',
    name: 'Ridge hypergradient',
    summary: 'The validation loss of ridge and its derivative in the penalty, by the implicit function theorem.',
    role: 'property',
    cite: ['bengio2000', 'pedregosa2016'],
  },
  hypergradient.ridgeHypergradient,
)

/** The algorithms of the module, keyed by factory name. */
export const linearAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', perceptron, sgd, evidence, hypergradient) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module (the paths, subset selection, the evidence, the hypergradient), keyed by name. */
export const linearFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    elasticNet,
    leastSquares,
    lars,
    evidence,
    groupLasso,
    subsets,
    hypergradient,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
