/**
 * `aifn-methods/inference/classifier-models`: Bayesian linear classifiers with Gaussian posteriors over their weights,
 * fitted by message passing.
 *
 * - The Bayes point machine, for a batch of labelled points: `bayesPointMachine` runs expectation propagation, one
 *   site per point, as a step-through algorithm (`BayesPointMachineOptions`, `BayesPointMachineState`) with a step or
 *   probit likelihood, and `bayesPointMachinePredict` gives $P(y = +1 \mid \xvec)$ under its posterior.
 * - AdPredictor, for a stream of impressions on sparse binary features: `adPredictor` makes the prior model,
 *   `adPredictorUpdate` folds in one impression by assumed-density filtering (probit likelihood), and
 *   `adPredictorProbability` is the predictive click probability.
 * - `classifierModelAlgorithms` and `classifierModelFunctions` are the module's registry entries.
 *
 * Labels are $\pm 1$ (a click or not, for AdPredictor). Models and states are plain data, and an update returns a new
 * one. Neither model is differentiable by `grad`.
 */
export {
  type BayesPointMachineOptions,
  type BayesPointMachineState,
  bayesPointMachine,
  bayesPointMachinePredict,
} from './bayesPointMachine'
export { adPredictor, adPredictorProbability, adPredictorUpdate, type AdPredictor } from './adpredictor'
export { classifierModelAlgorithms, classifierModelFunctions } from './registry'
