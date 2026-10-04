/**
 * `aifn-methods/inference/classifier-models`: the Bayes point machine by expectation propagation, and AdPredictor
 * (Bayesian online probit regression by assumed-density filtering).
 */
export {
  type BayesPointMachineOptions,
  type BayesPointMachineState,
  bayesPointMachine,
  bayesPointMachinePredict,
} from './bayesPointMachine'
export { adPredictor, adPredictorProbability, adPredictorUpdate, type AdPredictor } from './adpredictor'
export { classifierModelAlgorithms, classifierModelFunctions } from './registry'
