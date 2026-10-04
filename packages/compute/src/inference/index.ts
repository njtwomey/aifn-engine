/**
 * `aifn-compute/inference`: probabilistic inference: the model description language and factor graphs, exact inference, message passing,
 * expectation propagation, variational inference, Monte Carlo (MCMC and SMC), filtering, and the `infer` engines.
 * Children: model, exact, message-passing, expectation-propagation, variational, stochastic, filtering, engines.
 */

export { model, dist } from './model'
export { infer } from './engines'
export { variableElimination, forwardBackward, viterbi } from './exact'
export { beliefPropagation } from './message-passing'
export { metropolisHastings, hmc, nuts } from './stochastic'
export { kalmanFilter, rtsSmoother } from './filtering'
