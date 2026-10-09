/**
 * `aifn-compute/inference`: probabilistic inference, from describing a model to computing its posterior.
 *
 * - `model`: the model description language (`model`, `dist`, plates and chains), expansion, the log joint and
 *   ancestral sampling, and discrete factor graphs with their algebra.
 * - `exact`: exact inference on discrete factor graphs (enumeration, variable elimination) and on chains
 *   (`forwardBackward`, `viterbi`).
 * - `message-passing`: belief propagation, sum- and max-product, exact on trees and loopy otherwise, with the Bethe
 *   free energy and Gaussian belief propagation.
 * - `expectation-propagation`: expectation propagation and assumed density filtering, with Gaussian and
 *   exponential-family messages, for GP classification, probit regression and linear-Gaussian models.
 * - `variational`: variational inference by stochastic optimisation of the ELBO.
 * - `stochastic`: Monte Carlo inference: Metropolis-Hastings, Gibbs, slice sampling, HMC and NUTS, Langevin methods,
 *   sequential Monte Carlo, and multi-chain diagnostics.
 * - `filtering`: state-space filtering and smoothing: the Kalman filter and its extended and unscented forms, the
 *   Rauch-Tung-Striebel smoother, and Bayesian online changepoint detection.
 * - `engines`: `infer`, which chooses an engine by the shape of the model.
 *
 * Each child's algorithms run on their own, on factor graphs, log-densities or state-space models, or through `infer`
 * on a model description, as in probabilistic programming systems such as Infer.NET and PyMC.
 */

export { model, dist } from './model'
export { infer } from './engines'
export { variableElimination, forwardBackward, viterbi } from './exact'
export { beliefPropagation } from './message-passing'
export { metropolisHastings, hmc, nuts } from './stochastic'
export { kalmanFilter, rtsSmoother } from './filtering'
