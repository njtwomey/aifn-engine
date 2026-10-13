/**
 * `aifn-methods/optimisation/population`: population-based training (Jaderberg et al., 2017) as a step-through
 * simulator, on the paper's toy problem or any `PbtProblem`.
 *
 * - The simulator: `populationBasedTrainingSteps` trains a population in parallel and, every `readyInterval` steps,
 *   exploits by truncation selection and explores by perturbing or resampling hyperparameters; with neither it is the
 *   grid-search baseline. Each state holds every member's parameters, hyperparameters and score and the step's
 *   events (`PbtState`, `PbtEvent`, `PbtOptions`).
 * - The toy: `pbtToyProblem` scores $Q(\thetavec) = 1.2 - \theta_0^2 - \theta_1^2$ while training climbs the surrogate
 *   $\hat Q(\thetavec \mid \hvec) = 1.2 - h_0\theta_0^2 - h_1\theta_1^2$ (`PbtProblem`).
 *
 * Scores are higher-is-better. `populationAlgorithms` and `populationFunctions` register the entries.
 */

export {
  pbtToyProblem,
  populationBasedTrainingSteps,
  type PbtEvent,
  type PbtOptions,
  type PbtProblem,
  type PbtState,
} from './pbt'
export { populationAlgorithms, populationFunctions } from './registry'
