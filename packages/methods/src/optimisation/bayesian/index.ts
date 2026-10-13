/**
 * `aifn-methods/optimisation/bayesian`: Bayesian optimisation with a Gaussian-process surrogate, as a step-through
 * loop, and the acquisition functions it maximises.
 *
 * - Acquisitions of a posterior mean $\mu$ and standard deviation $\sigma$ at candidates, elementwise:
 *   `probabilityOfImprovement` $\Phi(z)$ and `expectedImprovement` $\delta\Phi(z) + \sigma\phi(z)$ against an
 *   incumbent, `upperConfidenceBound` $\mp\mu + \sqrt{\beta}\sigma$ with `ucbBeta`, the GP-UCB schedule of $\beta_t$.
 * - Entropy search: `maxValueEntropySearch`, the information an evaluation gives about the optimal value $y^*$, with
 *   `maxValueSamples` drawing $y^*$ by the Gumbel approximation.
 * - The loop: `bayesianOptimisationSteps`, one objective evaluation per step; each state holds the evaluations, the
 *   surrogate's mean and standard deviation and the acquisition on a grid (one or two dimensional for figures), the
 *   next point, the incumbent, the hyperparameters refitted by marginal likelihood, and, for Thompson sampling, the
 *   posterior draw that chose the point.
 *
 * One convention holds throughout: the objective is minimised by default (`minimise: false` maximises it), the
 * incumbent is the best value observed, and every acquisition is a score to maximise. Randomness comes from streams
 * only. `bayesianAlgorithms` and `bayesianFunctions` list the registered entries. Test objectives (Forrester, Branin
 * and a tabulated SVM tuning surface) are in `aifn-methods/data/objectives`.
 */

export {
  expectedImprovement,
  probabilityOfImprovement,
  ucbBeta,
  upperConfidenceBound,
  type ConfidenceBoundOptions,
  type ImprovementOptions,
} from './acquisitions'
export {
  maxValueEntropySearch,
  maxValueSamples,
  type MaxValueEntropyOptions,
  type MaxValueSamples,
  type MaxValueSamplesOptions,
} from './entropy'
export {
  bayesianOptimisationSteps,
  type AcquisitionName,
  type BayesianObjective,
  type BayesianOptimisation,
  type BayesianOptimisationOptions,
  type BayesianOptimisationState,
  type SearchBox,
} from './loop'
export { bayesianAlgorithms, bayesianFunctions } from './registry'
