/**
 * `aifn-methods/optimisation/bayesian`: Bayesian optimisation with a Gaussian-process or random-forest surrogate, as a
 * step-through loop, and the acquisition functions it maximises.
 *
 * - Acquisitions of a posterior mean $\mu$ and standard deviation $\sigma$ at candidates, elementwise:
 *   `probabilityOfImprovement` $\Phi(z)$ and `expectedImprovement` $\delta\Phi(z) + \sigma\phi(z)$ against an
 *   incumbent, `upperConfidenceBound` $\mp\mu + \sqrt{\beta}\sigma$ with `ucbBeta`, the GP-UCB schedule of $\beta_t$.
 * - Entropy search: `maxValueEntropySearch`, the information an evaluation gives about the optimal value $y^*$, with
 *   `maxValueSamples` drawing $y^*$ by the Gumbel approximation.
 * - `expectedImprovementLogNormal`, the expected improvement of a cost whose logarithm is modelled (SMAC).
 * - Batches: `multiPointExpectedImprovement`, the q-EI $\expect[\max(f^+ - \min_j y_j, 0)]$ by Monte Carlo with its
 *   standard error, and `localPenalisation`, which shrinks an acquisition around points already chosen.
 * - The loop: `bayesianOptimisationSteps`, one objective evaluation (or batch) per step; each state holds the
 *   evaluations, the surrogate's mean and standard deviation and the acquisition on a grid (one or two dimensional for
 *   figures), the next point, the incumbent, the hyperparameters refitted by marginal likelihood, and, for Thompson
 *   sampling, the posterior draw that chose the point. Options give a random-forest surrogate, a log cost and
 *   interleaved random points (SMAC), batches (kriging believer, constant liars, local penalisation, Thompson) and
 *   unknown constraints (the expected improvement weighted by the probability of feasibility). The objective may be a
 *   registered one named by its key (`'forrester'`, `'branin'`, `'svm-tuning-surface'`), so a run is plain data.
 * - Several fidelities: `multiFidelityBayesianOptimisationSteps` chooses a point and a fidelity per step with the
 *   autoregressive multi-fidelity GP as surrogate, by MF-GP-UCB, multi-fidelity max-value entropy search or the
 *   expected improvement per unit cost; each state holds every fidelity's posterior and acquisition on the grid, the
 *   cumulative cost and the incumbent at the highest fidelity (`Fidelities`, `MultiFidelityObjective`,
 *   `MultiFidelityAcquisitionName`).
 * - Studies: `bayesianOptimisationStudy`, several arms from several seeds on a named objective, returning best-so-far,
 *   simple-regret and cumulative-regret curves as plain tensors, for a worker.
 *
 * One convention holds throughout: the objective is minimised by default (`minimise: false` maximises it), the
 * incumbent is the best value observed, and every acquisition is a score to maximise. Randomness comes from streams
 * only. `bayesianAlgorithms` and `bayesianFunctions` list the registered entries. Test objectives (Forrester, Branin
 * and a tabulated SVM tuning surface) are in `aifn-methods/data/objectives`.
 */

export {
  expectedImprovement,
  expectedImprovementLogNormal,
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
  localPenalisation,
  multiPointExpectedImprovement,
  type MultiPointEstimate,
  type MultiPointOptions,
  type PenalisationOptions,
} from './batch'
export {
  bayesianOptimisationSteps,
  type AcquisitionName,
  type BatchOptions,
  type BatchStrategy,
  type BayesianObjective,
  type BayesianOptimisation,
  type BayesianOptimisationOptions,
  type BayesianOptimisationState,
  type NamedObjective,
  type ObjectiveName,
  type ObjectiveValue,
  type SearchBox,
} from './loop'
export {
  bayesianOptimisationStudy,
  type BayesianOptimisationStudy,
  type BayesianOptimisationStudyOptions,
  type StudyArm,
} from './study'
export {
  multiFidelityBayesianOptimisationSteps,
  type Fidelities,
  type MultiFidelityAcquisitionName,
  type MultiFidelityBayesianOptimisation,
  type MultiFidelityBayesianOptimisationOptions,
  type MultiFidelityBayesianOptimisationState,
  type MultiFidelityObjective,
} from './multi-fidelity'
export { bayesianAlgorithms, bayesianFunctions } from './registry'
