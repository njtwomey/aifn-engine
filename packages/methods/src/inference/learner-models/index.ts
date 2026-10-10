/**
 * `aifn-methods/inference/learner-models`: learner models for equitable ability estimation (Twomey et al., 2022),
 * which tell a zero caused by the context from one caused by low ability.
 *
 * - The models: IRT-ZILM, the zero-inflated learner model, with its baselines two-parameter IRT and a linear
 *   knowledge-tracing machine. `zilmProbability` is IRT-ZILM's $\pr(Y = 1) = (1 - \pi)\, \sigma(a(\theta - b))$ and
 *   `structuralZeroPosterior` the posterior that a zero came from the context.
 * - Fitting: `fitLearnerModel` by penalised joint maximum likelihood (autodiff and L-BFGS) on the objective
 *   `learnerObjective`, and `predictLearner` for every student and item.
 * - Evaluation: `abilityEquity` (ability bias by group and the equity gap), `parameterRecovery` (Pearson and Spearman
 *   correlations with the truth), `responseScores` (held-out accuracy, $F_1$, log-loss and Brier score) and
 *   `structuralZeroAuroc`.
 * - Streamed experiments: `learnerModelRun` compares the models on one dataset, and `learnerEquitySweep` across
 *   datasets with a growing zero-inflation rate.
 * - `learnerModelFunctions` is the module's registry.
 *
 * Responses are a $P \times I$ matrix (students by items) of 0 and 1, with NaN for an item not attempted; masks and
 * per-response outputs are row-major over it. Conditions are a $P \times K$ matrix of 0 and 1, item features an
 * $I \times F$ matrix. Fits are deterministic; the experiments are deterministic in their seed.
 */

export {
  fitLearnerModel,
  learnerObjective,
  predictLearner,
  structuralZeroPosterior,
  zilmProbability,
  type LearnerModelFit,
  type LearnerModelKind,
  type LearnerModelOptions,
  type LearnerObjective,
  type LearnerPredictions,
} from './zilm'
export {
  abilityEquity,
  parameterRecovery,
  responseScores,
  structuralZeroAuroc,
  type AbilityEquity,
  type GroupAbilityError,
  type ParameterRecovery,
  type ResponseScores,
} from './evaluation'
export {
  learnerEquitySweep,
  learnerModelRun,
  type LearnerData,
  type LearnerModelResult,
  type LearnerModelRunResult,
  type LearnerRunOptions,
  type LearnerSweepPoint,
  type LearnerSweepResult,
} from './run'
export { learnerModelFunctions } from './registry'
