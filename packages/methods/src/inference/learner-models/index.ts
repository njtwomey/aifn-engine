/**
 * `aifn-methods/inference/learner-models`: learner models for equitable ability estimation (Twomey et al., 2022).
 * IRT-ZILM, the zero-inflated learner model, with its baselines, two-parameter IRT and a linear knowledge-tracing
 * machine, fitted by penalised joint maximum likelihood (`zilm.ts`); equity and recovery measures (`evaluation.ts`);
 * and streamed experiments: one comparison, and a sweep of the zero-inflation gap between groups (`run.ts`).
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
