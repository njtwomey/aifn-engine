/**
 * `aifn-compute/learning/estimators`: the estimator protocol (plan §5.1). Fitted models, capabilities, datasets and
 * predictive distributions are the contract types of `aifn-compute/foundation/contracts`; this module adds what runs.
 *
 * - Capabilities: `Fitted` (`forward`), `Decides`, `Predicts`, `Expects`, `Scores`, `Transforms`, `Samples`,
 *   `Trained`, named by `Capability`; guards `hasPredictive` and the like; `capabilities(m)`.
 * - Mixins: `withDecision(model, rule)` (argmax, mode, threshold, cost matrix), `withExpectation`, `withSampling`,
 *   `readout(model, complete)`.
 * - Data: `Dataset` (`kind: 'dataset'`) and `dataset(x, y?)`, `Supervised`, `rowCount`, `takeRows`, `takeData`;
 *   `Estimator`, `FitOptions`; input checks `matrixShape`, `targetValues`.
 * - Predictives: contract `Distribution`s; `gaussianPredictive`, `bernoulliPredictive`, `categoricalPredictive`,
 *   `classProbabilities`, `expectation` (Gauss–Hermite on normal scores), `asTensor`.
 * - Registering: `defineModel(spec, factory)` attaches a `ModelInfo` (task, capabilities, `hyper` space) to an
 *   estimator factory; `isModelEntry`.
 * - Evaluation: `evaluate(model, data, metrics)` over registered metrics, each read by its `info.capability`.
 */

export {
  capabilities,
  hasDecide,
  hasExpect,
  hasForward,
  hasPredictive,
  hasSample,
  hasScore,
  hasTraining,
  hasTransform,
  type Capability,
  type DecisionOf,
  type Decides,
  type Expects,
  type Fitted,
  type HeadOf,
  type InputOf,
  type Model,
  type ModelInfo,
  type PredictiveOf,
  type Predicts,
  type Samples,
  type Scores,
  type Task,
  type Trained,
  type Transforms,
} from './capabilities'
export {
  dataset,
  rowCount,
  takeData,
  takeRows,
  type Column,
  type DataOf,
  type Dataset,
  type DatasetExtras,
  type DatasetMeta,
  type Estimator,
  type Features,
  type FitOptions,
  type ModelOf,
  type Supervised,
  type Table,
} from './data'
export {
  asTensor,
  bernoulliPredictive,
  categoricalPredictive,
  classProbabilities,
  expectation,
  gaussianPredictive,
  isClassDistribution,
  isUnivariate,
  type AnyUnivariate,
  type ClassDistribution,
  type Distribution,
} from './distribution'
export {
  readout,
  withDecision,
  withExpectation,
  withSampling,
  type Completers,
  type DecisionRule,
  type ReadoutOf,
} from './mixins'
export {
  evaluate,
  metricInput,
  outputFor,
  outputs,
  score,
  type CapabilityOf,
  type Outputs,
  type Requirement,
  type ServedMetric,
} from './evaluate'
export { matrixShape, targetValues } from './util'
export { defineModel, isModelEntry, type EstimatorFactory, type ModelEntry, type ModelSpec } from './define'
