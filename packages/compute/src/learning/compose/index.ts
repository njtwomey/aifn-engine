/**
 * `aifn-compute/learning/compose`: pipelines, column-wise transforms and target transforms (plan §5.4), as
 * sklearn.pipeline and sklearn.compose.
 *
 * - `pipeline(...steps)`: transforms chained into a final estimator; the fitted pipeline has exactly the final
 *   step's capabilities (typed by `Lift`) and exposes each step's fitted state in `steps`.
 * - `columns({ age: standardScaler(), city: oneHotEncoder() })`: per-column transforms of a table into one matrix,
 *   with `'passthrough'`, `'drop'` and a remainder.
 * - `transformTarget(regressor, map)`: fit on $z = g(y)$; point predictions are inverted, predictive distributions
 *   pushed forward through $g^{-1}$ (`pushForward`: log-normal for a Gaussian under `logTarget`
 *   (`logNormalPredictive`), `transformedPredictive` otherwise). Fixed maps: `logTarget`, `log1pTarget`,
 *   `affineTarget`; maps fitted on the training targets: `standardTarget`, `powerTarget` (Box–Cox, Yeo–Johnson).
 *
 * Every composition is fitted as one estimator, so its preprocessing learns only from the rows it is fitted on, and
 * cross-validating it keeps each test fold out of the scalers and encoders. Fitted compositions are plain objects
 * (`kind: 'model'`, with a `composition` tag) exposing their parts' fitted state; fitting never changes the steps.
 */

export {
  pipeline,
  type AnyEstimator,
  type FittedOf,
  type FittedSteps,
  type InputOfStep,
  type Last,
  type Lift,
  type PipelineModel,
  type TransformStep,
} from './pipeline'
export { columns, type ColumnSpec, type ColumnsModel } from './columns'
export {
  affineTarget,
  log1pTarget,
  logNormalPredictive,
  logTarget,
  powerTarget,
  pushForward,
  standardTarget,
  transformedPredictive,
  transformTarget,
  type LogNormalPredictive,
  type TargetMap,
  type TargetMapEstimator,
  type TargetModel,
  type TransformedPredictive,
} from './target'
export { composeFunctions } from './registry'
