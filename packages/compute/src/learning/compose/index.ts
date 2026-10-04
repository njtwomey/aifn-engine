/**
 * `aifn-compute/learning/compose`: pipelines, column-wise transforms and target transforms (plan §5.4).
 *
 * - `pipeline(...steps)`: transforms chained into a final estimator; the fitted pipeline has exactly the final
 *   step's capabilities (typed by `Lift`) and exposes each step's fitted state in `steps`.
 * - `columns({ age: standardScaler(), city: oneHotEncoder() })`: per-column transforms of a table into one matrix.
 * - `transformTarget(regressor, map)`: fit on z = g(y); point predictions are inverted, predictive distributions pushed
 *   forward through g⁻¹ (`pushForward`: log-normal for a Gaussian under `logTarget`, `transformedPredictive`
 *   otherwise). Maps: `logTarget`, `log1pTarget`, `affineTarget`, `standardTarget`, `powerTarget`.
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
