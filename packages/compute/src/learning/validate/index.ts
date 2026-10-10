/**
 * `aifn-compute/learning/validate`: splitters, cross-validation, hyperparameter search and nested cross-validation
 * (plan §5.5), as sklearn.model_selection.
 *
 * - Splitters (pure functions of the data and a stream): `kFold`, `stratifiedKFold`, `groupKFold`, `leaveOneOut`,
 *   `repeated`, `shuffleSplit`, `expandingWindow` (scikit-learn's `TimeSeriesSplit`) and `rollingOrigin` (a fixed
 *   window for forecasting); `assignment` turns splits into a $k \times n$ matrix of `TEST` (1), `TRAIN` (0) and
 *   `UNUSED` ($-1$).
 * - `crossValidate(estimator, data, splitter, metrics)`: every fold's fitted model, predictions, metrics (registered
 *   metrics, keyed by `info.key`) and training trace, with scores over folds and out-of-fold predictions.
 * - `gridSearch` and `randomSearch` over a `Space` of `aifn-compute/foundation/space` (grid values, or uniform draws
 *   with log scales), with the full results table; `nested(outer, inner, search, data)` for nested cross-validation
 *   and the optimism of the unnested estimate.
 *
 * Randomness is explicit: splitters, searches and the fits they run draw only from the stream they are given, each
 * from its own child of it; a shuffled splitter or a random search throws `DomainError` without one. Splits are sorted
 * int32 row indices. A splitter that cannot split the rows it is given throws `DomainError`.
 */

export {
  assignment,
  expandingWindow,
  groupKFold,
  kFold,
  leaveOneOut,
  repeated,
  rollingOrigin,
  shuffleSplit,
  stratifiedKFold,
  TEST,
  TRAIN,
  UNUSED,
  type Split,
  type SplitInput,
  type Splitter,
} from './splitters'
export {
  crossValidate,
  type CrossValidateOptions,
  type CrossValidation,
  type CrossValidationData,
  type Fittable,
  type Fold,
} from './cross'
export {
  gridSearch,
  nested,
  randomSearch,
  type NestedCrossValidation,
  type Search,
  type SearchModel,
  type SearchResult,
  type SearchRow,
} from './search'
export { validateFunctions } from './registry'
