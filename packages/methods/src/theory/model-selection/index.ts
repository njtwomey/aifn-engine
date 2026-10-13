/**
 * `aifn-methods/theory/model-selection`: selecting and assessing a tuned model over many simulated data sets, where
 * each estimate of its error can be set beside the true error.
 *
 * - `selectionStudy` tunes a penalty by grid-search cross-validation on each data set and records, per data set, the
 *   minimum cross-validation score, the training error, the nested cross-validation and test-set estimates (with
 *   standard errors) and the true errors of the minimiser, the one-standard-error choice and the oracle; then each
 *   estimate's mean, bias and spread against the true error (`SelectionStudyOptions`, `SelectionStudy`).
 * - Learners (`SelectionLearner`): a P-spline and ridge regression on polynomial features, each tuned by its penalty
 *   $\lambda$.
 *
 * The splitters, searches and the one-standard-error rule are those of `aifn-compute/learning/validate` and
 * `aifn-compute/learning/selection`. Data sets come from child streams, so a seed fixes the study. Options are plain
 * data and splitters, and results are plain, so the study runs in a compute worker. `modelSelectionFunctions`
 * registers it.
 */

export {
  selectionStudy,
  type EstimateSummary,
  type SelectionLearner,
  type SelectionLearnerName,
  type SelectionRepeats,
  type SelectionStudy,
  type SelectionStudyOptions,
} from './selection'
export { modelSelectionFunctions } from './registry'
