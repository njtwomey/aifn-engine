/**
 * `aifn-methods/theory/algorithm-comparison`: how often the tests that compare two learning algorithms reject when the
 * two are equally good, by simulation after Dietterich (1998).
 *
 * - The study: `typeIErrorStudy` runs McNemar's test, the paired $t$-test on one and on $r$ repeats of 10-fold
 *   cross-validation, the corrected resampled $t$-test and the $5 \times 2$cv $t$ and combined $F$ tests on many
 *   simulated data sets, and returns each test's $p$-values and rejection rate, and the naive and corrected rates by
 *   number of repeats (`TypeIErrorStudy`); `typeIErrorStudySteps` yields the same summary after each data set, for a
 *   worker to stream.
 * - One data set: `typeIErrorReplicate`, every test's $p$-value and the learners' true accuracy difference.
 * - `TYPE_I_TESTS` lists the tests in display order (`TypeITest`).
 *
 * The learners are depth-4 decision trees on a circular class boundary, one on rotated inputs, so they are equally
 * good by symmetry; a `handicap` makes one worse and the rates become power. Results are typed arrays and plain
 * objects, so the study runs in a worker. `algorithmComparisonFunctions` registers the functions.
 */

export { algorithmComparisonFunctions } from './registry'
export {
  TYPE_I_TESTS,
  typeIErrorReplicate,
  typeIErrorStudy,
  typeIErrorStudySteps,
  type TypeIErrorOptions,
  type TypeIErrorReplicate,
  type TypeIErrorStudy,
  type TypeIErrorStudyOptions,
  type TypeITest,
} from './type-i-error'
