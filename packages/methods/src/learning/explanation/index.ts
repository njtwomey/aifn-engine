/**
 * `aifn-methods/learning/explanation`: studies that compare the explanation methods of `aifn-compute/learning/explain`
 * on planted truths, for the lab's workers.
 *
 * - Data attribution: `dataValuationStudy` explains one logistic regression by influence functions, self-influence,
 *   TracIn, KNN-Shapley and TMC data Shapley (`VALUATION_METHODS`), and ranks the training points by how suspect each
 *   method finds them, to compare against planted label noise. It is a generator, so a worker can stream the slow
 *   data-Shapley stage to a page.
 *
 * `explanationFunctions` lists the module's functions by key.
 */

export {
  dataValuationStudy,
  VALUATION_METHODS,
  type ValuationMethod,
  type ValuationOptions,
  type ValuationSnapshot,
} from './valuation'
export { explanationFunctions } from './registry'
