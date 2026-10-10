/**
 * `aifn-methods/learning/reductions`: multiclass classifiers built from any binary classifier, as scikit-learn's
 * `sklearn.multiclass`.
 *
 * - Reductions: `oneVersusRest` ($K$ models, each class against the rest), `oneVersusOne` ($K(K - 1)/2$ pairwise
 *   models that vote), `outputCode` (one model per column of a code matrix, decoded by Hamming or loss distance) and
 *   `nestedDichotomies` (a tree of binary splits whose probabilities multiply, so it always has class probabilities).
 * - Code matrices for `outputCode`: `oneVersusRestCode`, `oneVersusOneCode`, `exhaustiveCode` (every split, up to 16
 *   classes) and `randomCode` (dense or sparse); `codeDistance` gives how far apart two codewords are at least, so how
 *   many binary errors a code corrects.
 * - Class trees for `nestedDichotomies`: `dichotomyTree` (balanced or chain) and `randomDichotomyTree`.
 * - `softmaxScores` turns $m \times K$ logit scores into class probabilities.
 *
 * The base is a `BinaryEstimator` fitted on labels 0 and 1 whose model gives one real margin per row (`score` or
 * `forward`). Labels are the integers $0, \dots, K - 1$, and every model's `decide` is the argmax of its `score`.
 * `reductionsFunctions` lists the module's functions by key.
 */

export {
  codeDistance,
  dichotomyTree,
  randomDichotomyTree,
  exhaustiveCode,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  oneVersusRestCode,
  outputCode,
  randomCode,
  softmaxScores,
  type BinaryEstimator,
  type BinaryModel,
  type Dichotomy,
  type NestedDichotomyModel,
  type ReductionModel,
} from './multiclass'
export { reductionsFunctions } from './registry'
