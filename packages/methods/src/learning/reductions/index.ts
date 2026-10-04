/**
 * `aifn-methods/learning/reductions`: multiclass by binary reductions: one-versus-rest, one-versus-one, output codes
 * and nested dichotomies.
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
