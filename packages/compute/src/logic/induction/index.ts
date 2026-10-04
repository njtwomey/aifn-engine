/**
 * `aifn-compute/logic/induction`: inductive logic programming. FOIL (`foilSteps`, `foil`): sequential covering with FOIL gain
 * over extensional background relations, recursion through the positive examples, its refinement operator searched
 * with `aifn-compute/optim/search`; Plotkin's generality: θ-subsumption, the least general generalisation of terms and clauses,
 * and clause reduction.
 */

export {
  coveredExamples,
  foil,
  foilClause,
  foilClauseText,
  foilCoverage,
  foilGain,
  foilLiteralTerm,
  foilProblem,
  foilRefinements,
  foilSimplify,
  foilSteps,
  foilVariableName,
  type FoilCandidate,
  type FoilEvent,
  type FoilLearned,
  type FoilLiteral,
  type FoilNode,
  type FoilOptions,
  type FoilProblem,
  type FoilRelation,
  type FoilResult,
  type FoilState,
} from './foil'
export { clauseLgg, reduceClause, termLgg, thetaSubsumes, thetaSubsumption, type HornClause } from './generalise'
export { inductionAlgorithms, inductionFunctions } from './registry'
