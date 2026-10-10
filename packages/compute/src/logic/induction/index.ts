/**
 * `aifn-compute/logic/induction`: inductive logic programming, learning clauses from examples and comparing clauses by
 * generality.
 *
 * - FOIL (Quinlan, 1990): `foilProblem` builds a problem from ground background facts and examples (closed-world
 *   negatives when none are given); `foilSteps` is the step-through algorithm (sequential covering, each step one
 *   literal chosen by FOIL gain or one clause learned) and `foil` runs it to the end. Recursion goes through the
 *   positive examples.
 * - FOIL's parts: `foilRefinements` (the refinement operator, searched with `aifn-compute/optim/search`), `foilGain`,
 *   `foilCoverage` and `coveredExamples`, `foilSimplify`; `foilClause`, `foilClauseText`, `foilLiteralTerm` and
 *   `foilVariableName` turn literals into terms and text.
 * - Plotkin's generality: `thetaSubsumption` and `thetaSubsumes` ($\theta$-subsumption of clauses), `termLgg` and
 *   `clauseLgg` (least general generalisation), and `reduceClause` (removing redundant literals).
 * - `inductionAlgorithms` and `inductionFunctions`: the module's registry entries.
 *
 * Background relations are extensional (sets of ground facts), and a FOIL problem's constants are interned: its
 * relations and examples are tuples of constant ids.
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
