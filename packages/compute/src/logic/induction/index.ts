/**
 * `aifn-compute/logic/induction`: inductive logic programming and concept learning, learning clauses and conjunctions
 * from examples and comparing them by generality.
 *
 * - FOIL (Quinlan, 1990): `foilProblem` builds a problem from ground background facts and examples (closed-world
 *   negatives when none are given); `foilSteps` is the step-through algorithm (sequential covering, each step one
 *   literal chosen by FOIL gain or one clause learned) and `foil` runs it to the end. Recursion goes through the
 *   positive examples.
 * - FOIL's parts: `foilRefinements` (the refinement operator, searched with `aifn-compute/optim/search`), `foilGain`,
 *   `foilCoverage` and `coveredExamples`, `foilSimplify`; `foilClause`, `foilClauseText`, `foilLiteralTerm` and
 *   `foilVariableName` turn literals into terms and text.
 * - Plotkin's generality: `thetaSubsumption` and `thetaSubsumes` ($\theta$-subsumption of clauses), with
 *   `thetaSubsumptionTrace` recording the backtracking search state by state (`renameApart`, `targetLiterals`,
 *   `predicateOf`); `termLgg` and `clauseLgg` (least general generalisation), traced by `termLggTraced` (one
 *   comparison at a time, with a shared `lggTable`) and `clauseLggTrace` (one literal pair at a time);
 *   `relativeLgg`, Golem's lgg relative to background facts (with `headLinked`); and `reduceClause`.
 * - Inverse entailment and meta-interpretive learning: `bottomClause` saturates an example under mode declarations
 *   (`parseMode`) and `bottomClauseSearch` searches the clauses below it (Progol and Aleph); `metagol` learns dyadic
 *   Datalog programs from metarules (`METARULES`), inventing predicates.
 * - Learning from failures (Popper): `learnFromFailures` generates, tests and constrains programs of a hypothesis
 *   space built by `lffClauses`, `lffSubsumption` and `lffPrograms`.
 * - Relational data: `mineClosedRules` mines AMIE's closed rules from a knowledge graph and `predictLinks` predicts
 *   facts from them; `attributeFeatures`, `existentialFeatures`, `aggregateFeatures`, `wordify` and `featureTable`
 *   propositionalise a one-to-many database for an attribute-value learner.
 * - Concept learning over attribute conjunctions (Mitchell): `covers`, the order `moreGeneralOrEqual` and
 *   `strictlyMoreGeneral`, the refinement operators `specialisations` and `generalisations`, `generaliseToCover` and
 *   `specialiseToExclude`, the language (`allConjunctions`, `allInstances`, `hasseEdges`, `conjunctionLevel`) and its
 *   printing; `candidateElimination` is the step-through version-space algorithm (with `findS`,
 *   `versionSpaceMembers`, `consistentSet` and `versionSpaceVote`), and `topDownSearch` and `bottomUpSearch` search
 *   the order in the two directions.
 * - `inductionAlgorithms` and `inductionFunctions`: the module's registry entries.
 *
 * Background relations are extensional (sets of ground facts) except in learning from failures, which takes Prolog
 * text, and a FOIL problem's constants are interned. Searches whose result a figure steps through return every state
 * (`*Trace`, `metagol`'s events, `learnFromFailures`' steps); they are plain functions of plain data. Failures to find
 * a hypothesis are reported in the result, not thrown.
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
export {
  clauseLgg,
  clauseLggTrace,
  headLinked,
  lggTable,
  predicateOf,
  reduceClause,
  relativeLgg,
  renameApart,
  targetLiterals,
  termLgg,
  termLggTraced,
  thetaSubsumes,
  thetaSubsumption,
  thetaSubsumptionTrace,
  type HornClause,
  type LggAction,
  type LggPair,
  type LggStep,
  type LggTable,
  type LiteralPairStep,
  type RelativeLgg,
  type SubsumptionEvent,
  type SubsumptionStep,
  type SubsumptionTraceOptions,
} from './generalise'
export {
  allConjunctions,
  allInstances,
  ANY,
  conjunctionCoverage,
  conjunctionLevel,
  conjunctionText,
  consistent,
  covers,
  fromInstance,
  generalisations,
  generaliseToCover,
  hasseEdges,
  instanceText,
  moreGeneralOrEqual,
  mostGeneral,
  sameConjunction,
  specialisations,
  specialiseToExclude,
  strictlyMoreGeneral,
  type AttributeInstance,
  type AttributeSpace,
  type Conjunction,
  type LabelledInstance,
} from './conjunctions'
export {
  candidateElimination,
  consistentSet,
  findS,
  versionSpaceMembers,
  versionSpaceVote,
  type CandidateEliminationEvent,
  type CandidateEliminationState,
} from './version-space'
export {
  bottomUpSearch,
  topDownSearch,
  type ScoredRefinement,
  type SearchStep,
  type TopDownOptions,
} from './generality-search'
export {
  bottomClause,
  bottomClauseSearch,
  parseMode,
  subsetText,
  type BottomClause,
  type BottomLiteral,
  type BottomSearchOptions,
  type BottomSearchStep,
  type ModeArgument,
  type ModeDeclaration,
  type SearchedClause,
} from './progol'
export {
  metagol,
  METARULES,
  type Argument,
  type DyadicAtom,
  type Metarule,
  type MilEvent,
  type MilOptions,
  type MilProblem,
  type MilResult,
  type PredicateTerm,
} from './metagol'
export {
  learnFromFailures,
  lffClauses,
  lffPrograms,
  lffSubsumption,
  type LffBias,
  type LffClause,
  type LffConstraint,
  type LffLiteral,
  type LffOptions,
  type LffPredicate,
  type LffProblem,
  type LffProgram,
  type LffResult,
  type LffStep,
} from './failures'
export {
  mineClosedRules,
  predictLinks,
  type LinkPrediction,
  type MinedRule,
  type MiningOptions,
  type RuleAtom,
  type Triple,
} from './rule-mining'
export {
  aggregateFeatures,
  attributeFeatures,
  existentialFeatures,
  featureTable,
  wordify,
  type AttributeTest,
  type ChildTable,
  type ExistentialOptions,
  type PropositionalFeature,
  type RelationalDatabase,
  type RelationalIndividual,
  type WordifyOptions,
} from './propositionalise'
export { inductionAlgorithms, inductionFunctions } from './registry'
