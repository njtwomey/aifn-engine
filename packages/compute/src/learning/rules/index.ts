/**
 * `aifn-compute/learning/rules`: rule learning, from the conditions a rule is built of to decision lists, rule sets
 * and rule ensembles, each learner with a trace a figure can step through.
 *
 * - Conditions and covers: `thresholdConditions` (at quantile edges) and `nominalConditions` (equality tests), all
 *   interval `Predicate`s of `aifn-compute/learning/explain`; `coverMask`, `andMask` and `ruleCounts` for the rows a
 *   rule covers; `ruleText` to print a rule; `mineAntecedents`, frequent conjunctions mined level-wise (Apriori);
 *   `applyClassRules` applies any decision list.
 * - Quality of a rule from its counts $(p, n)$ out of $(P, N)$: `ruleQuality` by name (`RULE_MEASURES` lists them,
 *   with TeX), and `laplace`, `mEstimate`, `ruleAccuracy`, `likelihoodRatio` (CN2's significance) and
 *   `ripperPruneValue`. Coverage, precision, WRAcc and FOIL gain are reached through `ruleQuality` only: their names
 *   belong to other modules (WRAcc is `wraccQuality` of `aifn-compute/learning/subgroups`, FOIL gain `foilGain` of
 *   `aifn-compute/logic/induction`).
 * - Separate and conquer: `sequentialCovering` (CN2's beam search under any measure, ordered or for one target class)
 *   with `applyCoveringRules`; `ripper` (IREP* and RIPPER$k$, with `ripperDescriptionLength`, `ruleBits`,
 *   `exceptionBits`, `subsetBits`) and `applyRipper`.
 * - From trees: `c45Rules` (C4.5rules: `generaliseRule` by `pessimisticError`, `ruleError`, subsets by description
 *   length), and RuleFit's terms, `ensembleRules`, `ruleFeatures` and `linearTerms` (the lasso path over them is
 *   `ruleFitPath` of `aifn-methods/learning/linear`).
 * - Rule lists over pre-mined antecedents: Bayesian rule lists (`ruleListLogPosterior`, `capturedCounts`,
 *   `ruleProbabilities` and the sampler `bayesianRuleListTrace`); associative classification
 *   (`classAssociationRules`, `cbaBuild` and `cbaPredict`, `cmarBuild` and `cmarPredict`, `ruleChiSquare`);
 *   certifiably optimal lists by branch and bound (`corels`).
 * - Boolean rules: `booleanRuleColumnGeneration`, a DNF of least Hamming loss within a complexity budget by column
 *   generation over `aggregateBinary` rows, with `dnfCover` and `dnfPredict`.
 * - `ruleFunctions`: the module's registry entries.
 *
 * Rows are arrays of numbers (`RuleRows`), classes integers $0, \dots, K - 1$, and covers `Uint8Array` masks of 0
 * and 1. Every learner is a plain function of plain data (and a seeded stream where it is random), so it can run in a
 * worker; the traces are arrays of plain states.
 */

export {
  andMask,
  applyClassRules,
  coverMask,
  mineAntecedents,
  nominalConditions,
  ruleCounts,
  ruleText,
  thresholdConditions,
  type Antecedent,
  type ClassRule,
  type RuleRows,
} from './conditions'
export {
  laplace,
  likelihoodRatio,
  mEstimate,
  ripperPruneValue,
  ruleAccuracy,
  ruleQuality,
  RULE_MEASURES,
  type RuleCounts,
  type RuleMeasure,
  type RuleMeasureInfo,
  type RuleMeasureOptions,
} from './quality'
export {
  applyCoveringRules,
  sequentialCovering,
  type CandidateRule,
  type CoveredRule,
  type CoveringOptions,
  type CoveringResult,
  type CoveringStep,
} from './covering'
export {
  applyRipper,
  exceptionBits,
  ripper,
  ripperDescriptionLength,
  ruleBits,
  subsetBits,
  type RipperOptions,
  type RipperResult,
  type RipperRule,
  type RipperStep,
} from './ripper'
export {
  c45Rules,
  generaliseRule,
  pessimisticError,
  ruleError,
  type Deletion,
  type RuleError,
  type TreeRulesResult,
  type TreeRulesStep,
} from './c45'
export {
  bayesianRuleListTrace,
  capturedCounts,
  ruleListLogPosterior,
  ruleProbabilities,
  type BrlPrior,
  type BrlScore,
  type BrlStep,
} from './brl'
export {
  cbaBuild,
  cbaPredict,
  classAssociationRules,
  cmarBuild,
  cmarPredict,
  ruleChiSquare,
  type CbaClassifier,
  type CbaStep,
  type ClassAssociationRule,
  type CmarClassifier,
} from './associative'
export {
  corels,
  type CorelsNode,
  type CorelsOptions,
  type CorelsResult,
  type CorelsStep,
  type NodeStatus,
} from './corels'
export {
  aggregateBinary,
  booleanRuleColumnGeneration,
  dnfCover,
  dnfPredict,
  type AggregatedBinary,
  type BooleanRuleOptions,
  type BooleanRuleResult,
  type ColumnGenerationStep,
  type DnfClause,
} from './boolean-rules'
export {
  ensembleRules,
  linearTerms,
  ruleFeatures,
  type EnsembleRule,
  type LinearTerms,
  type RuleColumns,
  type RuleTree,
} from './rulefit'
export { ruleFunctions } from './registry'
