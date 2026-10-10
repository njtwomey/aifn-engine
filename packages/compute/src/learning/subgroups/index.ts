/**
 * `aifn-compute/learning/subgroups`: subgroup discovery and exceptional model mining over a table, finding the
 * conjunctive descriptions whose rows are most unusual in a target.
 *
 * - Covers as bitsets: `bitset`, `bitsetFull`, `bitsetHas`, `bitsetAnd`, `bitsetAndNot`, `bitsetNot`, `bitsetCount`,
 *   `bitsetAndCount`, `bitsetIndices` and `bitsetJaccard`.
 * - The description language: `selectorLanguage` (nominal `=` and `≠`, numeric `≥` and `≤` at cut points by
 *   equal-frequency, equal-width or on-the-fly discretisation, with memoised covers and a canonical refinement
 *   operator), `cutPoints`, `compatibleSelector`, and `selectorKey`, `descriptionKey` and `formatDescription` to name
 *   selectors and descriptions.
 * - Quality measures with optimistic estimates: for a binary target `standardQuality` (the Klösgen family
 *   $(n/N)^a (p - p_0)$), `wraccQuality`, `binomialQuality`, `liftQuality`, `coverageQuality` and `chiSquareQuality`;
 *   for a numeric one `meanShiftQuality`.
 * - Search: `subgroupDiscovery` for the top $k$ (beam, best-first, depth-first or breadth-first by
 *   `aifn-compute/optim/search`, with branch and bound and a redundancy filter), `subgroupDiscoverySteps` to step
 *   through it, `subgroupSpace` and `subgroupRedundancy` for the search itself, and `subgroupOf` for a description
 *   chosen by hand; `sdMap`, exhaustive by FP-growth for count-based measures of a binary target.
 * - Exceptional model mining, where the target is a model fitted inside and outside: `correlationModel`,
 *   `regressionModel`, `logisticModel` and `associationModel`, each a quality measure without an optimistic estimate.
 * - The registry entries of the module: `subgroupAlgorithms` and `subgroupFunctions`.
 *
 * Everything is deterministic. An empty cover has quality $-\infty$; invalid options and inconsistent columns throw
 * `DomainError`.
 */

export {
  bitset,
  bitsetAnd,
  bitsetAndCount,
  bitsetAndNot,
  bitsetCount,
  bitsetFull,
  bitsetHas,
  bitsetIndices,
  bitsetJaccard,
  bitsetNot,
  type Bitset,
} from './cover'
export {
  compatibleSelector,
  cutPoints,
  descriptionKey,
  formatDescription,
  selectorKey,
  selectorLanguage,
  type Description,
  type Discretisation,
  type LanguageAttribute,
  type LanguageOptions,
  type Selector,
  type SelectorLanguage,
  type SelectorOp,
} from './language'
export {
  binomialQuality,
  chiSquareQuality,
  coverageQuality,
  liftQuality,
  meanShiftQuality,
  standardQuality,
  wraccQuality,
  type CountMeasure,
  type Direction,
  type NumericMeasure,
  type QualityMeasure,
} from './quality'
export {
  subgroupDiscovery,
  subgroupDiscoverySteps,
  subgroupOf,
  subgroupRedundancy,
  subgroupSpace,
  type Redundancy,
  type Subgroup,
  type SubgroupOptions,
} from './subgroups'
export { sdMap, type SdMapOptions } from './sdmap'
export {
  associationModel,
  correlationModel,
  logisticModel,
  regressionModel,
  type AssociationFit,
  type BivariateFit,
  type LogisticFit,
  type ModelFits,
  type ModelMeasure,
} from './emm'
export { subgroupAlgorithms, subgroupFunctions } from './registry'
