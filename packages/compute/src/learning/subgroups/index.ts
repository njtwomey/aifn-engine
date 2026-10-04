/**
 * `aifn-compute/learning/subgroups`: subgroup discovery and exceptional model mining over a table. A description language of
 * nominal and numeric selectors (with equal-frequency, equal-width or on-the-fly discretisation) and bitset covers;
 * quality measures (WRAcc and the Klösgen family, binomial test, lift, coverage, χ², the numeric mean shift) with their
 * optimistic estimates; search by `aifn-compute/optim/search` (beam, best-first, exhaustive, branch and bound) with redundancy
 * filtering; SD-Map; and exceptional-model classes (correlation, regression slope, logistic classifier, association).
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
