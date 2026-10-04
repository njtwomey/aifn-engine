/**
 * `aifn-compute/probability/information`: information measures of distributions: entropies (discrete and differential), cross
 * entropy, Kullback–Leibler and Jensen–Shannon divergences, f-divergences, Hellinger distance, mutual information
 * (discrete, Gaussian, KSG) and total correlation.
 */

export {
  asValue,
  conditionalEntropy,
  crossEntropy,
  differentialEntropy,
  entropy,
  fDivergence,
  flatProbabilities,
  fGenerators,
  hellingerDistance,
  jensenShannonDistance,
  jensenShannonDivergence,
  jointEntropy,
  klDivergence,
  mutualInformation,
  pointwiseMutualInformation,
  totalVariation,
  type BaseOption,
  type FGenerator,
  type Probabilities,
} from './measures'
export { gaussianMutualInformation, ksgMutualInformation } from './continuous'
export { informationFunctions } from './registry'
