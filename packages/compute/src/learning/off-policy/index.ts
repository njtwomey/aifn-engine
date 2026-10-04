/**
 * `aifn-compute/learning/off-policy`: off-policy (counterfactual) evaluation from logged bandit feedback: IPS, clipped and
 * self-normalised IPS, the direct method, doubly robust and switch-DR; slate estimators (pseudo-inverse and slate IPS);
 * propensity models; importance weights (their effective sample size is `importanceEffectiveSampleSize` in
 * `aifn-compute/probability/stats`). Logged-bandit generators are in `aifn-methods/data/synthetic`.
 */

export {
  clippedIps,
  directMethod,
  doublyRobust,
  importanceWeights,
  ips,
  snips,
  switchDoublyRobust,
  type BanditLog,
  type EstimateOptions,
  type OffPolicyEstimate,
} from './estimators'
export { slateIps, slatePseudoInverse, type SlateLog, type SlateLogging } from './slate'
export { empiricalPropensities, estimatePropensities, type PropensityModel } from './propensity'
export { offPolicyFunctions } from './registry'
