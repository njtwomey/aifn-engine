/**
 * `aifn-compute/learning/off-policy`: off-policy (counterfactual) evaluation of a target policy from logged bandit
 * feedback, after the estimators of the Open Bandit Pipeline (`obp.ope`).
 *
 * - Value estimators from a log of actions, rewards and propensities: `ips` (unbiased, high variance), `clippedIps`
 *   (weights capped at $M$), `snips` (self-normalised), `directMethod` (a reward model $\hat q$ alone),
 *   `doublyRobust` (the model corrected by IPS on its residuals) and `switchDoublyRobust` (DR where the weight is at
 *   most $\tau$, the model elsewhere); `importanceWeights` gives the weights
 *   $w_i = \pi(a_i \mid x_i)/\pi_0(a_i \mid x_i)$ themselves.
 * - Slates of $l$ slots from $m$ items: `slatePseudoInverse` (additive rewards, from the logging policy's pairwise
 *   marginals) and `slateIps` (the whole slate's probability), under a uniform or listed logging policy.
 * - Propensities when the log lacks them: `estimatePropensities` (multinomial logistic regression on the context) and
 *   `empiricalPropensities` (smoothed shares within discrete contexts).
 *
 * Policies and reward models are $n \times K$ matrices, one row per logged round. Every estimator returns an
 * `OffPolicyEstimate`: the value, a standard error and normal interval, the per-round terms and weights, and Kish's
 * effective sample size of the weights (`importanceEffectiveSampleSize` in `aifn-compute/probability/stats`). A
 * malformed log throws `DomainError`. Logged-bandit generators are in `aifn-methods/data/synthetic`.
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
