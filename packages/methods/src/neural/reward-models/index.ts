/**
 * `aifn-methods/neural/reward-models`: reward models from pairwise preferences and reward over-optimisation, on small
 * synthetic problems.
 *
 * - Data: `syntheticPreferences`, response pairs labelled by a known gold reward under the Bradley–Terry model, with
 *   Gumbel noise at a temperature or without noise.
 * - Fitting: `fitBradleyTerry`, a linear reward $\wvec^\top\phivec(\xvec)$ fitted by `logisticRegression` without an
 *   intercept on the feature differences of each pair.
 * - Over-optimisation: `bestOfNCurve`, the exact expected proxy and gold rewards of best-of-$n$ sampling from a pool
 *   (`bestOfNWeights` gives each rank's probability), against `bestOfNKl`, the bound $\log n - (n - 1)/n$ on the KL
 *   divergence from the base policy.
 *
 * Everything is deterministic: the data from their seed, the curve exactly from the pool.
 */

export {
  bestOfNCurve,
  bestOfNKl,
  bestOfNWeights,
  fitBradleyTerry,
  syntheticPreferences,
  type BestOfNPoint,
  type BradleyTerryReward,
  type FitBradleyTerryOptions,
  type SyntheticPreferences,
  type SyntheticPreferencesOptions,
} from './preferences'
export { rewardModelFunctions } from './registry'
