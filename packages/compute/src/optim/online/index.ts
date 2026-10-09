/**
 * `aifn-compute/optim/online`: online learning, where a learner acts each round before the round's loss is revealed,
 * and is judged by its regret against the best fixed choice in hindsight.
 *
 * - Prediction with expert advice, over $N$ experts: `hedge` (exponential weights, with a constant, tuned or anytime
 *   rate), `fixedShare` (Hedge that tracks an expert that changes) and `weightedMajority` (binary predictions,
 *   deterministic or randomised). Losses come as a matrix, or from a function of the round (an adaptive adversary).
 * - Online convex optimisation over a ball, a box, the simplex or $\reals^d$ (each method says which it supports):
 *   `onlineGradientDescent` (with `ogdStepSize`), `followTheRegularisedLeader` (with L1 and L2 terms, and
 *   FTRL-Proximal's per-coordinate rates), `onlineNewtonStep` (for exp-concave losses) and `onlineAdagrad` (for sparse
 *   gradients); `projectOnto` is the Euclidean projection onto those sets.
 * - Regret: `regretTrace` over time against a comparator or the best expert so far, `bestSwitchingLoss` (the
 *   comparator of tracking), and the bounds `hedgeTunedRate`, `hedgeRegretBound` and `ogdRegretBound`.
 * - `onlineToBatch` averages the iterates of an online learner into a batch estimate.
 * - `onlineAlgorithms` and `onlineFunctions` register the learners and functions for generic views and workers.
 *
 * Every learner is an `Algorithm` started with `undefined`, one step per round, done after the horizon; its state
 * carries the cumulative losses and the regret. Invalid options throw `DomainError`; a non-finite point of an online
 * convex optimiser sets `diverged`.
 */

export {
  fixedShare,
  hedge,
  weightedMajority,
  type ExpertLosses,
  type ExpertsState,
  type FixedShareOptions,
  type HedgeOptions,
  type HedgeRate,
  type WeightedMajorityState,
} from './experts'
export {
  followTheRegularisedLeader,
  ogdStepSize,
  onlineAdagrad,
  onlineGradientDescent,
  onlineNewtonStep,
  onlineToBatch,
  projectOnto,
  type FtrlOptions,
  type OnlineDomain,
  type OnlineLoss,
  type OnlineOptions,
  type OnlineState,
} from './convex'
export {
  bestSwitchingLoss,
  hedgeRegretBound,
  hedgeTunedRate,
  ogdRegretBound,
  regretTrace,
  type RegretTrace,
} from './regret'
export { onlineAlgorithms, onlineFunctions } from './registry'
