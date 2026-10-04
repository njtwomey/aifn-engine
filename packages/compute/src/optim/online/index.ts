/**
 * `aifn-compute/optim/online`: online learning. Prediction with expert advice (`hedge`, `fixedShare`, `weightedMajority`) and
 * online convex optimisation (`onlineGradientDescent`, `followTheRegularisedLeader` with L1/L2 and FTRL-Proximal,
 * `onlineNewtonStep`, `onlineAdagrad`) as step-through algorithms whose states carry the regret; regret traces, the
 * best switching comparator, the regret bounds, and online-to-batch conversion.
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
