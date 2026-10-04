/**
 * `aifn-compute/nn/experts`: mixtures of experts. Routing from a router's logits (dense softmax, top-k, noisy top-k, Switch
 * top-1, expert choice) with a capacity and token dropping; the mixture-of-experts layer over any expert layers; the
 * load-balancing, importance and router z auxiliary losses; and routing statistics (load per expert, router entropy).
 */

export {
  denseRouting,
  expertCapacity,
  GATE_KINDS,
  route,
  type GateKind,
  type Routing,
  type RoutingOptions,
} from './routing'
export { importanceLoss, loadBalancingLoss, routerZLoss, routingStatistics, type RoutingStatistics } from './losses'
export {
  MixtureOfExperts,
  type MixtureOfExpertsLayer,
  type MixtureOfExpertsOptions,
  type MixtureOfExpertsParams,
  type MixtureOfExpertsResult,
} from './layer'
export { expertFunctions } from './registry'
