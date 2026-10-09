/**
 * `aifn-compute/nn/experts`: mixtures of experts, with routing, capacity and token dropping, and the auxiliary losses
 * that keep a router balanced.
 *
 * - Routing from a router's logits `[T, N]`: `route` with the gates of `GATE_KINDS` (dense softmax, top-k, noisy
 *   top-k, Switch top-1, expert choice), with an `expertCapacity` per expert past which assignments are dropped;
 *   `denseRouting` from given probabilities, for gates that are not one softmax.
 * - The layer: `MixtureOfExperts` over any expert layers, whose `forward` also returns the routing.
 * - Auxiliary losses: `loadBalancingLoss` (Switch), `importanceLoss` (the squared coefficient of variation of the
 *   experts' total weights) and `routerZLoss` (ST-MoE); `routingStatistics` reads load per expert, router entropy,
 *   idle experts and dropping as plain numbers.
 *
 * The combine weights are differentiable in the router's logits, while the choice of experts is read from their primal
 * values and is constant to every derivative transform. The layer runs every expert on every token and zeroes the
 * unchosen ones: the outputs and gradients of sparse dispatch at notebook sizes. `expertFunctions` is the registry
 * entry.
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
