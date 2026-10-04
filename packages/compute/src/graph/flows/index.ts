/**
 * `aifn-compute/graph/flows`: network flows: maximum flow by Edmonds–Karp and minimum-cost flow, with their steps.
 */

export {
  edmondsKarpSteps,
  maxFlow,
  minCostFlow,
  minCostFlowSteps,
  type EdmondsKarpState,
  type FlowArc,
  type FlowNetwork,
  type MaxFlowOptions,
  type MaxFlowResult,
  type MinCostFlowState,
} from './flows'
export { flowsAlgorithms, flowsFunctions } from './registry'
