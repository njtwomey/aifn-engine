/**
 * `aifn-compute/graph/flows`: network flows, maximum flow and minimum-cost flow, with their steps.
 *
 * - Maximum flow and minimum cut: `maxFlow` (edge weights are capacities; returns the flow per edge, the cut and its
 *   capacity) and `edmondsKarpSteps`, its traceable algorithm (one shortest augmenting path per step).
 * - Minimum-cost flow: `minCostFlow` on a `FlowNetwork` (arcs with capacity and unit cost, supplies summing to zero)
 *   and `minCostFlowSteps`, successive shortest paths with potentials, so negative costs are allowed.
 * - `flowsAlgorithms` and `flowsFunctions`: the registry entries of the algorithms and the functions.
 *
 * Invalid input throws (`ShapeError`, `DomainError`); an unreachable demand is reported as `status: 'infeasible'` and
 * an unbounded flow as `value` Infinity.
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
