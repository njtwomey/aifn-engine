/**
 * `aifn-compute/graph/rewiring`: changes to a graph's edges that widen its bottlenecks against oversquashing.
 *
 * - `withVirtualNode`: one new node joined to every node, so every pair is at most two hops apart.
 * - `fosrRewire`: first-order spectral rewiring, adding one by one the non-edges that most increase the spectral gap.
 *
 * Both return new plain `Graph`s and leave the input unchanged. Measure the result with `cheegerConstant`
 * (`aifn-compute/graph/measures`) or the spectral gap of its random-walk matrix.
 */

export { fosrRewire, withVirtualNode, type FosrResult } from './rewiring'
export { rewiringFunctions } from './registry'
