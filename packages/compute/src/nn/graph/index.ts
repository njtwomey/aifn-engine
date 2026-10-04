/**
 * `aifn-compute/nn/graph`: graph neural network layers on `aifn-compute/graph/propagation`: graph convolution (Kipf and Welling, with
 * symmetric, random-walk or no normalisation), graph attention (GAT and GATv2, multi-head, with the attention weights),
 * GraphSAGE (mean, sum, max and max-pooling aggregators, neighbourhood sampling) and a generic message-passing layer
 * (Gilmer et al.). Each has a functional form taking the graph and a `Layer` closing over one.
 */

export {
  gcnCoefficients,
  GraphAttention,
  graphAttention,
  GraphConv,
  graphConv,
  MessagePassing,
  messagePassing,
  sageConv,
  SageConv,
  sampleNeighbours,
  type GcnNormalisation,
  type GraphAttentionLayer,
  type GraphAttentionOptions,
  type GraphAttentionParams,
  type GraphAttentionResult,
  type GraphConvOptions,
  type MessagePassingOptions,
  type MessagePassingParams,
  type SageAggregator,
  type SageOptions,
  type SageParams,
} from './layers'
export { graphLayerFunctions } from './registry'
