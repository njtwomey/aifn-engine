/**
 * `aifn-compute/nn/graph`: graph neural network layers on the message passing of `aifn-compute/graph/propagation`.
 *
 * - Graph convolution (Kipf and Welling): `graphConv` and the layer `GraphConv`, with `symmetric`, `random-walk` (the
 *   mean) or no normalisation; `gcnCoefficients` gives the weight $\hat{A}_{vu}$ of every message edge.
 * - Graph attention: `graphAttention` and the layer `GraphAttention`, GAT or GATv2, multi-head (concatenated or
 *   averaged), returning the attention weights on every edge.
 * - GraphSAGE: `sageConv` and the layer `SageConv`, with mean, sum, max and max-pooling aggregators, and
 *   `sampleNeighbours` to bound each node's neighbourhood.
 * - Generic message passing (Gilmer et al.): `messagePassing` and the layer `MessagePassing`, for any message and
 *   update functions.
 *
 * Each functional form takes the graph, the $V \times F$ node features and the parameters; each layer closes over one
 * graph (transductive use). Undirected edges carry messages both ways. Every layer is differentiable in its parameters
 * and features, and throws `ShapeError` for features without one row per node.
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
