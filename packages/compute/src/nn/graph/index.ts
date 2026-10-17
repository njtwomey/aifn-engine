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
 * - More expressive aggregation: GIN (`ginConv`, the layer `GinConv`) and principal neighbourhood aggregation
 *   (`pnaAggregate`, with its degree normaliser `pnaDelta`).
 * - Spectral filters: `chebyshevTerms`, `chebyshevFilter` and `chebConv` on a rescaled Laplacian, and the layer
 *   `ChebConv`.
 * - Relations and hypergraphs: R-GCN (`rgcnTerms`, `rgcnConv`, `basisWeights`), the two-stage hypergraph mean
 *   `hypergraphMeanPass` and the HGNN convolution `hgnnConv`.
 * - Readout and pooling: `batchGraphs` (block-diagonal batches), `globalPool` (sum, mean or max per graph),
 *   `attentionPool`, `diffPool` and `topKPool`.
 * - Depth: the dense GCN matrix `gcnMatrix` and its spectrum `adjacencySpectrum`, `dirichletEnergy` (oversmoothing),
 *   `pairNorm`, `dropEdge`, and `propagationSensitivity` (oversquashing: the powers of $\hat\Amat$ at one node).
 * - Graph transformers: `spatialBias` from hop distances and one head of `graphormerAttention`.
 * - Scale: `computationGraph` and `sampledComputationGraph`, the nodes a minibatch needs, in full or sampled.
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
export {
  GinConv,
  ginConv,
  pnaAggregate,
  pnaDelta,
  type GinLayerOptions,
  type GinOptions,
  type GinParams,
  type PnaAggregator,
  type PnaOptions,
  type PnaScaler,
} from './expressive'
export { ChebConv, chebConv, chebyshevFilter, chebyshevTerms, type ChebConvParams } from './spectral'
export {
  attentionPool,
  batchGraphs,
  diffPool,
  globalPool,
  topKPool,
  type DiffPooled,
  type GlobalPooling,
  type GraphBatch,
  type TopKPooled,
} from './pooling'
export { adjacencySpectrum, dirichletEnergy, dropEdge, gcnMatrix, pairNorm, propagationSensitivity } from './smoothing'
export { graphormerAttention, spatialBias, type GraphormerWeights } from './transformer'
export {
  computationGraph,
  sampledComputationGraph,
  type ComputationGraph,
  type SampledComputationGraph,
} from './sampling'
export { hgnnConv, hypergraphMeanPass } from './hypergraph'
export { basisWeights, rgcnConv, rgcnTerms, type RgcnNormalisation, type RgcnRelation } from './relational'
export { graphLayerFunctions } from './registry'
