/**
 * `aifn-methods/neural/graph`: graph neural network experiments: node classification, link prediction, depth,
 * graph generation and temporal memory.
 *
 * - The network: `gnnModel` builds two GCN, GAT or GraphSAGE layers from `aifn-compute/nn/graph`, tanh between them,
 *   one logit per class out (and GAT's attention weights on request).
 * - Training: `nodeLoss` (the cross-entropy on the labelled nodes) and `nodeClassificationTraining` (full-batch Adam
 *   as a traceable training loop).
 * - A streamed run: `nodeClassificationRun` yields the loss and accuracy curves on labelled and unlabelled nodes, and
 *   checkpoints of the logits, hidden features and attention, for a worker to stream to a page.
 *
 * - Link prediction with a graph autoencoder: `splitEdges` (held-out edges and as many non-edges) and
 *   `linkPredictionRun` (the GAE trained by Adam, streaming the held-out AUC).
 * - Depth without weights: `propagationStack` (plain, residual, initial-residual, PairNorm and DropEdge stacks, every
 *   layer returned) and `meanPropagation` (a fixed mean layer, on the full graph or on neighbour samples).
 * - Graph generation: `adjacencySequence` (a graph as GraphRNN's sequence of adjacency vectors under an ordering) and
 *   `sequenceBandwidth`.
 * - Temporal graphs: `memoryTrace` (the TGN memory replayed event by event), `timeEncoding`, `timeProjection`, and
 *   `timeRespectingReach` against `staticReach`.
 *
 * A graph is `aifn-compute/graph`'s plain data, and the node features a $V \times F$ tensor. Runs are deterministic
 * from their seed. The registry tables `graphNetworkFunctions` and `graphNetworkAlgorithms` list the module's entries.
 */

export {
  gnnModel,
  nodeClassificationRun,
  nodeClassificationTraining,
  nodeLoss,
  type GnnKind,
  type GnnModel,
  type GnnSpec,
  type NodeCheckpoint,
  type NodeClassificationOptions,
  type NodeHistory,
  type NodeSnapshot,
  type NodeTrainingOptions,
} from './node-classification'
export { linkPredictionRun, splitEdges, type EdgeSplit, type LinkCheckpoint, type LinkPredictionOptions } from './links'
export { meanPropagation, propagationStack, type PropagationOptions, type PropagationVariant } from './depth'
export { adjacencySequence, sequenceBandwidth } from './generation'
export {
  memoryTrace,
  staticReach,
  timeEncoding,
  timeProjection,
  timeRespectingReach,
  type MemoryTrace,
  type MemoryUpdate,
  type TimedEvent,
} from './temporal'
export { graphNetworkAlgorithms, graphNetworkFunctions } from './registry'
