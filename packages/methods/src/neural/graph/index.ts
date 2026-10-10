/**
 * `aifn-methods/neural/graph`: semi-supervised node classification with a two-layer graph neural network, the
 * experiment of Kipf and Welling (2017).
 *
 * - The network: `gnnModel` builds two GCN, GAT or GraphSAGE layers from `aifn-compute/nn/graph`, tanh between them,
 *   one logit per class out (and GAT's attention weights on request).
 * - Training: `nodeLoss` (the cross-entropy on the labelled nodes) and `nodeClassificationTraining` (full-batch Adam
 *   as a traceable training loop).
 * - A streamed run: `nodeClassificationRun` yields the loss and accuracy curves on labelled and unlabelled nodes, and
 *   checkpoints of the logits, hidden features and attention, for a worker to stream to a page.
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
export { graphNetworkAlgorithms, graphNetworkFunctions } from './registry'
