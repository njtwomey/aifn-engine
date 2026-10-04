/**
 * `aifn-methods/neural/graph`: semi-supervised node classification with a two-layer graph neural network (GCN, GAT or
 * GraphSAGE from `aifn-compute/nn/graph`), trained by Adam on a few labelled nodes; `nodeClassificationRun` streams a run.
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
