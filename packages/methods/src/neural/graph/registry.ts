/**
 * The models and algorithms of `aifn-methods/neural/graph`: semi-supervised node classification with a two-layer GNN.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as nodes from './node-classification'

const NOTES = ['graph-convolutional-network', 'graph-attention-network', 'graphsage']

const fn = definer<FunctionInfo>('function', 'neural/graph')
fn(
  {
    key: 'gnnModel',
    name: 'Two-layer graph neural network',
    summary: 'A GCN, GAT or GraphSAGE layer with tanh, then a layer to one logit per class.',
    role: 'construction',
    notes: NOTES,
    cite: ['kipf2017'],
  },
  nodes.gnnModel,
)
fn(
  {
    key: 'nodeClassificationRun',
    name: 'Semi-supervised node classification run',
    summary: 'Train a two-layer GNN on a few labelled nodes, streaming curves, logits and attention at checkpoints.',
    role: 'simulation',
    random: true,
    notes: NOTES,
    cite: ['kipf2017'],
  },
  nodes.nodeClassificationRun,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'neural/graph')
algorithm(
  {
    key: 'nodeClassificationTraining',
    name: 'Node classification by Adam',
    summary: 'Full-batch Adam on the labelled nodes’ cross-entropy of a two-layer GNN.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: NOTES,
    cite: ['kipf2017'],
  },
  nodes.nodeClassificationTraining,
)

type Table<I extends FunctionInfo | AlgorithmInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** The functions of the module, keyed by name. */
export const graphNetworkFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  nodes,
) as Table<FunctionInfo>
/** The algorithms of the module, keyed by factory name. */
export const graphNetworkAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  nodes,
) as Table<AlgorithmInfo>
