/**
 * The models and algorithms of `aifn-methods/neural/graph`: semi-supervised node classification with a two-layer GNN.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as depth from './depth'
import * as generation from './generation'
import * as links from './links'
import * as nodes from './node-classification'
import * as temporal from './temporal'

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

const LINKS = ['link-prediction-with-graph-neural-networks']
fn(
  {
    key: 'splitEdges',
    name: 'Link-prediction edge split',
    summary: 'Hold out a share of the edges and draw as many non-edges as test pairs; train on the rest.',
    role: 'construction',
    random: true,
    notes: LINKS,
    cite: ['kipf2016vgae'],
  },
  links.splitEdges,
)
fn(
  {
    key: 'linkPredictionRun',
    name: 'Graph autoencoder run',
    tex: '\\sigma(z_u^\\top z_v),\\; Z = \\hat A\\,\\mathrm{ReLU}(\\hat A X W_1) W_2',
    summary: 'Train a graph autoencoder on the kept edges, streaming the held-out AUC.',
    role: 'simulation',
    random: true,
    notes: LINKS,
    cite: ['kipf2016vgae'],
  },
  links.linkPredictionRun,
)
const DEPTH = ['oversmoothing', 'residual-and-jumping-knowledge-graph-networks']
fn(
  {
    key: 'propagationStack',
    name: 'Weight-free propagation stack',
    summary: 'Every layer of plain, residual, initial-residual, PairNorm or DropEdge propagation without weights.',
    role: 'simulation',
    notes: DEPTH,
    cite: ['kipf2017', 'chen2020gcnii', 'zhao2020pairnorm', 'rong2020dropedge'],
  },
  depth.propagationStack,
)
fn(
  {
    key: 'meanPropagation',
    name: 'Mean propagation',
    tex: "h'_v = \\tfrac12\\big(h_v + \\mathrm{mean}_{u \\to v} h_u\\big)",
    summary: 'A fixed mean layer, on the full graph or on per-depth neighbour samples.',
    role: 'transform',
    notes: ['neighbour-sampling-and-scalable-graph-networks'],
    cite: ['hamilton2017graphsage'],
  },
  depth.meanPropagation,
)
const GENERATION = ['graph-generative-models']
fn(
  {
    key: 'adjacencySequence',
    name: 'Adjacency sequence',
    summary: 'A graph as the rows of its reordered lower-triangular adjacency: what GraphRNN generates.',
    role: 'construction',
    notes: GENERATION,
    cite: ['you2018graphrnn'],
  },
  generation.adjacencySequence,
)
fn(
  {
    key: 'sequenceBandwidth',
    name: 'Sequence bandwidth',
    tex: 'M = \\max_{S_{ij} = 1} (i - j)',
    summary: 'The furthest any edge reaches back in an adjacency sequence.',
    role: 'property',
    notes: GENERATION,
    cite: ['you2018graphrnn'],
  },
  generation.sequenceBandwidth,
)
const TEMPORAL = ['temporal-graph-networks']
fn(
  {
    key: 'memoryTrace',
    name: 'Temporal memory replay',
    tex: 's_u \\leftarrow \\mathrm{mem}(s_u, s_v, e, \\Delta t_u)',
    summary: 'Replay timed events, updating both endpoints’ memories from their states just before each.',
    role: 'simulation',
    notes: TEMPORAL,
    cite: ['rossi2020tgn'],
  },
  temporal.memoryTrace,
)
fn(
  {
    key: 'timeEncoding',
    name: 'Functional time encoding',
    tex: '\\phi(\\Delta t) = [\\cos(\\omega_k \\Delta t + b_k)]_k',
    summary: 'Elapsed time as a vector of cosines of given frequencies and phases.',
    role: 'transform',
    notes: TEMPORAL,
    cite: ['xu2020tgat'],
  },
  temporal.timeEncoding,
)
fn(
  {
    key: 'timeProjection',
    name: 'JODIE time projection',
    tex: '\\hat s(t + \\Delta) = (1 + \\Delta w) \\odot s(t)',
    summary: 'Project a memory forward in time by a learned elementwise drift.',
    role: 'transform',
    notes: TEMPORAL,
    cite: ['kumar2019jodie'],
  },
  temporal.timeProjection,
)
fn(
  {
    key: 'timeRespectingReach',
    name: 'Time-respecting reach',
    summary: 'The nodes information can have reached along paths of increasing event times.',
    role: 'property',
    notes: TEMPORAL,
  },
  temporal.timeRespectingReach,
)
fn(
  {
    key: 'staticReach',
    name: 'Static reach',
    summary: 'The nodes joined by any path of the events so far, ignoring their times.',
    role: 'property',
    notes: TEMPORAL,
  },
  temporal.staticReach,
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
  links,
  depth,
  generation,
  temporal,
) as Table<FunctionInfo>
/** The algorithms of the module, keyed by factory name. */
export const graphNetworkAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  nodes,
) as Table<AlgorithmInfo>
