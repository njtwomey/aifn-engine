/** The layers and functions of `aifn-compute/nn/graph`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as expressive from './expressive'
import * as hypergraph from './hypergraph'
import * as layers from './layers'
import * as pooling from './pooling'
import * as relational from './relational'
import * as sampling from './sampling'
import * as smoothing from './smoothing'
import * as spectral from './spectral'
import * as transformer from './transformer'

const fn = definer<FunctionInfo>('function', 'nn/graph')
const GCN = ['graph-convolutional-network']
const GAT = ['graph-attention-network']
const SAGE = ['graphsage']
const MPNN = ['message-passing-neural-network']

fn(
  {
    key: 'graphConv',
    name: 'Graph convolution',
    tex: "H' = \\hat A H W + b",
    summary: 'Degree-normalised averaging over each node and its neighbours, then a linear map.',
    role: 'transform',
    notes: [...GCN, ...MPNN],
    cite: ['kipf2017'],
  },
  layers.graphConv,
)
fn(
  {
    key: 'gcnCoefficients',
    name: 'GCN propagation coefficients',
    tex: '\\hat A = \\tilde D^{-1/2}(A + I)\\tilde D^{-1/2}',
    summary: 'The weight of every message edge: symmetric, random-walk or unnormalised, with self-loops.',
    role: 'construction',
    notes: GCN,
    cite: ['kipf2017'],
  },
  layers.gcnCoefficients,
)
fn(
  { key: 'GraphConv', name: 'Graph convolution layer', role: 'construction', notes: GCN, cite: ['kipf2017'] },
  layers.GraphConv,
)
fn(
  {
    key: 'graphAttention',
    name: 'Graph attention',
    tex: "h'_v = \\sum_{u \\to v} \\alpha_{uv} W h_u",
    summary: 'Multi-head attention over each node’s neighbourhood with LeakyReLU scores (GAT) or GATv2 scores.',
    role: 'transform',
    notes: [...GAT, ...MPNN],
    cite: ['velickovic2018', 'brody2022'],
  },
  layers.graphAttention,
)
fn(
  {
    key: 'GraphAttention',
    name: 'Graph attention layer',
    role: 'construction',
    notes: GAT,
    cite: ['velickovic2018', 'brody2022'],
  },
  layers.GraphAttention,
)
fn(
  {
    key: 'sageConv',
    name: 'GraphSAGE convolution',
    tex: "h'_v = W_1 h_v + W_2\\,\\mathrm{AGG}\\{h_u\\}",
    summary: 'A node’s own features plus an aggregate (mean, sum, max or max-pooling) of its neighbours’.',
    role: 'transform',
    notes: [...SAGE, ...MPNN],
    cite: ['hamilton2017graphsage'],
  },
  layers.sageConv,
)
fn(
  { key: 'SageConv', name: 'GraphSAGE layer', role: 'construction', notes: SAGE, cite: ['hamilton2017graphsage'] },
  layers.SageConv,
)
fn(
  {
    key: 'sampleNeighbours',
    name: 'Neighbourhood sampling',
    summary: 'Keep a uniform sample of at most S incoming edges per node, bounding a layer’s cost.',
    role: 'simulation',
    random: true,
    notes: SAGE,
    cite: ['hamilton2017graphsage'],
  },
  layers.sampleNeighbours,
)
fn(
  {
    key: 'messagePassing',
    name: 'Message passing step',
    tex: "m_v = \\bigoplus_{u \\to v} M(h_u, h_v, w_{uv}),\\; h'_v = U(h_v, m_v)",
    summary: 'A generic step: a message function on each edge, an aggregate at each node, an update function.',
    role: 'transform',
    notes: MPNN,
    cite: ['gilmer2017'],
  },
  layers.messagePassing,
)
fn(
  { key: 'MessagePassing', name: 'Message-passing layer', role: 'construction', notes: MPNN, cite: ['gilmer2017'] },
  layers.MessagePassing,
)

const GIN = ['graph-isomorphism-network', 'expressive-power-of-message-passing']
fn(
  {
    key: 'ginConv',
    name: 'Graph isomorphism network convolution',
    tex: "h'_v = \\mathrm{MLP}\\big((1 + \\epsilon) h_v + \\sum_{u \\to v} h_u\\big)",
    summary: 'An MLP of a node’s own vector plus the sum of its neighbours’: as strong as the 1-WL test.',
    role: 'transform',
    notes: GIN,
    cite: ['xu2019'],
  },
  expressive.ginConv,
)
fn({ key: 'GinConv', name: 'GIN layer', role: 'construction', notes: GIN, cite: ['xu2019'] }, expressive.GinConv)
const PNA = ['principal-neighbourhood-aggregation']
fn(
  {
    key: 'pnaAggregate',
    name: 'Principal neighbourhood aggregation',
    tex: '\\bigoplus = \\big[I, S(d, 1), S(d, -1)\\big] \\otimes \\big[\\mu, \\max, \\min, \\sigma\\big]',
    summary: 'Several aggregators of the neighbours, each scaled by functions of the degree, concatenated.',
    role: 'transform',
    notes: PNA,
    cite: ['corso2020pna'],
  },
  expressive.pnaAggregate,
)
fn(
  {
    key: 'pnaDelta',
    name: 'PNA degree normaliser',
    tex: '\\delta = \\frac{1}{|V|}\\sum_v \\log(d_v + 1)',
    summary: 'The mean log degree over the training graphs, which centres the degree scalers.',
    role: 'property',
    notes: PNA,
    cite: ['corso2020pna'],
  },
  expressive.pnaDelta,
)
const CHEB = ['spectral-graph-convolution']
fn(
  {
    key: 'chebyshevTerms',
    name: 'Chebyshev terms of a graph signal',
    tex: 'T_k(\\tilde L)x = 2\\tilde L\\,T_{k-1}(\\tilde L)x - T_{k-2}(\\tilde L)x',
    summary: 'The Chebyshev polynomials of the rescaled Laplacian applied to a signal, by the recurrence.',
    role: 'transform',
    notes: CHEB,
    cite: ['hammond2011', 'defferrard2016'],
  },
  spectral.chebyshevTerms,
)
fn(
  {
    key: 'chebyshevFilter',
    name: 'Chebyshev graph filter',
    tex: 'y = \\sum_k \\theta_k T_k(\\tilde L)x',
    summary: 'A polynomial spectral filter, exactly K-hop local, with no eigendecomposition.',
    role: 'transform',
    notes: CHEB,
    cite: ['hammond2011', 'defferrard2016'],
  },
  spectral.chebyshevFilter,
)
fn(
  {
    key: 'chebConv',
    name: 'ChebNet convolution',
    tex: "H' = \\sum_k T_k(\\tilde L) H \\Theta_k + b",
    summary: 'One weight matrix per Chebyshev order of the rescaled Laplacian.',
    role: 'transform',
    notes: CHEB,
    cite: ['defferrard2016'],
  },
  spectral.chebConv,
)
fn(
  { key: 'ChebConv', name: 'ChebNet layer', role: 'construction', notes: CHEB, cite: ['defferrard2016'] },
  spectral.ChebConv,
)
const READOUT = ['graph-pooling-and-readout', 'molecular-property-prediction-with-graph-networks']
fn(
  {
    key: 'batchGraphs',
    name: 'Batch of graphs',
    summary: 'Several graphs as one block-diagonal graph, with the graph index of every node.',
    role: 'construction',
    notes: READOUT,
  },
  pooling.batchGraphs,
)
fn(
  {
    key: 'globalPool',
    name: 'Global readout',
    tex: 'r_b = \\bigoplus_{v : b_v = b} h_v',
    summary: 'The sum, mean or max of each graph’s node rows.',
    role: 'transform',
    notes: READOUT,
    cite: ['xu2019'],
  },
  pooling.globalPool,
)
fn(
  {
    key: 'attentionPool',
    name: 'Attention readout',
    tex: 'r_b = \\sum_{v : b_v = b} \\mathrm{softmax}_b(s)_v\\, h_v',
    summary: 'A weighted mean of each graph’s node rows, the weights a softmax of gate scores within the graph.',
    role: 'transform',
    notes: READOUT,
    cite: ['li2016gated'],
  },
  pooling.attentionPool,
)
fn(
  {
    key: 'diffPool',
    name: 'DiffPool coarsening',
    tex: "A' = S^\\top A S,\\; X' = S^\\top Z",
    summary: 'Coarsen a graph by a learned soft assignment of nodes to clusters, with link and entropy losses.',
    role: 'transform',
    notes: ['graph-pooling-and-readout'],
    cite: ['ying2018diffpool'],
  },
  pooling.diffPool,
)
fn(
  {
    key: 'topKPool',
    name: 'Top-k pooling (gPool)',
    tex: "y = Hp / \\lVert p \\rVert,\\; H' = H_{\\mathrm{top}\\,k} \\odot \\tanh(y_{\\mathrm{top}\\,k})",
    summary: 'Keep the k nodes with the highest projection scores, gate their rows and keep the edges among them.',
    role: 'transform',
    notes: ['graph-pooling-and-readout'],
    cite: ['gao2019graphunet'],
  },
  pooling.topKPool,
)
const DEPTH = ['oversmoothing', 'residual-and-jumping-knowledge-graph-networks']
fn(
  {
    key: 'gcnMatrix',
    name: 'GCN propagation matrix',
    tex: '\\hat A = \\tilde D^{-1/2}(A + I)\\tilde D^{-1/2}',
    summary: 'The dense matrix of a graph convolution’s propagation coefficients.',
    role: 'construction',
    notes: [...DEPTH, 'oversquashing-and-graph-rewiring'],
    cite: ['kipf2017'],
  },
  smoothing.gcnMatrix,
)
fn(
  {
    key: 'adjacencySpectrum',
    name: 'Spectrum of the GCN matrix',
    tex: '\\lambda = \\max(|\\mu_2|, |\\mu_V|)',
    summary: 'The eigenvalues of the propagation matrix and the contraction factor that bounds smoothing per layer.',
    role: 'property',
    notes: ['oversmoothing'],
    cite: ['cai2020oversmoothing'],
  },
  smoothing.adjacencySpectrum,
)
fn(
  {
    key: 'dirichletEnergy',
    name: 'Dirichlet energy',
    tex: 'E(H) = \\frac12 \\sum_{i,j} \\tilde A_{ij} \\| h_i/\\sqrt{\\tilde d_i} - h_j/\\sqrt{\\tilde d_j} \\|^2',
    summary: 'How far node features are from the smooth limit of propagation: zero when oversmoothed.',
    role: 'property',
    notes: DEPTH,
    cite: ['cai2020oversmoothing'],
  },
  smoothing.dirichletEnergy,
)
fn(
  {
    key: 'pairNorm',
    name: 'PairNorm',
    tex: "x'_i = s\\,(x_i - \\bar x) \\big/ \\sqrt{\\tfrac1n \\sum_j \\lVert x_j - \\bar x \\rVert^2}",
    summary: 'Centre the features and fix their mean squared norm, so propagation cannot collapse them.',
    role: 'transform',
    notes: DEPTH,
    cite: ['zhao2020pairnorm'],
  },
  smoothing.pairNorm,
)
fn(
  {
    key: 'dropEdge',
    name: 'DropEdge',
    summary: 'Keep each edge with probability 1 - p, a fresh subgraph per layer or step.',
    role: 'simulation',
    random: true,
    notes: DEPTH,
    cite: ['rong2020dropedge'],
  },
  smoothing.dropEdge,
)
fn(
  {
    key: 'propagationSensitivity',
    name: 'Propagation sensitivity',
    tex: '\\left|\\partial h_v^{(r)} / \\partial x_u\\right| \\lesssim (\\hat A^r)_{vu}',
    summary: 'The powers of the GCN matrix at one node: how much each node can influence it through r layers.',
    role: 'property',
    notes: ['oversquashing-and-graph-rewiring'],
    cite: ['topping2022curvature'],
  },
  smoothing.propagationSensitivity,
)
const TRANSFORMER = ['graph-transformer']
fn(
  {
    key: 'spatialBias',
    name: 'Graphormer spatial bias',
    tex: 'B_{ij} = b_{\\phi(i, j)}',
    summary: 'An attention bias per pair of nodes, looked up by their shortest-path distance.',
    role: 'construction',
    notes: TRANSFORMER,
    cite: ['ying2021graphormer'],
  },
  transformer.spatialBias,
)
fn(
  {
    key: 'graphormerAttention',
    name: 'Graphormer attention',
    tex: '\\mathrm{softmax}(QK^\\top/\\sqrt{d_k} + B)V',
    summary: 'Full self-attention over the nodes, the graph entering only through the spatial bias.',
    role: 'transform',
    notes: TRANSFORMER,
    cite: ['ying2021graphormer'],
  },
  transformer.graphormerAttention,
)
const SCALE = ['neighbour-sampling-and-scalable-graph-networks']
fn(
  {
    key: 'computationGraph',
    name: 'Computation graph of a minibatch',
    summary: 'The nodes an L-layer network needs for its targets, depth by depth: the neighbourhood explosion.',
    role: 'property',
    notes: SCALE,
    cite: ['hamilton2017graphsage'],
  },
  sampling.computationGraph,
)
fn(
  {
    key: 'sampledComputationGraph',
    name: 'Sampled computation graph',
    summary: 'The nodes needed when each node samples at most S neighbours per depth, as GraphSAGE does.',
    role: 'simulation',
    random: true,
    notes: SCALE,
    cite: ['hamilton2017graphsage'],
  },
  sampling.sampledComputationGraph,
)
const HYPER = ['hypergraph-neural-networks']
fn(
  {
    key: 'hypergraphMeanPass',
    name: 'Two-stage hypergraph mean',
    summary: 'Each hyperedge averages its members, then each node averages its hyperedges.',
    role: 'transform',
    notes: HYPER,
    cite: ['zhou2006hypergraph'],
  },
  hypergraph.hypergraphMeanPass,
)
fn(
  {
    key: 'hgnnConv',
    name: 'HGNN convolution',
    tex: 'D_v^{-1/2} H W D_e^{-1} H^\\top D_v^{-1/2} X \\Theta',
    summary: 'The hypergraph convolution: node to hyperedge and back, degree-normalised on both sides.',
    role: 'transform',
    notes: HYPER,
    cite: ['feng2019hgnn'],
  },
  hypergraph.hgnnConv,
)
const RGCN = ['heterogeneous-graph-networks']
fn(
  {
    key: 'rgcnTerms',
    name: 'R-GCN relation terms',
    tex: '\\frac{1}{c_{i,r}} \\sum_{j \\in N_i^r} W_r^\\top h_j',
    summary: 'Each relation’s normalised sum of transformed neighbour vectors.',
    role: 'transform',
    notes: RGCN,
    cite: ['schlichtkrull2018rgcn'],
  },
  relational.rgcnTerms,
)
fn(
  {
    key: 'rgcnConv',
    name: 'Relational graph convolution',
    tex: "h'_i = W_0^\\top h_i + \\sum_r \\frac{1}{c_{i,r}} \\sum_{j \\in N_i^r} W_r^\\top h_j",
    summary: 'A self-connection plus one weighted, normalised neighbour sum per relation.',
    role: 'transform',
    notes: RGCN,
    cite: ['schlichtkrull2018rgcn'],
  },
  relational.rgcnConv,
)
fn(
  {
    key: 'basisWeights',
    name: 'R-GCN basis decomposition',
    tex: 'W_r = \\sum_b a_{rb} V_b',
    summary: 'Every relation weight as a combination of a few shared bases.',
    role: 'construction',
    notes: RGCN,
    cite: ['schlichtkrull2018rgcn'],
  },
  relational.basisWeights,
)

/** Every function and layer constructor of the module, keyed by name. */
export const graphLayerFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    layers,
    expressive,
    spectral,
    pooling,
    smoothing,
    transformer,
    sampling,
    hypergraph,
    relational,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
