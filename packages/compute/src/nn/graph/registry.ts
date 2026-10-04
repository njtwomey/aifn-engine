/** The layers and functions of `aifn-compute/nn/graph`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as layers from './layers'

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

/** Every function and layer constructor of the module, keyed by name. */
export const graphLayerFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', layers) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
