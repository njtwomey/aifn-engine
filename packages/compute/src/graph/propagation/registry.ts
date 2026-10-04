/** The functions of `aifn-compute/graph/propagation`, registered with the notes they serve. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as labels from './labels'
import * as propagation from './propagation'

const fn = definer<FunctionInfo>('function', 'graph/propagation')

fn(
  {
    key: 'propagate',
    name: 'Message passing',
    summary:
      'Gather source features along edges, apply an edge function, aggregate at destinations (sum, mean or max).',
    role: 'transform',
    notes: ['message-passing-neural-network', 'graph-convolutional-network', 'graph-attention-network'],
  },
  propagation.propagate,
)
fn(
  { key: 'messageEdges', name: 'Message edges', role: 'construction', notes: ['message-passing-neural-network'] },
  propagation.messageEdges,
)

fn(
  {
    key: 'aggregateEdges',
    name: 'Segment aggregation over edges',
    summary: 'Sum, mean or max of per-edge rows at their destination nodes.',
    role: 'transform',
    notes: ['message-passing-neural-network', 'graphsage'],
  },
  propagation.aggregateEdges,
)
fn(
  {
    key: 'edgeSoftmax',
    name: 'Softmax over incoming edges',
    tex: "\\alpha_{uv} = \\frac{e^{s_{uv}}}{\\sum_{u' \\to v} e^{s_{u'v}}}",
    summary: 'Normalise edge scores over each node’s incoming edges, per head: the attention weights of a GAT.',
    role: 'transform',
    notes: ['graph-attention-network'],
    cite: ['velickovic2018'],
  },
  propagation.edgeSoftmax,
)

const LP = ['label-propagation']

fn(
  {
    key: 'harmonicLabels',
    name: 'Harmonic label propagation',
    tex: 'F_U = (D_{UU} - W_{UU})^{-1} W_{UL} Y_L',
    summary: 'Scores that minimise the weighted squared differences across edges with the labelled nodes fixed.',
    role: 'solver',
    notes: LP,
    cite: ['zhu2003'],
  },
  labels.harmonicLabels,
)
fn(
  {
    key: 'labelSpreading',
    name: 'Label spreading',
    tex: 'F^* = (1-\\alpha)(I - \\alpha S)^{-1} Y',
    summary: 'The limit of F ← αSF + (1 − α)Y with the normalised affinity S: labels spread and may be revised.',
    role: 'solver',
    notes: LP,
    cite: ['zhou2004lgc'],
  },
  labels.labelSpreading,
)
fn(
  {
    key: 'spreadingMatrix',
    name: 'Normalised affinity for spreading',
    tex: 'S = D^{-1/2} W D^{-1/2}',
    role: 'construction',
    notes: [...LP, 'spectral-clustering'],
    cite: ['zhou2004lgc'],
  },
  labels.spreadingMatrix,
)
fn(
  {
    key: 'graphAffinity',
    name: 'Graph affinity matrix',
    summary: 'Edge weights as affinities: as given, connectivity, or a heat kernel of distances.',
    role: 'construction',
    notes: [...LP, 'spectral-clustering'],
  },
  labels.graphAffinity,
)
fn(
  {
    key: 'pointAffinity',
    name: 'Gaussian affinity of a point cloud',
    tex: 'W_{ij} = e^{-\\gamma \\lVert x_i - x_j \\rVert^2}',
    summary: 'Gaussian affinities between points, zero on the diagonal, optionally kept only on a k-NN graph.',
    role: 'construction',
    notes: [...LP, 'label-propagation-for-label-proportions', 'spectral-clustering'],
    cite: ['zhou2004lgc'],
  },
  labels.pointAffinity,
)
fn(
  {
    key: 'randomWalkMatrix',
    name: 'Random-walk matrix',
    tex: 'S = D^{-1} W',
    summary: 'Row-normalised affinities with the diagonal removed: the transition probabilities of a random walk.',
    role: 'construction',
    notes: [...LP, 'label-propagation-for-label-proportions'],
    cite: ['poyiadzi2018'],
  },
  labels.randomWalkMatrix,
)
fn(
  {
    key: 'spreadingResolvent',
    name: 'Spreading resolvent',
    tex: 'R = (1-\\alpha)(I - \\alpha S)^{-1}',
    summary: 'The matrix that maps given labels to the limit of spreading: a geometric sum of walks of every length.',
    role: 'construction',
    notes: [...LP, 'label-propagation-for-label-proportions'],
    cite: ['zhou2004lgc'],
  },
  labels.spreadingResolvent,
)
fn({ key: 'labelMatrix', name: 'One-hot label matrix', role: 'construction', notes: LP }, labels.labelMatrix)
fn({ key: 'normaliseScores', name: 'Normalised class scores', role: 'transform', notes: LP }, labels.normaliseScores)

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/propagation')
algorithm(
  {
    key: 'labelPropagationSteps',
    name: 'Label propagation',
    summary: 'F ← D⁻¹WF with the labelled rows clamped, converging to the harmonic solution.',
    problem: 'graph',
    state: { iterate: 'scores', flags: ['converged'] },
    notes: LP,
    cite: ['zhu2003'],
  },
  labels.labelPropagationSteps,
)
algorithm(
  {
    key: 'labelSpreadingSteps',
    name: 'Label spreading',
    summary: 'F ← αSF + (1 − α)Y from F₀ = Y.',
    problem: 'graph',
    state: { iterate: 'scores', flags: ['converged'] },
    notes: LP,
    cite: ['zhou2004lgc'],
  },
  labels.labelSpreadingSteps,
)

/** The label-propagation algorithms, keyed by factory name. */
export const propagationAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', labels) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const propagationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', propagation, labels) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
