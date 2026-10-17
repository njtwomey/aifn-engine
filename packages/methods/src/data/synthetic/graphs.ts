/**
 * Graph datasets: for node classification and link prediction, the contextual stochastic block model, a graph whose
 * nodes carry features and classes, with the conversion from a target degree and homophily to its edge probabilities;
 * for graph classification, `graphFamilies`, small cycles, stars and random trees.
 *
 * Node $i$ belongs to block $y_i$; each pair of nodes is joined independently with probability $p_{\text{in}}$ when
 * they share a block and $p_{\text{out}}$ otherwise (the planted-partition stochastic block model of Holland, Laskey
 * and Leinhardt 1983, "Stochastic blockmodels: first steps", Social Networks 5(2), drawn by `stochasticBlockModel` of
 * `aifn-compute/graph/structures`). Each node carries the features
 *
 * $$\xvec_i = \mu\,\evec_{y_i} + \epsilonvec_i \in \reals^d, \qquad \epsilonvec_i \sim \Gauss(\zeros, \Imat),$$
 *
 * with $\evec_c$ the $c$-th unit vector and $d$ at least the number of blocks: the contextual form of Deshpande, Sen,
 * Montanari and Mossel (2018), "Contextual stochastic block models", NeurIPS, with the class means along unit vectors
 * in place of a random direction. The edge probabilities set how much the graph says about the classes (homophily)
 * and $\mu$ how much the features say.
 */

import type { DatasetInfo, Graph, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { edgeHomophily } from 'aifn-compute/graph/measures'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  blockMembership,
  cycleGraph,
  randomRecursiveTree,
  starGraph,
  stochasticBlockModel,
} from 'aifn-compute/graph/structures'
import { checkCount, generatorRecipe, labels, matrix, type GraphDataset } from '../types'

/** Options of `contextualBlockModel`. */
export interface BlockModelOptions {
  /** Nodes per block, e.g. `[30, 30]`; nodes are numbered block by block. Default `blocks` blocks of `blockSize`. */
  sizes?: readonly Size[]
  /** The number of blocks $C$ when `sizes` is left out (default 2). */
  blocks?: Size
  /** The nodes per block when `sizes` is left out (default 40). */
  blockSize?: Size
  /** The probability $p_{\text{in}}$ of an edge between two nodes of the same block, in $[0, 1]$ (default 0.12). */
  pIn?: number
  /** The probability $p_{\text{out}}$ of an edge between two nodes of different blocks, in $[0, 1]$ (default 0.02). */
  pOut?: number
  /** The feature dimension $d$, at least the number of blocks (default the number of blocks). */
  features?: Size
  /** The feature signal $\mu \ge 0$: the distance of each class mean from the origin, in noise standard deviations. */
  signal?: number
}

/** A contextual block model: a `GraphDataset` with the edge homophily of its graph. */
export interface BlockModelDataset extends GraphDataset {
  /** The block of each node. */
  readonly y: Tensor
  /** The share of edges that join two nodes of the same block (edge homophily); NaN with no edges. */
  readonly homophily: number
}

/**
 * Draw a contextual stochastic block model (see the file comment). Throws `DomainError` for an empty block, a
 * probability outside $[0, 1]$, a negative signal or fewer feature dimensions than blocks.
 *
 * @param s The random stream: the edges are drawn from its child `'edges'` (one uniform per pair $i < j$, in
 *   row-major order, as `stochasticBlockModel` draws), and the features from its child `'features'`, row by row.
 * @param options The block sizes, the two edge probabilities, the feature dimension and the feature signal.
 * @returns A classification `GraphDataset`: `x` ($n \times d$), `y` the block of each node (int32), `graph` the
 *   undirected graph, and `homophily`, the share of edges within a block.
 *
 * @example Two blocks of 40: homophilous edges and informative features
 * const d = contextualBlockModel(stream(0), { sizes: [40, 40], pIn: 0.12, pOut: 0.02, signal: 2 })
 * print('nodes:', d.graph.nodes, ' edges:', d.graph.edges.length, ' homophily:', d.homophily)
 * print('x:', d.x.shape, ' first labels:', toFlat(d.y).slice(0, 3))
 *
 * @example Equal probabilities: about half of the edges stay in a block
 * const d = contextualBlockModel(stream(1), { sizes: [40, 40], pIn: 0.06, pOut: 0.06 })
 * print('homophily:', d.homophily)
 */
export function contextualBlockModel(s: Stream, options: BlockModelOptions = {}): BlockModelDataset {
  const where = 'contextualBlockModel'
  const { blocks = 2, blockSize = 40, pIn = 0.12, pOut = 0.02, signal = 1 } = options
  const sizes = options.sizes ?? Array.from({ length: blocks }, () => blockSize)
  const C = sizes.length
  const d = options.features ?? C
  if (C < 1 || sizes.some((m) => !(Number.isInteger(m) && m >= 1)))
    throw new DomainError(where, `${where}: block sizes must be positive integers`)
  if (!(pIn >= 0 && pIn <= 1 && pOut >= 0 && pOut <= 1))
    throw new DomainError(where, `${where}: probabilities must lie in [0, 1]`)
  if (!(signal >= 0)) throw new DomainError(where, `${where}: the signal must be non-negative`)
  if (!(Number.isInteger(d) && d >= C)) throw new DomainError(where, `${where}: need at least one feature per block`)
  const graph = stochasticBlockModel(s, sizes, { pIn, pOut })
  const y = blockMembership(sizes)
  const block = y.data
  const n = block.length
  // The cosine half of each Box-Muller pair: the draws a scalar `normal` would give, one per feature in row order.
  const z = standardNormals(child(s, 'features'), 2 * n * d)
  const x = new Float64Array(n * d)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < d; k++) x[i * d + k] = (k === block[i] ? signal : 0) + z[2 * (i * d + k)]
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: labels(block),
    graph,
    homophily: edgeHomophily(graph, y),
    meta: {
      name: 'contextual block model',
      description:
        `${C} blocks of ${sizes.join(', ')} nodes, edges with probability ${pIn} within a block and ${pOut} ` +
        `between blocks, features ${signal} e_y + N(0, I) in ${d} dimensions.`,
      task: 'classification',
      featureNames: Array.from({ length: d }, (_, k) => `x${k + 1}`),
      labelNames: Array.from({ length: C }, (_, c) => `block ${c}`),
      source: 'Deshpande, Sen, Montanari and Mossel (2018), "Contextual stochastic block models", NeurIPS',
      key: s.key,
      recipe: generatorRecipe('contextualBlockModel', s.key, { sizes: [...sizes], pIn, pOut, features: d, signal }),
    },
  }
}

/**
 * The edge probabilities of a stochastic block model with `classes` equal blocks of $m$ nodes in which a node expects
 * $d$ neighbours, a share $h$ of them (the edge homophily) in its own block:
 *
 * $$p_{\text{in}} = \frac{h d}{m - 1}, \qquad p_{\text{out}} = \frac{(1 - h) d}{m (C - 1)}.$$
 *
 * Throws `DomainError` unless there are at least two blocks of at least two nodes, $d \ge 0$ and $h \in [0, 1]$, or
 * when a probability would exceed 1 (the degree is too large for the blocks).
 *
 * @param options The model's shape and targets.
 * @param options.classes The number of blocks $C \ge 2$.
 * @param options.blockSize The nodes per block $m \ge 2$.
 * @param options.degree The expected degree $d$ of every node.
 * @param options.homophily The expected share $h$ of a node's neighbours in its own block.
 * @returns `pIn` and `pOut`, ready for `contextualBlockModel` or `stochasticBlockModel`.
 *
 * @example Blocks of 100, degree 6 and homophily 0.2: a heterophilous graph
 * const p = blockModelProbabilities({ classes: 2, blockSize: 100, degree: 6, homophily: 0.2 })
 * print('pIn =', p.pIn, ' pOut =', p.pOut)
 * const d = contextualBlockModel(stream(0), { sizes: [100, 100], ...p })
 * print('measured homophily:', d.homophily)
 */
export function blockModelProbabilities(options: {
  classes: number
  blockSize: number
  degree: number
  homophily: number
}): { pIn: number; pOut: number } {
  const { classes, blockSize: m, degree: d, homophily: h } = options
  const where = 'blockModelProbabilities'
  if (!(classes >= 2 && m >= 2 && d >= 0 && h >= 0 && h <= 1))
    throw new DomainError(where, `${where}: need two or more blocks of two or more nodes, degree ≥ 0 and h in [0, 1]`)
  const pIn = (h * d) / (m - 1)
  const pOut = ((1 - h) * d) / (m * (classes - 1))
  if (pIn > 1 || pOut > 1) throw new DomainError(where, `${where}: the degree is too large for the block size`)
  return { pIn, pOut }
}

/** The graph families of `graphFamilies`, in label order. */
export const GRAPH_FAMILIES = ['cycle', 'star', 'tree'] as const

/** A family of `graphFamilies`. */
export type GraphFamily = (typeof GRAPH_FAMILIES)[number]

/** Options of `graphFamilies`. */
export interface GraphFamilyOptions {
  /** The number of graphs; family $k$ is used for graphs $k, k + 3, k + 6, \dots$ (default 90). */
  count?: number
  /** The smallest node count, at least 3 (default 6). */
  minNodes?: number
  /** The largest node count, inclusive (default 13). */
  maxNodes?: number
}

/** A labelled collection of small graphs. */
export interface GraphFamilyDataset {
  /** The graphs, undirected. */
  readonly graphs: readonly Graph[]
  /** The family index of each graph (0 cycle, 1 star, 2 tree). */
  readonly family: readonly number[]
  /** The node count of each graph. */
  readonly size: readonly number[]
}

/**
 * Small graphs for graph classification: cycles, stars and random recursive trees (`randomRecursiveTree`) on $n$
 * nodes, $n$ uniform on `minNodes` to `maxNodes`, cycling through the families. Two labels come with every graph, its
 * family and its size, so the same graphs pose a task that needs the shape and one that needs a count. With constant
 * node features every node of a cycle sees the same neighbourhood at every depth, so a mean or max readout gives every
 * cycle the same vector whatever its length; only a sum readout can tell their sizes apart. Throws `DomainError`
 * unless $3 \le$ `minNodes` $\le$ `maxNodes` are integers.
 *
 * @param s The random stream: the sizes are drawn from its child `'size'`, one uniform per graph in order, and the
 *   trees from its child `'tree'`, shared by the trees in order.
 * @param options The number of graphs and the size range.
 * @returns The graphs with their family and size labels.
 *
 * @example Six graphs: two of each family
 * const d = graphFamilies(stream(1), { count: 6 })
 * print('families:', d.family, ' sizes:', d.size)
 * print('edges:', d.graphs.map((g) => g.edges.length))
 */
export function graphFamilies(s: Stream, options: GraphFamilyOptions = {}): GraphFamilyDataset {
  const { count = 90, minNodes = 6, maxNodes = 13 } = options
  if (!(Number.isInteger(minNodes) && minNodes >= 3 && Number.isInteger(maxNodes) && maxNodes >= minNodes))
    throw new DomainError('graphFamilies', 'graphFamilies: need integer sizes with 3 ≤ minNodes ≤ maxNodes')
  checkCount(count, 'graphFamilies')
  const u = count > 0 ? toFlat(uniform(child(s, 'size'), 0, 1, { shape: [count] })) : []
  const trees = child(s, 'tree')
  const graphs: Graph[] = []
  const family: number[] = []
  const size: number[] = []
  for (let i = 0; i < count; i++) {
    const f = i % GRAPH_FAMILIES.length
    const n = minNodes + Math.floor(u[i] * (maxNodes - minNodes + 1))
    graphs.push(f === 0 ? cycleGraph(n) : f === 1 ? starGraph(n - 1) : randomRecursiveTree(trees, n))
    family.push(f)
    size.push(n)
  }
  return { graphs, family, size }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'contextualBlockModel',
    name: 'Contextual stochastic block model',
    summary: 'A block-model graph whose nodes carry noisy features pointing to their block: node classification.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      blocks: int(2, 10, { default: 2 }),
      blockSize: int(2, 500, { default: 40 }),
      pIn: real(0, 1, { default: 0.12 }),
      pOut: real(0, 1, { default: 0.02 }),
      signal: real(0, 10, { default: 1 }),
    }),
    truth: false,
    random: true,
    notes: [
      'semi-supervised-node-classification',
      'graph-neural-networks-under-heterophily',
      'graph-learning-benchmarks',
      'link-prediction-with-graph-neural-networks',
      'oversmoothing',
      'residual-and-jumping-knowledge-graph-networks',
    ],
    cite: ['deshpande2018csbm'],
  },
  contextualBlockModel,
)
