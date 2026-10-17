/**
 * Random-walk node embeddings: node2vec's biased second-order walks, their window co-occurrence counts, and skip-gram
 * with negative sampling (SGNS) fitted to the walks.
 *
 * `node2vecWalks` samples truncated second-order random walks: a walker that arrived at $v$ from $t$ moves to a
 * neighbour $x$ of $v$ with probability proportional to $w_{vx}\,\alpha_{pq}(t, x)$, where $\alpha = 1/p$ if $x = t$
 * (return), $1$ if $x$ is a neighbour of $t$, and $1/q$ otherwise (move outward). With $p = q = 1$ the walk is the
 * uniform random walk of DeepWalk (Perozzi, Al-Rfou and Skiena, 2014, Algorithm 1: walks per node in a shuffled node
 * order). The search bias is Grover and Leskovec (2016), "node2vec: scalable feature learning for networks", KDD,
 * §3.2.2. `walkCooccurrence` counts node pairs within a window of each other on the walks, both directions.
 * `skipGram` fits SGNS to the (centre, context) pairs of the walks by plain stochastic gradient descent with a
 * linearly decaying step, as word2vec does: each observed pair is a positive and $k$ nodes drawn from the
 * walk-frequency distribution raised to the power 0.75 are negatives (Mikolov et al., 2013, §2.2). Its optimum is the
 * shifted PMI of the counts (Levy and Goldberg, 2014), which `netmfMatrix` gives in closed form.
 *
 * Differences from the papers: the window is uniform (every context within `window` positions counts once) instead of
 * word2vec's sampled window; DeepWalk used hierarchical softmax, node2vec and this file negative sampling; transition
 * probabilities are computed on the fly instead of from node2vec's precomputed alias tables.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { aliasSample, aliasTable, child, permutation, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { adjacency, type Graph } from 'aifn-compute/graph'

/** Options of `node2vecWalks`. */
export interface WalkOptions {
  /** Walks started from every node (default 10). */
  readonly walksPerNode?: number
  /** Nodes per walk, the start included (default 40). */
  readonly length?: number
  /** Return parameter $p > 0$ (default 1): a small $p$ makes the walk step back often. */
  readonly p?: number
  /** In-out parameter $q > 0$ (default 1): $q < 1$ pushes the walk outward, $q > 1$ keeps it near its last node. */
  readonly q?: number
}

/** The neighbours of a node and the probability of moving to each, as `node2vecTransition` returns them. */
export interface WalkTransition {
  /** The neighbours, in adjacency order. */
  readonly to: number[]
  /** The probability of moving to each neighbour; sums to 1 (both lists are empty for a node with no neighbours). */
  readonly probability: number[]
}

/** A node's out-neighbours, their edge weights, and the same neighbours as a set for the $\alpha_{pq}$ test. */
type Neighbours = { to: Int32Array; weight: Float64Array; set: Set<number> }

/**
 * The neighbour lists of every node, from the graph's adjacency (undirected edges both ways).
 *
 * @param g The graph.
 * @returns One entry per node: the neighbours in adjacency order, their weights and the set of them.
 */
function neighbourLists(g: Graph): Neighbours[] {
  return adjacency(g).map((arcs) => ({
    to: Int32Array.from(arcs.map((a) => a.to)),
    weight: Float64Array.from(arcs.map((a) => a.weight)),
    set: new Set(arcs.map((a) => a.to)),
  }))
}

/**
 * The unnormalised node2vec weights $w_{vx}\,\alpha_{pq}(t, x)$ of the moves out of `current`.
 *
 * @param nb The neighbour lists of every node.
 * @param previous The node $t$ the walker came from, or null on the first step ($\alpha = 1$ for every move).
 * @param current The node $v$ the walker is at.
 * @param p The return parameter.
 * @param q The in-out parameter.
 * @returns One weight per neighbour of `current`, in adjacency order.
 */
function biasedWeights(nb: Neighbours[], previous: number | null, current: number, p: number, q: number) {
  const here = nb[current]
  const w = new Float64Array(here.to.length)
  for (let k = 0; k < here.to.length; k++) {
    const x = here.to[k]
    const alpha = previous === null ? 1 : x === previous ? 1 / p : nb[previous].set.has(x) ? 1 : 1 / q
    w[k] = here.weight[k] * alpha
  }
  return w
}

/**
 * The node2vec transition out of `current` for a walker that arrived from `previous` (null for the first step, which
 * is proportional to the edge weights alone). A node with no neighbours gives empty lists. Throws `DomainError` for
 * $p$ or $q$ not positive, a node outside the graph, or neighbours whose weights sum to zero (no move is defined).
 *
 * @param g The graph (undirected for the usual walk; a directed graph walks along out-edges).
 * @param previous The node the walker came from, or null.
 * @param current The node the walker is at.
 * @param p The return parameter, positive.
 * @param q The in-out parameter, positive.
 * @returns The neighbours of `current` in adjacency order and the probability of moving to each.
 *
 * @example On a triangle with a tail, after arriving at node 1 from node 0
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [1, 2], [0, 2], [2, 3]], { directed: false })
 * print('uniform (p = q = 1):', node2vecTransition(g, 0, 2, 1, 1))
 * print('outward (q = 0.25):', node2vecTransition(g, 0, 2, 1, 0.25))
 * print('return (p = 0.25):', node2vecTransition(g, 0, 2, 0.25, 1))
 */
export function node2vecTransition(
  g: Graph,
  previous: number | null,
  current: number,
  p: number,
  q: number,
): WalkTransition {
  if (!(p > 0 && q > 0)) throw new DomainError('node2vecTransition', 'node2vecTransition: p and q must be positive')
  const nb = neighbourLists(g)
  if (!(Number.isInteger(current) && current >= 0 && current < nb.length))
    throw new DomainError('node2vecTransition', 'node2vecTransition: current must be a node of the graph')
  if (previous !== null && !(Number.isInteger(previous) && previous >= 0 && previous < nb.length))
    throw new DomainError('node2vecTransition', 'node2vecTransition: previous must be a node of the graph or null')
  const w = biasedWeights(nb, previous, current, p, q)
  const total = w.reduce((a, b) => a + b, 0)
  if (w.length > 0 && !(total > 0))
    throw new DomainError('node2vecTransition', `node2vecTransition: the moves out of node ${current} have zero weight`)
  return { to: Array.from(nb[current].to), probability: Array.from(w, (v) => v / total) }
}

/**
 * Truncated node2vec walks. Pass $r = 0, 1, \dots$ starts one walk at every node, in the order of a fresh permutation
 * (child stream `'order'`, $r$); each step uses one uniform draw from the child stream `'steps'`. A walk that reaches a
 * node with no out-neighbours, or whose moves all have zero weight, stops early. Throws `DomainError` for $p$ or $q$
 * not positive or a length that is not a positive integer.
 *
 * @param s The random stream.
 * @param g The graph.
 * @param options Walks per node, walk length and the bias parameters $p$ and $q$.
 * @returns The walks, each an array of node indices starting at its start node; walk $rn + i$ is pass $r$'s $i$-th.
 *
 * @example Short DeepWalk walks on a path
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(5, [[0, 1], [1, 2], [2, 3], [3, 4]], { directed: false })
 * const walks = node2vecWalks(stream(1), g, { walksPerNode: 1, length: 6 })
 * for (const w of walks) print(w)
 */
export function node2vecWalks(s: Stream, g: Graph, options: WalkOptions = {}): number[][] {
  const { walksPerNode = 10, length = 40, p = 1, q = 1 } = options
  if (!(p > 0 && q > 0)) throw new DomainError('node2vecWalks', 'node2vecWalks: p and q must be positive')
  if (!(Number.isInteger(length) && length >= 1))
    throw new DomainError('node2vecWalks', 'node2vecWalks: length must be a positive integer')
  const nb = neighbourLists(g)
  const n = g.nodes
  const draws = toFlat(uniform(child(s, 'steps'), 0, 1, { shape: [Math.max(1, n * walksPerNode * (length - 1))] }))
  let u = 0
  const walks: number[][] = []
  for (let r = 0; r < walksPerNode; r++) {
    const order = toFlat(permutation(child(s, 'order', r), n))
    for (const start of order) {
      const walk = [start]
      let previous: number | null = null
      let current = start
      while (walk.length < length) {
        if (nb[current].to.length === 0) break
        const w = biasedWeights(nb, previous, current, p, q)
        let total = 0
        for (const v of w) total += v
        if (!(total > 0)) break
        let target = draws[u++] * total
        let k = 0
        while (k < w.length - 1 && target >= w[k]) target -= w[k++]
        previous = current
        current = nb[current].to[k]
        walk.push(current)
      }
      walks.push(walk)
    }
  }
  return walks
}

/**
 * Window co-occurrence counts on walks: for every position $j$ of every walk and every $r = 1, \dots,$ `window`, the
 * pair (node at $j$, node at $j + r$) adds 1 to both $\#(w, c)$ and $\#(c, w)$. The count matrix is symmetric and its
 * total is the number of (centre, context) pairs that `skipGram` trains on; `ppmi(counts, { shift: b })` of
 * `aifn-compute/text/cooccurrence` turns it into the shifted PMI that approaches `netmfMatrix`. Throws `DomainError`
 * for a window that is not a positive integer.
 *
 * @param walks The walks, each a list of node indices in $0, \dots, n - 1$.
 * @param n The number of nodes.
 * @param window The window $T \ge 1$.
 * @returns The $n \times n$ count matrix.
 *
 * @example Counts of one walk with window 1
 * print(walkCooccurrence([[0, 1, 2, 1]], 3, 1))
 */
export function walkCooccurrence(walks: readonly (readonly number[])[], n: number, window: number): Tensor {
  if (!(Number.isInteger(window) && window >= 1))
    throw new DomainError('walkCooccurrence', 'walkCooccurrence: window must be a positive integer')
  const c = new Float64Array(n * n)
  for (const walk of walks)
    for (let j = 0; j < walk.length; j++)
      for (let r = 1; r <= window && j + r < walk.length; r++) {
        c[walk[j] * n + walk[j + r]] += 1
        c[walk[j + r] * n + walk[j]] += 1
      }
  return fromData(c, [n, n])
}

/** Options of `skipGram`. */
export interface SkipGramOptions {
  /** Embedding dimension $d$ (default 16). */
  readonly dim?: number
  /** Context window $T$ on each side (default 5). */
  readonly window?: number
  /** Negatives $k$ per positive pair (default 5). */
  readonly negatives?: number
  /** Passes over the pairs (default 1). */
  readonly epochs?: number
  /** Initial step size, decayed linearly to $10^{-4}$ of itself over all passes (default 0.025, word2vec's). */
  readonly rate?: number
  /** Exponent of the walk frequencies in the noise distribution (default 0.75). */
  readonly noisePower?: number
}

/** The result of `skipGram`. */
export interface SkipGramResult {
  /** Centre ("input") vectors, $n \times d$: the embedding. */
  readonly input: Tensor
  /** Context ("output") vectors, $n \times d$. */
  readonly output: Tensor
  /**
   * Mean SGNS loss per pair in each pass:
   * $-\log\sigma(\uvec_c^\top\vvec_w) - \sum_{\text{neg}}\log\sigma(-\uvec_{c'}^\top\vvec_w)$.
   */
  readonly loss: number[]
  /** The number of (centre, context) pairs per pass. */
  readonly pairs: number
}

/**
 * Skip-gram with negative sampling on random walks (see the file comment). Input vectors start uniform in
 * $\pm 0.5/d$ (child stream `'init'`), output vectors at zero, as in word2vec; negatives come from the child stream
 * `'negatives'`, pass by pass. A negative equal to the positive context is skipped. Nodes that never appear on a walk
 * keep their initial vectors. Plain data in and out, so a worker can run it (about $10^5$ pairs times $k + 1$ times
 * $d$ operations per pass on a 34-node graph). Throws `DomainError` for a dimension that is not a positive integer or
 * a negative count that is not a non-negative integer.
 *
 * @param s The random stream.
 * @param walks The walks (from `node2vecWalks`).
 * @param n The number of nodes.
 * @param options Dimension, window, negatives, passes, step size and noise exponent.
 * @returns The input and output vectors, the loss of each pass and the number of pairs.
 *
 * @example The loss falls over passes on walks of a ring
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], { directed: false })
 * const walks = node2vecWalks(stream(1), g, { walksPerNode: 5, length: 20 })
 * const fit = skipGram(stream(2), walks, 6, { dim: 4, window: 2, epochs: 5, rate: 0.05 })
 * print('pairs per pass:', fit.pairs)
 * print('mean loss per pass:', fit.loss)
 */
export function skipGram(
  s: Stream,
  walks: readonly (readonly number[])[],
  n: number,
  options: SkipGramOptions = {},
): SkipGramResult {
  const { dim = 16, window = 5, negatives = 5, epochs = 1, rate = 0.025, noisePower = 0.75 } = options
  if (!(Number.isInteger(dim) && dim >= 1))
    throw new DomainError('skipGram', 'skipGram: dim must be a positive integer')
  if (!(Number.isInteger(negatives) && negatives >= 0))
    throw new DomainError('skipGram', 'skipGram: negatives must be a non-negative integer')
  const centres: number[] = []
  const contexts: number[] = []
  const frequency = new Float64Array(n)
  for (const walk of walks) {
    for (const v of walk) frequency[v] += 1
    for (let j = 0; j < walk.length; j++)
      for (let r = 1; r <= window; r++) {
        if (j + r < walk.length) {
          centres.push(walk[j])
          contexts.push(walk[j + r])
        }
        if (j - r >= 0) {
          centres.push(walk[j])
          contexts.push(walk[j - r])
        }
      }
  }
  const m = centres.length
  const table = aliasTable(Array.from(frequency, (f) => Math.pow(f, noisePower)))
  const init = toFlat(uniform(child(s, 'init'), -0.5 / dim, 0.5 / dim, { shape: [n * dim] }))
  const input = Float64Array.from(init)
  const output = new Float64Array(n * dim)
  const grad = new Float64Array(dim)
  const loss: number[] = []
  const total = Math.max(1, m * epochs)
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
  for (let e = 0; e < epochs; e++) {
    const noise = negatives > 0 ? toFlat(aliasSample(child(s, 'negatives', e), table, { shape: [m * negatives] })) : []
    let sum = 0
    for (let i = 0; i < m; i++) {
      const lr = rate * Math.max(1e-4, 1 - (e * m + i) / total)
      const w = centres[i] * dim
      grad.fill(0)
      for (let k = 0; k <= negatives; k++) {
        const c = k === 0 ? contexts[i] : noise[i * negatives + k - 1]
        if (k > 0 && c === contexts[i]) continue
        const label = k === 0 ? 1 : 0
        const o = c * dim
        let f = 0
        for (let a = 0; a < dim; a++) f += input[w + a] * output[o + a]
        const sf = sigmoid(f)
        sum -= Math.log(Math.max(1e-12, label === 1 ? sf : 1 - sf))
        const g = (label - sf) * lr
        for (let a = 0; a < dim; a++) {
          grad[a] += g * output[o + a]
          output[o + a] += g * input[w + a]
        }
      }
      for (let a = 0; a < dim; a++) input[w + a] += grad[a]
    }
    loss.push(m ? sum / m : NaN)
  }
  return { input: fromData(input, [n, dim]), output: fromData(output, [n, dim]), loss, pairs: m }
}
