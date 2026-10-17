/**
 * The computation graph of an $L$-layer message-passing network for a minibatch of target nodes: the nodes whose
 * vectors the forward pass needs at each depth, with the full neighbourhood (every node within $k$ hops of a target,
 * following the messages backwards) or with GraphSAGE neighbour sampling (a fresh sample of at most $S_k$ incoming
 * edges per node at depth $k$, drawn by `sampleNeighbours`), as in Hamilton, Ying and Leskovec (2017), "Inductive
 * representation learning on large graphs", NeurIPS, Algorithm 2 (the sampled sets $\mathcal{B}^k$ built from the
 * targets outwards, one sample per depth). The growth of the full set with depth is the neighbourhood explosion that
 * sampling bounds.
 *
 * Both take plain inputs and return plain data, so they run in a worker; on a 2000-node graph with three layers they
 * take a few tens of milliseconds.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { kHopNeighbourhood } from 'aifn-compute/graph/traversal'
import { sampleNeighbours } from './layers'

/** Sizes of a computation graph by depth. */
export interface ComputationGraph {
  /** `touched[k]` is the number of distinct nodes needed down to depth $k$ ($k = 0$: the targets), $L + 1$ entries. */
  touched: number[]
  /** The distinct nodes needed down to depth $L$ (int32), the targets first, then by depth. */
  nodes: Tensor
}

/**
 * The full computation graph: every node within $k$ hops of a target along the message edges into it, for
 * $k = 0, \dots, L$ (`kHopNeighbourhood` backwards). Throws `DomainError` for a target outside the graph or a
 * negative number of layers.
 *
 * @param g The graph (an undirected edge is followed both ways).
 * @param targets The minibatch of target nodes.
 * @param layers The number of layers $L$.
 * @returns The number of nodes needed down to each depth, and those nodes.
 *
 * @example From node 3 of a triangle with a tail, two layers need every node
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [0, 2], [1, 2], [2, 3]], { directed: false })
 * print('touched:', computationGraph(g, [3], 2).touched)
 */
export function computationGraph(g: Graph, targets: readonly number[], layers: Size): ComputationGraph {
  const r = kHopNeighbourhood(g, targets, layers, { direction: 'in' })
  return { touched: r.sizes, nodes: r.nodes }
}

/** A sampled computation graph: its sizes and the neighbour sample of each depth. */
export interface SampledComputationGraph extends ComputationGraph {
  /** `samples[k - 1]` is the sampled graph used at depth $k$: directed, at most $S_k$ incoming edges per node. */
  samples: Graph[]
}

/**
 * The sampled computation graph of GraphSAGE: at depth $k$ every node draws at most $S_k$ of its incoming neighbours,
 * a fresh sample per depth, and the needed set grows by the sampled neighbours of the nodes already needed. Throws as
 * `sampleNeighbours` does for a sample size that is not a positive integer, and as `computationGraph` for a bad
 * target.
 *
 * @param g The graph.
 * @param targets The minibatch of target nodes.
 * @param fanouts The sample sizes $S_1, \dots, S_L$, from the targets outwards ($S_1$ for the targets' own neighbours).
 * @param s The random stream; depth $k$ draws from `child(s, 'depth', k)`.
 * @returns The number of nodes needed down to each depth, those nodes, and the sampled graph of each depth.
 *
 * @example One neighbour per node per depth
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [0, 2], [1, 2], [2, 3]], { directed: false })
 * print('touched:', sampledComputationGraph(g, [3], [1, 1], stream(0)).touched)
 */
export function sampledComputationGraph(
  g: Graph,
  targets: readonly number[],
  fanouts: readonly number[],
  s: Stream,
): SampledComputationGraph {
  const samples = fanouts.map((size, k) => sampleNeighbours(g, size, child(s, 'depth', k + 1)))
  for (const v of targets)
    if (!(Number.isInteger(v) && v >= 0 && v < g.nodes))
      throw new DomainError('sampledComputationGraph', `sampledComputationGraph: target ${v} is outside the graph`)
  const order = [...new Set(targets)]
  const seen = new Set(order)
  const touched = [seen.size]
  for (const sample of samples) {
    // Every node needed so far brings in its sampled in-neighbours (the sample is directed, so these are its edges in).
    const incoming: number[][] = Array.from({ length: g.nodes }, () => [])
    for (const e of sample.edges) incoming[e.to].push(e.from)
    const next: number[] = []
    for (const v of order)
      for (const u of incoming[v])
        if (!seen.has(u)) {
          seen.add(u)
          next.push(u)
        }
    order.push(...next)
    touched.push(seen.size)
  }
  return { touched, nodes: fromData(Int32Array.from(order)), samples }
}
