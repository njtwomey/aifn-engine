/**
 * Hop distances and hop neighbourhoods by breadth-first search: all-pairs hop counts (`hopDistances`, one search per
 * source, cheaper than Floyd-Warshall on a unit-weight copy and with no weights to strip) and the nodes within $k$ hops
 * of a set of sources (`kHopNeighbourhood`), the receptive field of a $k$-layer message-passing network and the ego
 * network of a node.
 *
 * Edge weights are ignored. Neighbours are visited in `adjacency` order, so the order of discovery is reproducible.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { adjacency, reverse, type Graph } from '../graph'

/**
 * The hop count between every pair of nodes: entry $(i, j)$ is the number of edges on a shortest path from $i$ to $j$
 * (directed edges followed forwards only), Infinity where there is none, and 0 on the diagonal. One breadth-first
 * search per source, $O(V(V + E))$.
 *
 * @param g The graph.
 * @returns A float64 $V \times V$ tensor of hop counts.
 *
 * @example A path 0 - 1 - 2 - 3 and an isolated node 4
 * const edges = [[0, 1], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * print('hops =', hopDistances(g))
 */
export function hopDistances(g: Graph): Tensor {
  const lists = adjacency(g)
  const V = g.nodes
  const out = new Float64Array(V * V).fill(Infinity)
  const queue = new Int32Array(V)
  for (let source = 0; source < V; source++) {
    const row = source * V
    out[row + source] = 0
    let head = 0
    let tail = 0
    queue[tail++] = source
    while (head < tail) {
      const v = queue[head++]
      for (const { to } of lists[v])
        if (out[row + to] === Infinity) {
          out[row + to] = out[row + v] + 1
          queue[tail++] = to
        }
    }
  }
  return fromData(out, [V, V])
}

/** Options of `kHopNeighbourhood`. */
export interface HopOptions {
  /**
   * `'out'` (default) follows edges forwards, from a node to its out-neighbours; `'in'` follows them backwards, to the
   * nodes with an edge into it, which are the nodes whose messages reach it. The same for an undirected graph.
   */
  direction?: 'out' | 'in'
}

/** The result of `kHopNeighbourhood`. */
export interface HopNeighbourhood {
  /**
   * The nodes within $k$ hops (int32), in order of discovery: the sources first (repeats dropped), then the nodes at 1
   * hop, then 2, and so on, each layer in breadth-first order.
   */
  nodes: Tensor
  /** The hop count from the nearest source of each of the $V$ nodes (int32), $-1$ beyond $k$ hops. */
  depth: Tensor
  /** `sizes[j]` is the number of nodes within $j$ hops, for $j = 0, \dots, k$ ($k + 1$ entries). */
  sizes: number[]
}

/**
 * The nodes within $k$ hops of any of `sources`, by a multi-source breadth-first search: the distances are to the
 * nearest source, unlike `breadthFirstSearch` with several sources, which tries them as successive roots. With
 * `direction: 'in'` it is the computation graph of a $k$-layer message-passing network for the sources (every node
 * whose features reach them). Throws `DomainError` for a source outside the graph or a negative or fractional $k$.
 *
 * @param g The graph.
 * @param sources The source nodes: a node, an array of nodes or an int32 vector.
 * @param k The number of hops $k$.
 * @param options The direction to follow edges in (`'out'` by default).
 * @returns The nodes within $k$ hops in order of discovery, every node's hop count, and how many nodes lie within
 *   each number of hops.
 *
 * @example From node 3 of a triangle with a tail
 * const edges = [[0, 1], [0, 2], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges, directed: false }
 * const { nodes, depth, sizes } = kHopNeighbourhood(g, 3, 2)
 * print('nodes:', nodes, ' depth:', depth, ' sizes:', sizes)
 *
 * @example On a directed path, forwards and backwards
 * const edges = [[0, 1], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * print('out of 1:', kHopNeighbourhood(g, [1], 1).nodes)
 * print('into 1:', kHopNeighbourhood(g, [1], 1, { direction: 'in' }).nodes)
 */
export function kHopNeighbourhood(
  g: Graph,
  sources: number | readonly number[] | Tensor,
  k: Size,
  options: HopOptions = {},
): HopNeighbourhood {
  const where = 'kHopNeighbourhood'
  if (!(Number.isInteger(k) && k >= 0)) throw new DomainError(where, `${where}: k = ${k} must be a count`)
  const lists = adjacency(options.direction === 'in' ? reverse(g) : g)
  const start = typeof sources === 'number' ? [sources] : isTensor(sources) ? Array.from(toFlat(sources)) : sources
  const depth = new Int32Array(g.nodes).fill(-1)
  const order: number[] = []
  for (const v of start) {
    if (!(Number.isInteger(v) && v >= 0 && v < g.nodes))
      throw new DomainError(where, `${where}: source ${v} is outside 0…${g.nodes - 1}`)
    if (depth[v] < 0) {
      depth[v] = 0
      order.push(v)
    }
  }
  const sizes = [order.length]
  let from = 0
  for (let hop = 1; hop <= k; hop++) {
    const to = order.length
    for (let i = from; i < to; i++)
      for (const arc of lists[order[i]])
        if (depth[arc.to] < 0) {
          depth[arc.to] = hop
          order.push(arc.to)
        }
    from = to
    sizes.push(order.length)
  }
  return { nodes: fromData(Int32Array.from(order)), depth: fromData(depth), sizes }
}
