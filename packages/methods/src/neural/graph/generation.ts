/**
 * The sequence form of a graph that autoregressive graph generators learn (You, Ying, Ren, Hamilton and Leskovec
 * 2018, "GraphRNN: generating realistic graphs with deep auto-regressive models", ICML). Under a node ordering $\pi$,
 * node $\pi_i$ is added at step $i$ with its edges to the nodes already placed, recorded as the vector
 * $S_i \in \{0, 1\}^i$ with $S_{ij} = 1$ when $\pi_i$ and $\pi_j$ are adjacent ($j < i$). The sequence
 * $(S_1, \dots, S_{n-1})$ is the strictly lower triangle of the reordered adjacency matrix, row by row. Under a
 * breadth-first ordering every edge joins a node to one placed at most $M$ steps earlier, $M$ bounded by the size of
 * two consecutive breadth-first layers, so a model need only predict the last $M$ entries of each row.
 *
 * A breadth-first ordering from a root is `breadthFirstSearch(g, [root, ...others]).order`
 * (`aifn-compute/graph/traversal`), and a random one `permutation(s, n)`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'

/**
 * The adjacency sequence of a graph under an ordering: row $i$ (for $i = 0, \dots, n - 1$) lists, for each earlier
 * position $j < i$, whether the nodes at positions $i$ and $j$ are adjacent; row 0 is empty. Edges are read as
 * undirected and self-loops are left out. Throws `DomainError` unless the ordering is a permutation of the nodes.
 *
 * @param g The graph.
 * @param order The node at each position, a permutation of $0, \dots, n - 1$ (an array or an int32 vector).
 * @returns The $n$ rows, row $i$ of length $i$, of zeros and ones.
 *
 * @example The 4-cycle under its breadth-first ordering from node 0
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * import { breadthFirstSearch } from 'aifn-compute/graph/traversal'
 * const g = cycleGraph(4)
 * const order = breadthFirstSearch(g, [0, 1, 2, 3]).order
 * print('order:', order)
 * print('rows:', adjacencySequence(g, order))
 */
export function adjacencySequence(g: Graph, order: readonly number[] | Tensor): number[][] {
  const pi = isTensor(order) ? Array.from(toFlat(order)) : [...order]
  const position = new Map(pi.map((v, i) => [v, i]))
  const valid = pi.every((v) => Number.isInteger(v) && v >= 0 && v < g.nodes)
  if (pi.length !== g.nodes || position.size !== g.nodes || !valid)
    throw new DomainError('adjacencySequence', 'adjacencySequence: the ordering must be a permutation of the nodes')
  const rows = pi.map((_, i) => new Array<number>(i).fill(0))
  for (const e of g.edges) {
    if (e.from === e.to) continue
    const a = position.get(e.from)!
    const b = position.get(e.to)!
    if (a > b) rows[a][b] = 1
    else rows[b][a] = 1
  }
  return rows
}

/**
 * The bandwidth of an adjacency sequence: the largest step back any edge reaches, $\max (i - j)$ over the ones of row
 * $i$ at position $j$; 0 for a graph without edges.
 *
 * @param rows The adjacency sequence, as `adjacencySequence` returns it.
 * @returns The bandwidth $M$.
 *
 * @example A path in order has bandwidth 1; reversing two nodes widens it
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * print('in order:', sequenceBandwidth(adjacencySequence(chainGraph(4), [0, 1, 2, 3])))
 * print('shuffled:', sequenceBandwidth(adjacencySequence(chainGraph(4), [0, 2, 3, 1])))
 */
export function sequenceBandwidth(rows: readonly (readonly number[])[]): number {
  let m = 0
  rows.forEach((row, i) =>
    row.forEach((x, j) => {
      if (x) m = Math.max(m, i - j)
    }),
  )
  return m
}
