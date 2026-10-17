/**
 * Neighbourhood scores of node pairs, the classical link-prediction heuristics (Liben-Nowell and Kleinberg 2007,
 * "The link-prediction problem for social networks", JASIST 58(7)): common neighbours
 * $\lvert\Ncal(u) \cap \Ncal(v)\rvert$ and Adamic-Adar $\sum_{w \in \Ncal(u) \cap \Ncal(v)} 1 / \log \deg(w)$
 * (Adamic and Adar 2003, "Friends and neighbors on the Web", Social Networks 25(3)), as networkx's `common_neighbors`
 * and `adamic_adar_index`. The graph is read as undirected and simple: self-loops are ignored and a pair joined twice
 * is one neighbour.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from '../graph'

/** A pair of nodes $(u, v)$. */
export type NodePair = readonly [number, number]

/** The result of `pairHeuristics`: one score of each kind per pair, in the order the pairs were given. */
export interface PairHeuristics {
  /** The number of common neighbours of each pair, $\lvert\Ncal(u) \cap \Ncal(v)\rvert$ (float64). */
  commonNeighbours: Tensor
  /**
   * The Adamic-Adar index of each pair: common neighbours weighted by $1 / \log \deg(w)$, so that a rare shared
   * neighbour counts more than a hub (float64).
   */
  adamicAdar: Tensor
}

/**
 * Common-neighbour and Adamic-Adar scores of node pairs (see the file comment). A common neighbour has degree at
 * least 2, so $\log \deg(w) > 0$. Throws `DomainError` for a pair with an end outside the graph.
 *
 * @param g The graph, read as undirected; usually the training graph of a link-prediction split.
 * @param pairs The node pairs to score.
 * @returns The two scores of every pair.
 *
 * @example A triangle 0-1-2 with a tail 2-3: pairs (0, 3) and (1, 3) share node 2, of degree 3
 * const edges = [[0, 1], [0, 2], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges, directed: false }
 * const h = pairHeuristics(g, [[0, 3], [1, 3]])
 * print('common neighbours:', h.commonNeighbours)
 * print('Adamic-Adar:', h.adamicAdar, ' 1 / log 3 =', 1 / Math.log(3))
 */
export function pairHeuristics(g: Graph, pairs: readonly NodePair[]): PairHeuristics {
  const near: Set<number>[] = Array.from({ length: g.nodes }, () => new Set<number>())
  for (const e of g.edges)
    if (e.from !== e.to) {
      near[e.from].add(e.to)
      near[e.to].add(e.from)
    }
  const common = new Float64Array(pairs.length)
  const adamic = new Float64Array(pairs.length)
  pairs.forEach(([u, v], k) => {
    if (!(near[u] && near[v]))
      throw new DomainError('pairHeuristics', `pairHeuristics: pair (${u}, ${v}) is outside 0…${g.nodes - 1}`)
    for (const w of near[u])
      if (near[v].has(w)) {
        common[k]++
        adamic[k] += 1 / Math.log(near[w].size)
      }
  })
  return { commonNeighbours: fromData(common), adamicAdar: fromData(adamic) }
}
