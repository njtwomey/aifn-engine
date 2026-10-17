/**
 * The similarity graphs of graph-based semi-supervised learning, built from a point cloud $\Xmat$ ($n \times d$) in two
 * choices: which pairs are joined (the neighbourhood rule) and what weight each edge carries.
 *
 * Rules: the full graph; the symmetrised $k$-nearest-neighbour graph ($i \sim j$ when either is among the other's $k$
 * nearest); the mutual $k$-NN graph (both); the $\varepsilon$-ball graph
 * ($\lVert \xvec_i - \xvec_j \rVert \le \varepsilon$), all from von Luxburg (2007), "A tutorial on spectral
 * clustering", §2.2; and a greedy $b$-matching, in which every node has at most $b$ edges (Jebara, Wang and Chang,
 * 2009, §3, solved greedily). Weights: 1 per edge, or the heat kernel
 * $\exp(-\lVert \xvec_i - \xvec_j \rVert^2 / 2\sigma^2)$. The joins are those of `kNearestNeighbourGraph`,
 * `epsilonBallGraph` and `bMatchingGraph` of `aifn-compute/graph/structures` (ties to the smaller index), and the
 * weights those of `graphAffinity`; `pointAffinity` is the same full and $k$-NN heat graphs as a matrix only. The
 * Laplacian $\Lmat = \Dmat - \Wmat$ of the affinity is `laplacian` of `aifn-compute/graph/matrices`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { bMatchingGraph, epsilonBallGraph, kNearestNeighbourGraph } from 'aifn-compute/graph/structures'
import { fromEdges, weightOf, type Graph } from '../graph'
import { graphAffinity } from './labels'

/** Which pairs of points are joined. */
export type NeighbourhoodRule =
  /** Every pair. */
  | { kind: 'full' }
  /** $i$ and $j$ when either is among the other's $k$ nearest. */
  | { kind: 'knn'; k: Size }
  /** $i$ and $j$ when both are among the other's $k$ nearest. */
  | { kind: 'mutual-knn'; k: Size }
  /** Pairs at a distance of at most $\varepsilon$. */
  | { kind: 'epsilon'; epsilon: number }
  /** The shortest pairs in order while both ends have fewer than $b$ edges. */
  | { kind: 'b-matching'; b: Size }

/** The weight of an edge: 1, or a heat kernel of the distance with bandwidth $\sigma > 0$. */
export type EdgeWeighting = { kind: 'binary' } | { kind: 'heat'; sigma: number }

/** One undirected edge, $i < j$, with the distance between its ends and its weight. */
export interface GraphEdge {
  /** The smaller end $i$. */
  readonly from: number
  /** The larger end $j$. */
  readonly to: number
  /** The distance $\lVert \xvec_i - \xvec_j \rVert$. */
  readonly distance: number
  /** The edge's weight: 1, or the heat kernel of the distance. */
  readonly weight: number
}

/** A neighbourhood graph: its affinity matrix, its edges, each node's degree and the graph of distances. */
export interface NeighbourhoodGraph {
  /** $\Wmat$, $n \times n$, symmetric with a zero diagonal. */
  readonly affinity: Tensor
  /** The undirected edges, each once with `from < to`, in row-major order. */
  readonly edges: readonly GraphEdge[]
  /** The number of edges at each node. */
  readonly counts: readonly number[]
  /** The undirected graph whose edge weights are the distances, as the builders of `graph/structures` return it. */
  readonly graph: Graph
}

/**
 * Build a neighbourhood graph on a point cloud; see the file comment. A $k$ larger than $n - 1$ is capped at
 * $n - 1$. Throws `DomainError` for a $k$ or $b$ that is not a positive integer, a negative $\varepsilon$ or a
 * $\sigma$ that is not positive.
 *
 * @param x The points, one per row ($n \times d$).
 * @param rule The neighbourhood rule.
 * @param weighting The edge weight (default the heat kernel with $\sigma = 1$).
 * @returns The affinity matrix, the edge list, the edge count of each node, and the graph of distances.
 *
 * @example The rules on the points 0, 1 and 3 of a line
 * const x = [[0], [1], [3]]
 * const pairs = (rule) => neighbourhoodGraph(x, rule, { kind: 'binary' }).edges.map((e) => [e.from, e.to])
 * print('1-NN:', pairs({ kind: 'knn', k: 1 }), ' mutual:', pairs({ kind: 'mutual-knn', k: 1 }))
 * print('epsilon 1.5:', pairs({ kind: 'epsilon', epsilon: 1.5 }), ' b = 1:', pairs({ kind: 'b-matching', b: 1 }))
 *
 * @example Heat weights on the full graph
 * print(neighbourhoodGraph([[0], [1], [3]], { kind: 'full' }, { kind: 'heat', sigma: 1 }).affinity)
 */
export function neighbourhoodGraph(
  x: MatrixLike,
  rule: NeighbourhoodRule,
  weighting: EdgeWeighting = { kind: 'heat', sigma: 1 },
): NeighbourhoodGraph {
  const where = 'neighbourhoodGraph'
  if (weighting.kind === 'heat' && !(weighting.sigma > 0))
    throw new DomainError(where, `${where}: sigma must be positive`)
  const n = isTensor(x) ? x.shape[0] : x.length
  let graph: Graph
  if (rule.kind === 'knn' || rule.kind === 'mutual-knn') {
    if (!(Number.isInteger(rule.k) && rule.k >= 1))
      throw new DomainError(where, `${where}: k must be a positive integer`)
    const k = Math.min(rule.k, n - 1)
    graph =
      k >= 1
        ? kNearestNeighbourGraph(x, k, { mode: rule.kind === 'knn' ? 'symmetric' : 'mutual' })
        : fromEdges(n, [], { directed: false })
  } else if (rule.kind === 'epsilon') {
    if (!(rule.epsilon >= 0)) throw new DomainError(where, `${where}: epsilon must be non-negative`)
    graph = epsilonBallGraph(x, rule.epsilon)
  } else if (rule.kind === 'b-matching') {
    if (!(Number.isInteger(rule.b) && rule.b >= 1))
      throw new DomainError(where, `${where}: b must be a positive integer`)
    graph = bMatchingGraph(x, rule.b)
  } else graph = epsilonBallGraph(x, Infinity)
  const kernel = weighting.kind === 'binary' ? ({ kind: 'connectivity' } as const) : weighting
  const weight = (d: number) => (weighting.kind === 'binary' ? 1 : Math.exp(-(d * d) / (2 * weighting.sigma ** 2)))
  const counts = new Array<number>(graph.nodes).fill(0)
  const edges = graph.edges.map((e) => {
    counts[e.from]++
    counts[e.to]++
    const distance = weightOf(e)
    return { from: e.from, to: e.to, distance, weight: weight(distance) }
  })
  return { affinity: graphAffinity(graph, kernel), edges, counts, graph }
}
