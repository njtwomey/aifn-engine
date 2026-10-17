/**
 * The relational graph convolution of Schlichtkrull, Kipf, Bloem, van den Berg, Titov and Welling (2018), "Modeling
 * relational data with graph convolutional networks", ESWC, eq. 2 and 3 and §5.2: one weight matrix per relation, a
 * normalised sum of the transformed neighbour vectors per relation, and a self-connection,
 *
 * $$\hvec'_i = \Wmat_0^\top\hvec_i + \sum_r \frac{1}{c_{i,r}} \sum_{j \in \Ncal_i^r} \Wmat_r^\top \hvec_j,$$
 *
 * in the engine's row convention ($\Hmat\Wmat$, weights $F \times G$). Each relation is a directed graph on the shared
 * node indices, an edge $j \to i$ meaning that $i$ receives from $j$ under that relation. The normalisation is
 * `relation` ($c_{i,r} = \lvert\Ncal_i^r\rvert$, the paper's choice for entity classification), `node`
 * ($c_{i,r} = \sum_r \lvert\Ncal_i^r\rvert$, its choice for link prediction) or `none`. `basisWeights` builds every
 * $\Wmat_r = \sum_b a_{rb} \Vmat_b$ from $B$ shared bases (eq. 3); the block-diagonal decomposition (eq. 4) is not
 * included. Edge weights are ignored: each neighbour counts once, as in the paper. The activation is left to the
 * caller.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { add, div, matmul, maximum, mul, ones, type Value } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { propagate } from 'aifn-compute/graph/propagation'

/** One relation of an R-GCN: its directed edges on the shared node indices and its weight $\Wmat_r$. */
export interface RgcnRelation {
  /** The edges $j \to i$ of this relation ($i$ receives from $j$). */
  graph: Graph
  /** The relation's weight $\Wmat_r$, $F \times G$. */
  weight: Value
}

/**
 * How the messages of a relation are scaled: by $1/\lvert\Ncal_i^r\rvert$ (`relation`), by
 * $1/\sum_r \lvert\Ncal_i^r\rvert$ (`node`), or not at all (`none`).
 */
export type RgcnNormalisation = 'relation' | 'node' | 'none'

/**
 * Every relation's contribution $\frac{1}{c_{i,r}} \sum_{j \in \Ncal_i^r} \Wmat_r^\top\hvec_j$, one $V \times G$
 * matrix per relation; a node with no neighbour under a relation gets a zero row. Differentiable in `h` and the
 * weights. Throws `DomainError` when the relations' graphs do not share one node count.
 *
 * @param h The node features $\Hmat$, $V \times F$.
 * @param relations The relations, each with its graph and weight.
 * @param normalisation The scaling of each relation's sum (default `relation`).
 * @returns The per-relation terms, in the order of `relations`.
 *
 * @example Node 2 receives from 0 under one relation and from 0 and 1 under another
 * const g = (edges) => ({ kind: 'graph', nodes: 3, edges: edges.map(([from, to]) => ({ from, to })) })
 * const I = tensor([[1, 0], [0, 1]])
 * const relations = [{ graph: g([[0, 2]]), weight: I }, { graph: g([[0, 2], [1, 2]]), weight: I }]
 * const h = tensor([[1, 0], [0, 1], [0, 0]])
 * for (const t of rgcnTerms(h, relations)) print(toRows(t)[2])
 */
export function rgcnTerms(
  h: Value,
  relations: readonly RgcnRelation[],
  normalisation: RgcnNormalisation = 'relation',
): Value[] {
  if (relations.some((r) => r.graph.nodes !== relations[0].graph.nodes))
    throw new DomainError('rgcnTerms', 'rgcnTerms: every relation must be a graph on the same nodes')
  const plain = { message: (source: Value) => source }
  if (normalisation === 'relation')
    return relations.map((r) => propagate(r.graph, matmul(h, r.weight), { ...plain, aggregate: 'mean' }))
  const sums = relations.map((r) => propagate(r.graph, matmul(h, r.weight), { ...plain, aggregate: 'sum' }))
  if (normalisation === 'none' || relations.length === 0) return sums
  const nodes = relations[0].graph.nodes
  const counts = relations
    .map((r) => propagate(r.graph, ones([nodes, 1]), { ...plain, aggregate: 'sum' }))
    .reduce((a, b) => add(a, b))
  const scale = maximum(counts, 1)
  return sums.map((s) => div(s, scale))
}

/**
 * One relational graph convolution, before the activation: the self-connection $\Hmat\Wmat_0$ plus the sum of the
 * relation terms of `rgcnTerms`. Differentiable in `h` and every weight.
 *
 * @param h The node features $\Hmat$, $V \times F$.
 * @param relations The relations.
 * @param selfWeight The self-connection $\Wmat_0$, $F \times G$.
 * @param normalisation The scaling of each relation's sum (default `relation`).
 * @returns The pre-activation features, $V \times G$.
 *
 * @example Per-relation and per-node normalisation
 * const g = (edges) => ({ kind: 'graph', nodes: 3, edges: edges.map(([from, to]) => ({ from, to })) })
 * const I = tensor([[1, 0], [0, 1]])
 * const relations = [{ graph: g([[0, 2]]), weight: I }, { graph: g([[0, 2], [1, 2]]), weight: I }]
 * const h = tensor([[1, 0], [0, 1], [0, 0]])
 * print('relation:', toRows(rgcnConv(h, relations, I))[2], ' node:', toRows(rgcnConv(h, relations, I, 'node'))[2])
 */
export function rgcnConv(
  h: Value,
  relations: readonly RgcnRelation[],
  selfWeight: Value,
  normalisation: RgcnNormalisation = 'relation',
): Value {
  return rgcnTerms(h, relations, normalisation).reduce((acc, t) => add(acc, t), matmul(h, selfWeight))
}

/**
 * The basis decomposition $\Wmat_r = \sum_{b=1}^{B} a_{rb} \Vmat_b$: $B$ shared bases and $R \times B$ coefficients,
 * so the number of weights does not grow with the number of relations. Differentiable in both. Throws `DomainError`
 * when a row of coefficients does not have one entry per basis.
 *
 * @param coefficients The coefficients $a_{rb}$, one row of $B$ numbers (or scalars) per relation.
 * @param bases The bases $\Vmat_b$, each $F \times G$.
 * @returns The $R$ relation weights.
 *
 * @example Two bases, three relations
 * const V1 = tensor([[1, 0], [0, 1]])
 * const V2 = tensor([[0, 1], [1, 0]])
 * for (const W of basisWeights([[0, 1], [0.5, 0], [1, -1]], [V1, V2])) print(W)
 */
export function basisWeights(coefficients: readonly (readonly Value[])[], bases: readonly Value[]): Value[] {
  if (coefficients.some((row) => row.length !== bases.length) || bases.length === 0)
    throw new DomainError('basisWeights', `basisWeights: every row needs one coefficient per basis (${bases.length})`)
  return coefficients.map((row) => row.map((a, b) => mul(a, bases[b])).reduce((x, y) => add(x, y)))
}
